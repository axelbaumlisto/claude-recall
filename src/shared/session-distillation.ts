/**
 * End-of-session distillation: turn the failures of a session into candidate
 * lessons, then let the promotion engine act on them.
 *
 * This ran only from the Claude Code Stop hook, so on a Pi host failures were
 * captured and never distilled — `candidate_lessons` stayed empty while the
 * failure memories piled up. The logic itself never needed a transcript file:
 * the detector takes entries, and everything else reads the database.
 */

import { ConfigService } from '../services/config';
import { MemoryService } from '../services/memory';
import { OutcomeStorage } from '../services/outcome-storage';
import { PromotionEngine } from '../services/promotion-engine';
import { extractHindsightHint } from '../hooks/llm-classifier';
import { DetectedFailure, detectFailures, detectTranscriptFailures } from '../hooks/failure-detectors';
import type { ToolInteraction } from '../hooks/shared';
import { hookLog, isDuplicate, safeErrorMessage, searchExisting } from '../hooks/shared';

const LOG_TAG = 'session-distillation';

/** lesson_kind values extractHindsightHint may legitimately return. */
const VALID_LESSON_KINDS = new Set([
  'rule', 'preference', 'anti_pattern', 'workflow', 'debug_fix', 'failure_preventer',
]);

/**
 * Generate candidate lessons from high-confidence failures.
 * Deduplicates against existing lessons and increments evidence count for similar ones.
 *
 * The lesson text must be failure-specific. Detectors emit a constant
 * what_should_do ("Check command syntax..."), so using it verbatim made every
 * unrelated failure "similar" to every other — evidence counts inflated across
 * unrelated failures and the promotion engine could only ever promote generic
 * boilerplate. Prefer an LLM hindsight hint; without one, ground the generic
 * remedy in what actually failed so similarity matching compares failures,
 * not the shared remedy string.
 */
export async function generateCandidateLessons(
  failures: DetectedFailure[],
  episodeId: string,
  projectId: string,
): Promise<void> {
  try {
    const outcomeStorage = OutcomeStorage.getInstance();
    // Each hint is an LLM call (via the subscription CLI it can take seconds),
    // and this loop runs INLINE in the Stop hook's ~40s budget — cap the LLM
    // calls per run; failures past the cap keep the grounded generic lesson.
    let hintBudget = 5;
    for (const f of failures) {
      if (f.confidence < 0.7) continue;

      // The grounded text already names what failed, so a repeat of the same
      // failure matches here — and then there is nothing to ask the model
      // about. Hindsight was being bought before this check, so every repeat
      // of a known failure paid for a call and threw the answer away.
      const grounded = `${f.content.what_should_do} (failure: ${f.content.what_failed})`;
      const known = outcomeStorage.findSimilarLessons(grounded, projectId);
      if (known.length > 0) {
        outcomeStorage.incrementEvidenceCount(known[0].id);
        continue;
      }

      let lessonText = grounded;
      let lessonKind = 'failure_preventer';
      let appliesWhen = extractTagsFromContext(f.content.context);

      const hint = hintBudget-- > 0 ? await extractHindsightHint(
        `${f.content.what_failed}${f.content.why_failed ? ` — ${f.content.why_failed}` : ''}`,
        f.content.context || '',
      ) : null;
      if (hint) {
        lessonText = hint.hint_text;
        if (VALID_LESSON_KINDS.has(hint.hint_kind)) {
          lessonKind = hint.hint_kind;
        }
        if (hint.applies_when.length > 0) {
          appliesWhen = hint.applies_when;
        }
      }

      const similar = outcomeStorage.findSimilarLessons(lessonText, projectId);
      if (similar.length > 0) {
        outcomeStorage.incrementEvidenceCount(similar[0].id);
      } else {
        outcomeStorage.createCandidateLesson({
          project_id: projectId,
          episode_id: episodeId,
          lesson_text: lessonText,
          lesson_kind: lessonKind,
          applies_when: appliesWhen,
          outcome_type: 'negative',
          reward_band: -1,
          confidence: f.confidence,
          durability: 'project',
        });
      }
    }
  } catch (err) {
    hookLog(LOG_TAG, `Candidate lesson generation error: ${safeErrorMessage(err)}`);
  }
}

function extractTagsFromContext(context: string): string[] {
  const tags: string[] = [];
  const words = context.toLowerCase().split(/\s+/).filter(w => w.length >= 4);
  // Take up to 5 significant words as tags
  for (const w of words) {
    if (tags.length >= 5) break;
    if (!['that', 'this', 'with', 'from', 'were', 'been'].includes(w)) {
      tags.push(w);
    }
  }
  return tags;
}

/**
 * Detect the failures a session left behind and store them as memories.
 *
 * Takes entries rather than a transcript path: Claude Code reads them from its
 * transcript file, Pi already holds them in memory, and the detector never
 * cared where they came from.
 */
export function storeDetectedFailures(
  entries: object[],
  interactions?: ToolInteraction[],
): DetectedFailure[] {
  try {
    if (entries.length === 0 && !interactions?.length) return [];

    // Pi passes interactions it already holds; Claude Code passes transcript
    // entries and lets the detector reconstruct them.
    const failures = interactions
      ? detectFailures(interactions, entries)
      : detectTranscriptFailures(entries);
    if (failures.length === 0) return [];

    const projectId = ConfigService.getInstance().getProjectId();
    const memoryService = MemoryService.getInstance();
    const stored: DetectedFailure[] = [];

    for (const failure of failures) {
      const existing = searchExisting(failure.content.what_failed.substring(0, 100));
      if (isDuplicate(failure.content.what_failed, existing, 0.6)) continue;

      memoryService.store({
        key: `hook_failure_${failure.signal}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        value: failure.content,
        type: 'failure',
        context: { projectId, timestamp: Date.now() },
      });
      stored.push(failure);
    }

    hookLog(LOG_TAG, `Stored ${stored.length} of ${failures.length} detected failure(s)`);
    return stored;
  } catch (err) {
    hookLog(LOG_TAG, `Failure detection error: ${safeErrorMessage(err)}`);
    return [];
  }
}

/**
 * Promote candidate lessons that have earned it. Never throws: a session must
 * end whether or not its lessons could be promoted.
 */
export function runPromotionCycle(projectId: string): void {
  try {
    const result = PromotionEngine.getInstance().runCycle(projectId);
    if (result.promoted > 0 || result.archived > 0) {
      hookLog(LOG_TAG, `Promotion: ${result.promoted} promoted, ${result.archived} archived`);
    }
  } catch (err) {
    hookLog(LOG_TAG, `Promotion error: ${safeErrorMessage(err)}`);
  }
}

/**
 * The whole end-of-session pass, for any runtime: detect failures, distil them
 * into candidate lessons, promote what has earned it.
 *
 * `failures` lets a caller add failures it detected by other means (Claude
 * Code's PostToolUseFailure events). Never throws.
 */
export async function distilSession(input: {
  entries: object[];
  projectId: string;
  episodeId: string;
  /** Interactions the runtime already has, when it has them (Pi). */
  interactions?: ToolInteraction[];
  failures?: DetectedFailure[];
  /**
   * Give up distilling after this long and still promote what is already
   * stored. In-process runtimes keep the process alive until this resolves, so
   * an unbounded pass delays the user's exit by one LLM call per failure.
   */
  deadlineMs?: number;
}): Promise<void> {
  const detected = storeDetectedFailures(input.entries, input.interactions);
  const all = [...detected, ...(input.failures ?? [])];

  if (all.length > 0) {
    const lessons = generateCandidateLessons(all, input.episodeId, input.projectId);
    if (input.deadlineMs === undefined) {
      await lessons;
    } else {
      let timer: NodeJS.Timeout | undefined;
      const deadline = new Promise<void>((resolve) => {
        timer = setTimeout(() => {
          hookLog(LOG_TAG, `deadline reached after ${input.deadlineMs}ms — promoting what is stored`);
          resolve();
        }, input.deadlineMs);
      });
      await Promise.race([lessons, deadline]);
      if (timer) clearTimeout(timer);
    }
  }

  runPromotionCycle(input.projectId);
}
