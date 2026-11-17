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
import { BOILERPLATE_LESSONS } from '../services/memory-quality';
import { extractHindsightHint } from '../hooks/llm-classifier';
import { DetectedFailure, detectFailures, detectTranscriptFailures } from '../hooks/failure-detectors';
import type { ToolInteraction } from '../hooks/shared';
import { hookLog, isDuplicate, jaccardSimilarity, safeErrorMessage, searchExisting } from '../hooks/shared';

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
  opts: {
    /** Aborted when the caller has run out of time: no further failure is taken up. */
    stop?: AbortSignal;
    /**
     * Record no lesson for a failure the model had nothing to say about and
     * whose remedy is still the detector's default. Such a lesson is the
     * failure restated — "check prerequisites (failure: X)" — and once promoted
     * it is a rule loaded every turn that says only what the failure memory
     * already says. A remedy enriched by a paired fix is kept.
     */
    skipBoilerplate?: boolean;
  } = {},
): Promise<void> {
  try {
    const outcomeStorage = OutcomeStorage.getInstance();
    // Each hint is an LLM call (via the subscription CLI it can take seconds),
    // and this loop runs INLINE in the Stop hook's ~40s budget — cap the LLM
    // calls per run; failures past the cap keep the grounded generic lesson.
    let hintBudget = 5;
    for (const f of failures) {
      // Checked per failure, not per await: a call already in flight is paid
      // for, so its answer is still recorded — but nothing new is started.
      if (opts.stop?.aborted) break;
      if (f.confidence < 0.7) continue;

      let lessonText = `${f.content.what_should_do} (failure: ${f.content.what_failed})`;
      let lessonKind = 'failure_preventer';
      let appliesWhen = extractTagsFromContext(f.content.context);

      const hint = hintBudget-- > 0 ? await extractHindsightHint(
        `${f.content.what_failed}${f.content.why_failed ? ` — ${f.content.why_failed}` : ''}`,
        f.content.context || '',
      ) : null;
      if (!hint && opts.skipBoilerplate
        && BOILERPLATE_LESSONS.has(f.content.what_should_do.trim().toLowerCase())) {
        continue;
      }
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

/** Similarity above which a detected failure is one already stored. */
const KNOWN_FAILURE_THRESHOLD = 0.6;

/** `what_failed` of a stored failure, wherever its writer put it. */
function storedWhatFailed(value: unknown): string | undefined {
  const v = value as { what_failed?: unknown; content?: { what_failed?: unknown } } | null;
  if (typeof v?.what_failed === 'string') return v.what_failed; // captured as the tool failed
  if (typeof v?.content?.what_failed === 'string') return v.content.what_failed; // written by storeNew()
  return undefined;
}

/**
 * Whether a detected failure is already on record.
 *
 * `isDuplicate` compares against the stored value as one JSON blob, where the
 * failing command is a small part of field names, remedies and context: the
 * same failure scored under the threshold and was stored again, at every
 * session end. Stored failures say what failed in a field of their own, so
 * that is compared first; `isDuplicate` still catches whatever it caught.
 */
function alreadyOnRecord(whatFailed: string): boolean {
  const existing = searchExisting(whatFailed.substring(0, 100));
  return isDuplicate(whatFailed, existing, KNOWN_FAILURE_THRESHOLD)
    || existing.some((memory) => {
      const stored = storedWhatFailed(memory.value);
      return stored !== undefined && jaccardSimilarity(whatFailed, stored) >= KNOWN_FAILURE_THRESHOLD;
    });
}

/**
 * Run a detector, store the failures not yet on record, and hand back every
 * failure it found.
 *
 * All of them, not only the new ones: storing and learning are different
 * questions. A failure seen before must not be stored twice, but a repeat is
 * exactly what a candidate lesson gains evidence from.
 */
function storeNew(detect: () => DetectedFailure[]): DetectedFailure[] {
  try {
    const failures = detect();
    if (failures.length === 0) return [];

    const projectId = ConfigService.getInstance().getProjectId();
    const memoryService = MemoryService.getInstance();
    let stored = 0;

    for (const failure of failures) {
      if (alreadyOnRecord(failure.content.what_failed)) continue;

      memoryService.store({
        key: `hook_failure_${failure.signal}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        value: {
          title: `Avoid: ${failure.content.what_failed.substring(0, 50)}`,
          description: failure.content.why_failed.substring(0, 100),
          content: failure.content,
        },
        type: 'failure',
        context: { projectId, timestamp: Date.now() },
        relevanceScore: failure.confidence,
      });
      stored++;
    }

    hookLog(LOG_TAG, `Stored ${stored} failure(s) from ${failures.length} detected`);
    return failures;
  } catch (err) {
    hookLog(LOG_TAG, `Failure detection error: ${safeErrorMessage(err)}`);
    return [];
  }
}

/**
 * Detect the failures in a transcript and store the new ones — what Claude
 * Code's Stop hook does with the tail of its transcript file.
 */
export function storeDetectedFailures(entries: object[]): DetectedFailure[] {
  return entries.length === 0 ? [] : storeNew(() => detectTranscriptFailures(entries));
}

export interface PromotionResult {
  promoted: number;
  archived: number;
}

/**
 * Promote candidate lessons that have earned it. Never throws: a session must
 * end whether or not its lessons could be promoted.
 */
export function runPromotionCycle(projectId: string): PromotionResult {
  try {
    const result = PromotionEngine.getInstance().runCycle(projectId);
    if (result.promoted > 0 || result.archived > 0) {
      hookLog(LOG_TAG, `Promotion: ${result.promoted} promoted, ${result.archived} archived`);
    }
    return result;
  } catch (err) {
    hookLog(LOG_TAG, `Promotion error: ${safeErrorMessage(err)}`);
    return { promoted: 0, archived: 0 };
  }
}

/**
 * The end-of-session pass for a runtime that holds its tool calls in memory:
 * detect the session's failures, distil them into candidate lessons, then
 * promote — once, after the lessons exist, so what a session taught can be
 * promoted by that session. Never throws.
 *
 * Claude Code composes the same three steps itself: it reads a transcript
 * rather than holding interactions, and needs the failure count in between.
 */
export async function distilSession(input: {
  projectId: string;
  /** The episode this session already opened; lessons are filed under it. */
  episodeId: string;
  interactions?: ToolInteraction[];
  /**
   * Stop distilling after this long and promote what is already stored.
   * In-process runtimes keep the process alive while this works, so an
   * unbounded pass delays the user's exit by one LLM call per failure. At the
   * deadline no further failure is taken up; a call already in flight is left
   * to finish, so the exit is late by at most that one call.
   */
  deadlineMs?: number;
}): Promise<PromotionResult> {
  const { interactions } = input;
  const failures = interactions?.length ? storeNew(() => detectFailures(interactions)) : [];

  if (failures.length > 0) {
    const stop = new AbortController();
    const lessons = generateCandidateLessons(failures, input.episodeId, input.projectId, {
      stop: stop.signal,
      // Without a model there is often nothing here but the detector's default
      // remedy, and on a Pi host that is the usual case, not the exception.
      skipBoilerplate: true,
    });
    if (input.deadlineMs === undefined) {
      await lessons;
    } else {
      let timer: NodeJS.Timeout | undefined;
      const deadline = new Promise<void>((resolve) => {
        timer = setTimeout(() => {
          hookLog(LOG_TAG, `deadline reached after ${input.deadlineMs}ms — promoting what is stored`);
          stop.abort();
          resolve();
        }, input.deadlineMs);
      });
      await Promise.race([lessons, deadline]);
      clearTimeout(timer);
    }
  }

  return runPromotionCycle(input.projectId);
}
