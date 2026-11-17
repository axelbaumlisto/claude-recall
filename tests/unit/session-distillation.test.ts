const mockStore = jest.fn();
const mockDetect = jest.fn();
const mockSearchExisting = jest.fn().mockReturnValue([]);
const mockRunCycle = jest.fn().mockReturnValue({ promoted: 0, archived: 0 });
const mockCreateLesson = jest.fn();
const mockFindSimilar = jest.fn().mockReturnValue([]);
const mockIncrementEvidence = jest.fn();
const mockHint = jest.fn().mockResolvedValue(null);

jest.mock('../../src/services/memory', () => ({
  MemoryService: { getInstance: () => ({ store: mockStore }) },
}));
jest.mock('../../src/services/config', () => ({
  ConfigService: { getInstance: () => ({ getProjectId: () => 'proj' }) },
}));
jest.mock('../../src/services/outcome-storage', () => ({
  OutcomeStorage: {
    getInstance: () => ({
      createCandidateLesson: mockCreateLesson,
      findSimilarLessons: mockFindSimilar,
      incrementEvidenceCount: mockIncrementEvidence,
    }),
  },
}));
jest.mock('../../src/services/promotion-engine', () => ({
  PromotionEngine: { getInstance: () => ({ runCycle: mockRunCycle }) },
}));
jest.mock('../../src/hooks/llm-classifier', () => ({ extractHindsightHint: mockHint }));
jest.mock('../../src/hooks/failure-detectors', () => ({ detectTranscriptFailures: mockDetect, detectFailures: mockDetect }));
jest.mock('../../src/hooks/shared', () => ({
  hookLog: jest.fn(),
  safeErrorMessage: (e: any) => String(e),
  searchExisting: mockSearchExisting,
  isDuplicate: jest.requireActual('../../src/hooks/shared').isDuplicate,
  jaccardSimilarity: jest.requireActual('../../src/hooks/shared').jaccardSimilarity,
}));

import { distilSession, runPromotionCycle, storeDetectedFailures } from '../../src/shared/session-distillation';

/**
 * This pass used to live inside the Claude Code Stop hook, so a Pi host
 * captured failures and never distilled them — candidate_lessons stayed empty
 * while the failure memories piled up and were injected every turn.
 */
describe('session distillation', () => {
  const failure = (what: string, confidence = 0.9) => ({
    signal: 'non-zero-exit',
    confidence,
    content: { what_failed: what, why_failed: 'Exit code 1', what_should_do: 'check the command', context: 'running tests' },
  });

  beforeEach(() => {
    jest.clearAllMocks();
    mockFindSimilar.mockReturnValue([]);
    mockSearchExisting.mockReturnValue([]);
    mockHint.mockResolvedValue(null);
    mockRunCycle.mockReturnValue({ promoted: 0, archived: 0 });
  });

  describe('storeDetectedFailures', () => {
    it('stores a new failure in the shape the Stop hook has always written', () => {
      mockDetect.mockReturnValue([{
        signal: 'non-zero-exit',
        confidence: 0.85,
        content: { what_failed: 'npm test exited 1', why_failed: 'Exit code 1: 3 failing', what_should_do: 'check', context: 'tests' },
      }]);

      storeDetectedFailures([{ role: 'assistant' }]);

      // Other readers of these rows expect this layout; moving the code into a
      // shared module is not a licence to change what it writes.
      expect(mockStore).toHaveBeenCalledTimes(1);
      expect(mockStore.mock.calls[0][0]).toMatchObject({
        type: 'failure',
        relevanceScore: 0.85,
        value: {
          title: 'Avoid: npm test exited 1',
          description: 'Exit code 1: 3 failing',
          content: { what_failed: 'npm test exited 1' },
        },
      });
    });

    it('returns every failure it detected, including the ones already on record', () => {
      // Storing and learning are different questions. A repeat must not be
      // stored twice, but it is exactly what a lesson gains evidence from.
      mockDetect.mockReturnValue([failure('npm test exited 1'), failure('cargo build exited 101')]);
      mockSearchExisting.mockImplementation((query: string) =>
        query.startsWith('npm test') ? [{ value: { what_failed: 'npm test exited 1' }, score: 1 }] : []);

      const detected = storeDetectedFailures([{ role: 'assistant' }]);

      expect(detected.map(f => f.content.what_failed)).toEqual(['npm test exited 1', 'cargo build exited 101']);
      expect(mockStore).toHaveBeenCalledTimes(1);
    });

    it.each([
      ['captured as the tool failed', { content: 'npm test exited 1. Exit code 1', what_failed: 'npm test exited 1', why_failed: 'Exit code 1', what_should_do: 'Check command syntax, file paths, and prerequisites before running', context: 'Bash command returned non-zero exit code 1', preventative_checks: ['Verify tool inputs are correct'] }],
      ['written by an earlier detector pass', { title: 'Avoid: npm test exited 1', description: 'Exit code 1', content: { what_failed: 'npm test exited 1', why_failed: 'Exit code 1', context: 'tests' } }],
    ])('does not store a failure again when it was %s', (_how, stored) => {
      mockDetect.mockReturnValue([failure('npm test exited 1')]);
      mockSearchExisting.mockReturnValue([{ value: stored, score: 1 }]);

      storeDetectedFailures([{ role: 'assistant' }]);

      expect(mockStore).not.toHaveBeenCalled();
    });

    it('does nothing for a session with no entries', () => {
      expect(storeDetectedFailures([])).toEqual([]);
      expect(mockDetect).not.toHaveBeenCalled();
    });

    it('swallows a detector failure rather than breaking session end', () => {
      mockDetect.mockImplementation(() => { throw new Error('boom'); });

      expect(() => storeDetectedFailures([{ role: 'assistant' }])).not.toThrow();
    });
  });

  describe('distilSession', () => {
    /** A session that made tool calls; what they were is the detector's business. */
    const session = [{ call: { id: 't1', name: 'bash', input: {}, entryIndex: 0 } }] as any;

    it('turns a detected failure into a candidate lesson, then promotes', async () => {
      mockDetect.mockReturnValue([failure('npm test exited 1')]);

      await distilSession({ interactions: session, projectId: 'proj', episodeId: 'ep1' });

      expect(mockCreateLesson).toHaveBeenCalledTimes(1);
      expect(mockCreateLesson.mock.calls[0][0]).toMatchObject({ project_id: 'proj', episode_id: 'ep1' });
      expect(mockRunCycle).toHaveBeenCalledWith('proj');
    });

    it('prefers the model’s hindsight over the detector’s generic remedy', async () => {
      mockDetect.mockReturnValue([failure('npm test exited 1')]);
      mockHint.mockResolvedValue({ hint_text: 'run the suite before claiming done', hint_kind: 'rule', applies_when: ['tests'] });

      await distilSession({ interactions: session, projectId: 'proj', episodeId: 'ep1' });

      expect(mockCreateLesson.mock.calls[0][0]).toMatchObject({
        lesson_text: 'run the suite before claiming done',
        lesson_kind: 'rule',
        applies_when: ['tests'],
      });
    });

    it('counts evidence instead of duplicating an existing lesson', async () => {
      mockDetect.mockReturnValue([failure('npm test exited 1')]);
      mockFindSimilar.mockReturnValue([{ id: 'lesson-1' }]);

      await distilSession({ interactions: session, projectId: 'proj', episodeId: 'ep1' });

      expect(mockIncrementEvidence).toHaveBeenCalledWith('lesson-1');
      expect(mockCreateLesson).not.toHaveBeenCalled();
    });

    it('ignores low-confidence detections', async () => {
      mockDetect.mockReturnValue([failure('maybe something broke', 0.3)]);

      await distilSession({ interactions: session, projectId: 'proj', episodeId: 'ep1' });

      expect(mockCreateLesson).not.toHaveBeenCalled();
    });

    it('learns from a failure even when its memory is already on record', async () => {
      // Under Pi every failure is captured as the tool fails, so by session end
      // each one is "already stored". That must not cost it its lesson.
      mockDetect.mockReturnValue([failure('npm test exited 1')]);
      mockSearchExisting.mockReturnValue([{ value: { what_failed: 'npm test exited 1' }, score: 1 }]);

      await distilSession({ interactions: session, projectId: 'proj', episodeId: 'ep1' });

      expect(mockStore).not.toHaveBeenCalled();
      expect(mockCreateLesson).toHaveBeenCalledTimes(1);
    });

    it('still runs promotion when the session had no failures', async () => {
      mockDetect.mockReturnValue([]);

      await distilSession({ interactions: session, projectId: 'proj', episodeId: 'ep1' });

      expect(mockRunCycle).toHaveBeenCalledWith('proj');
    });

    it('is the promotion cycle alone for a runtime that kept no tool calls', async () => {
      mockRunCycle.mockReturnValue({ promoted: 2, archived: 1 });

      await expect(distilSession({ projectId: 'proj', episodeId: 'ep1' }))
        .resolves.toEqual({ promoted: 2, archived: 1 });

      expect(mockDetect).not.toHaveBeenCalled();
      expect(mockCreateLesson).not.toHaveBeenCalled();
    });
  });

  describe('the deadline', () => {
    afterEach(() => jest.useRealTimers());

    it('takes up no further failure once the deadline has passed', async () => {
      // The session may exit at the deadline, but work that carries on behind
      // it keeps the process alive and keeps paying for model calls.
      jest.useFakeTimers();
      let answerFirst: (hint: null) => void = () => {};
      mockHint.mockImplementationOnce(() => new Promise(resolve => { answerFirst = resolve; }));
      mockDetect.mockReturnValue(['a', 'b', 'c'].map(name => failure(`command ${name} exited 1`)));
      const session = [{ call: { id: 't1', name: 'bash', input: {}, entryIndex: 0 } }] as any;

      const done = distilSession({ interactions: session, projectId: 'proj', episodeId: 'ep', deadlineMs: 1000 });
      await jest.advanceTimersByTimeAsync(1000);
      await done;
      expect(mockRunCycle).toHaveBeenCalledTimes(1);

      // The call that was in flight comes back: its lesson is kept...
      answerFirst(null);
      await jest.advanceTimersByTimeAsync(0);

      // ...and the two failures behind it are never asked about.
      expect(mockHint).toHaveBeenCalledTimes(1);
      expect(mockCreateLesson).toHaveBeenCalledTimes(1);
    });
  });

  describe('runPromotionCycle', () => {
    it('never throws when promotion fails', () => {
      mockRunCycle.mockImplementation(() => { throw new Error('db locked'); });

      expect(() => runPromotionCycle('proj')).not.toThrow();
    });
  });
});
