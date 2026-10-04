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
jest.mock('../../src/hooks/failure-detectors', () => ({ detectTranscriptFailures: mockDetect }));
jest.mock('../../src/hooks/shared', () => ({
	hookLog: jest.fn(),
	safeErrorMessage: (e: any) => String(e),
	searchExisting: mockSearchExisting,
	isDuplicate: jest.requireActual('../../src/hooks/shared').isDuplicate,
	comparableText: jest.requireActual('../../src/hooks/shared').comparableText,
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
		content: { what_failed: what, what_should_do: 'check the command', context: 'running tests' },
	});

	beforeEach(() => {
		jest.clearAllMocks();
		mockFindSimilar.mockReturnValue([]);
		mockSearchExisting.mockReturnValue([]);
		mockHint.mockResolvedValue(null);
		mockRunCycle.mockReturnValue({ promoted: 0, archived: 0 });
	});

	describe('storeDetectedFailures', () => {
		it('stores what the detector found, as failure memories', () => {
			mockDetect.mockReturnValue([failure('npm test exited 1')]);

			const stored = storeDetectedFailures([{ role: 'assistant' }]);

			expect(stored).toHaveLength(1);
			expect(mockStore).toHaveBeenCalledTimes(1);
			expect(mockStore.mock.calls[0][0].type).toBe('failure');
		});

		it('recognises a repeat of a detector-written failure, not just a captured one', () => {
			// Stored failures carry { what_failed, why_failed, … }; comparing against
			// the whole JSON blob used to drown the text and let repeats through.
			mockDetect.mockReturnValue([failure('npm test exited 1')]);
			mockSearchExisting.mockReturnValue([
				{ value: { what_failed: 'npm test exited 1', why_failed: 'Exit code 1', context: 'tests' }, score: 1 },
			]);

			expect(storeDetectedFailures([{ role: 'assistant' }])).toHaveLength(0);
			expect(mockStore).not.toHaveBeenCalled();
		});

		it('skips a failure already stored', () => {
			mockDetect.mockReturnValue([failure('npm test exited 1')]);
			mockSearchExisting.mockReturnValue([{ value: { what_failed: 'npm test exited 1' }, score: 1 }]);

			expect(storeDetectedFailures([{ role: 'assistant' }])).toHaveLength(0);
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
		it('turns a detected failure into a candidate lesson, then promotes', async () => {
			mockDetect.mockReturnValue([failure('npm test exited 1')]);

			await distilSession({ entries: [{ role: 'assistant' }], projectId: 'proj', episodeId: 'ep1' });

			expect(mockCreateLesson).toHaveBeenCalledTimes(1);
			expect(mockCreateLesson.mock.calls[0][0]).toMatchObject({ project_id: 'proj', episode_id: 'ep1' });
			expect(mockRunCycle).toHaveBeenCalledWith('proj');
		});

		it('prefers the model’s hindsight over the detector’s generic remedy', async () => {
			mockDetect.mockReturnValue([failure('npm test exited 1')]);
			mockHint.mockResolvedValue({ hint_text: 'run the suite before claiming done', hint_kind: 'rule', applies_when: ['tests'] });

			await distilSession({ entries: [{ role: 'assistant' }], projectId: 'proj', episodeId: 'ep1' });

			expect(mockCreateLesson.mock.calls[0][0]).toMatchObject({
				lesson_text: 'run the suite before claiming done',
				lesson_kind: 'rule',
				applies_when: ['tests'],
			});
		});

		it('does not pay for hindsight about a failure it already learned from', async () => {
			mockDetect.mockReturnValue([failure('npm test exited 1')]);
			mockFindSimilar.mockReturnValue([{ id: 'lesson-1' }]);

			await distilSession({ entries: [{ role: 'assistant' }], projectId: 'proj', episodeId: 'ep1' });

			// The repeat counts as evidence, and the model is never asked.
			expect(mockIncrementEvidence).toHaveBeenCalledWith('lesson-1');
			expect(mockHint).not.toHaveBeenCalled();
		});

		it('asks the model only about a failure it has not seen', async () => {
			mockDetect.mockReturnValue([failure('a brand new failure')]);
			mockFindSimilar.mockReturnValue([]);

			await distilSession({ entries: [{ role: 'assistant' }], projectId: 'proj', episodeId: 'ep1' });

			expect(mockHint).toHaveBeenCalledTimes(1);
		});

		it('counts evidence instead of duplicating an existing lesson', async () => {
			mockDetect.mockReturnValue([failure('npm test exited 1')]);
			mockFindSimilar.mockReturnValue([{ id: 'lesson-1' }]);

			await distilSession({ entries: [{ role: 'assistant' }], projectId: 'proj', episodeId: 'ep1' });

			expect(mockIncrementEvidence).toHaveBeenCalledWith('lesson-1');
			expect(mockCreateLesson).not.toHaveBeenCalled();
		});

		it('ignores low-confidence detections', async () => {
			mockDetect.mockReturnValue([failure('maybe something broke', 0.3)]);

			await distilSession({ entries: [{ role: 'assistant' }], projectId: 'proj', episodeId: 'ep1' });

			expect(mockCreateLesson).not.toHaveBeenCalled();
		});

		it('still runs promotion when the session had no failures', async () => {
			mockDetect.mockReturnValue([]);

			await distilSession({ entries: [{ role: 'assistant' }], projectId: 'proj', episodeId: 'ep1' });

			expect(mockRunCycle).toHaveBeenCalledWith('proj');
		});

		it('accepts failures the caller detected by other means', async () => {
			mockDetect.mockReturnValue([]);

			await distilSession({
				entries: [],
				projectId: 'proj',
				episodeId: 'ep1',
				failures: [failure('tool reported an error') as any],
			});

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
