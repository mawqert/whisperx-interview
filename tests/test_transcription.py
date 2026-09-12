import io
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import ANY, Mock, patch

import app
import transcribe_worker as worker
import run_logging


class LoggingTests(unittest.TestCase):
    def test_diarization_progress_is_throttled_and_completes(self):
        with redirect_stdout(io.StringIO()) as output:
            report = worker.diarization_progress()
            for percent in (0, 1, 2, 5, 50, 51, 99, 100):
                report(percent)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 5)
        self.assertIn('step=segmentation', lines[0])
        self.assertIn('step=embeddings/clustering', lines[2])
        self.assertIn('step=complete', lines[-1])

    def test_diarizer_setup_does_not_probe_embedding_model(self):
        from runtime_setup import prepare_diarizer

        class Pipeline:
            segmentation_batch_size = 32
            embedding_batch_size = 32

            @property
            def _embedding(self):
                raise AssertionError('Setup must not initialize embedding probes')

        prepare_diarizer(types.SimpleNamespace(model=Pipeline()))

    def test_worker_streams_both_outputs_and_redacts_token(self):
        output, retained = io.StringIO(), io.StringIO()
        command = [sys.executable, '-c',
                   'import os, sys; print(os.environ["HF_TOKEN"]); print("error detail", file=sys.stderr)']
        with redirect_stdout(output):
            result = app.run_worker(command, stdout=retained, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'HF_TOKEN': 'test-secret'}, timeout=5)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(output.getvalue(), retained.getvalue())
        self.assertIn('[redacted]', output.getvalue())
        self.assertIn('error detail', output.getvalue())
        self.assertNotIn('test-secret', output.getvalue())

    def test_worker_timeout_stops_process(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            app.run_worker([sys.executable, '-c', 'import time; time.sleep(10)'],
                           stdout=io.StringIO(), stderr=subprocess.STDOUT,
                           env=os.environ.copy(), timeout=0.1)

    def test_gpu_metrics_are_optional(self):
        with patch.object(run_logging.subprocess, 'run', side_effect=FileNotFoundError), \
                redirect_stdout(io.StringIO()) as output:
            run_logging.log_load()
        self.assertIn('statistics unavailable', output.getvalue())


class SpeakerBlockTests(unittest.TestCase):
    def test_correction_inside_one_segment_preserves_three_turns(self):
        segments = [{'start': 0, 'end': 3, 'speaker': 'candidate',
                     'text': 'filter Нет. sorted', 'words': [
                         {'word': 'filter', 'start': 0, 'end': 1, 'speaker': 'candidate'},
                         {'word': 'Нет.', 'start': 1, 'end': 2, 'speaker': 'interviewer'},
                         {'word': 'sorted', 'start': 2, 'end': 3, 'speaker': 'candidate'}]}]
        blocks = app.speaker_blocks(segments)
        self.assertEqual([b['speaker'] for b in blocks], ['candidate', 'interviewer', 'candidate'])
        self.assertEqual([b['text'] for b in blocks], ['filter', 'Нет.', 'sorted'])
        self.assertEqual([(b['start'], b['end']) for b in blocks], [(0, 1), (1, 2), (2, 3)])

    def test_same_speaker_segments_stay_separate(self):
        segments = [{'start': i, 'end': i + 1, 'speaker': 'A', 'text': 'Ответ.'}
                    for i in range(3)]
        self.assertEqual(len(app.speaker_blocks(segments)), 3)

    def test_incomplete_alignment_preserves_all_text(self):
        segment = {'start': 0, 'end': 2, 'speaker': 'A', 'text': 'Запускаем Kubernetes.',
                   'words': [{'word': 'Запускаем', 'start': 0, 'end': 1, 'speaker': 'A'}]}
        self.assertEqual(app.speaker_blocks([segment])[0]['text'], segment['text'])

    def test_unaligned_word_is_kept_without_guessing_speaker(self):
        segment = {'start': 0, 'end': 2, 'speaker': 'A', 'text': 'Это API', 'words': [
            {'word': 'Это', 'start': 0, 'end': 1, 'speaker': 'A'}, {'word': 'API'}]}
        blocks = app.speaker_blocks([segment])
        self.assertEqual([b['text'] for b in blocks], ['Это', 'API'])
        self.assertIsNone(blocks[1]['speaker'])

    def test_long_speech_and_pauses_split_blocks(self):
        words = [{'word': 'слово', 'start': i, 'end': i + 1, 'speaker': 'A'}
                 for i in range(65)]
        segment = {'start': 0, 'end': 65, 'text': ' '.join(w['word'] for w in words), 'words': words}
        blocks = app.speaker_blocks([segment])
        self.assertEqual(len(blocks), 3)
        self.assertTrue(all(b['end'] - b['start'] <= 30 for b in blocks))
        words[1]['start'] = 4
        words[1]['end'] = 5
        segment.update(words=words[:2], text='слово слово', end=5)
        self.assertEqual(len(app.speaker_blocks([segment])), 2)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.wx = types.ModuleType('whisperx')
        self.audio = object()
        self.raw = {'segments': [{'text': 'Запускаем Docker', 'start': 0, 'end': 1}]}
        self.model = Mock()
        self.model.transcribe.return_value = self.raw
        self.wx.load_audio = Mock(return_value=self.audio)
        self.wx.load_model = Mock(return_value=self.model)
        self.wx.load_align_model = Mock(return_value=('aligner', {'language': 'ru'}))
        self.aligned = {**self.raw, 'word_segments': []}
        self.wx.align = Mock(return_value=self.aligned)
        self.wx.assign_word_speakers = Mock(return_value=self.aligned)
        self.diarizer = Mock(return_value='speaker-segments')
        self.diarize = types.ModuleType('whisperx.diarize')
        self.diarize.DiarizationPipeline = Mock(return_value=self.diarizer)
        self.torch = types.ModuleType('torch')
        self.torch.cuda = types.SimpleNamespace(empty_cache=Mock())
        modules = patch.dict(sys.modules, {'torch': self.torch, 'whisperx': self.wx,
                                          'whisperx.diarize': self.diarize})
        modules.start()
        self.addCleanup(modules.stop)
        runtime = patch.object(worker, 'load_runtime', return_value=(self.torch, self.wx, self.diarize.DiarizationPipeline))
        runtime.start()
        self.addCleanup(runtime.stop)

    def test_pipeline_reuses_audio_and_preserves_settings(self):
        result = worker.transcribe_audio('audio.wav', token='secret', min_speakers=2, max_speakers=3)
        self.wx.load_model.assert_called_once_with(
            'large-v3', device='cuda', compute_type='int8_float16',
            language='ru', task='transcribe', use_auth_token='secret')
        self.wx.load_audio.assert_called_once_with('audio.wav')
        self.model.transcribe.assert_called_once_with(self.audio, batch_size=4)
        self.wx.align.assert_called_once_with(self.raw['segments'], 'aligner',
                                             {'language': 'ru'}, self.audio, 'cuda',
                                             return_char_alignments=False)
        self.diarizer.assert_called_once_with(self.audio, min_speakers=2, max_speakers=3,
                                              progress_callback=ANY)
        self.wx.assign_word_speakers.assert_called_once_with('speaker-segments', self.aligned)
        self.assertEqual(result['language'], 'ru')
        self.assertEqual(self.torch.cuda.empty_cache.call_count, 3)

    def test_silence_skips_alignment_and_diarization(self):
        self.model.transcribe.return_value = {'segments': []}
        result = worker.transcribe_audio('silence.wav', token='secret')
        self.assertEqual(result['segments'], [])
        self.assertEqual(result['word_segments'], [])
        self.assertEqual(result['language'], 'ru')
        self.wx.load_align_model.assert_not_called()
        self.diarize.DiarizationPipeline.assert_not_called()

    def test_full_model_uses_converted_alias_and_small_batch(self):
        result = worker.transcribe_audio('audio.wav', token='secret')
        self.wx.load_model.assert_called_once_with(
            'large-v3', device='cuda', compute_type='int8_float16', language='ru',
            task='transcribe', use_auth_token='secret')
        self.model.transcribe.assert_called_once_with(self.audio, batch_size=4)
        self.assertEqual(result['run_info']['model'], 'large-v3')
        self.assertEqual(result['asr_segments'], self.raw['segments'])
        self.assertIsNot(result['asr_segments'], self.raw['segments'])

    def test_diarization_preserves_model_defaults(self):
        self.diarizer.model.segmentation_batch_size = 32
        self.diarizer.model.embedding_batch_size = 32
        worker.transcribe_audio('audio.wav', token='secret')
        self.assertEqual(self.diarizer.model.segmentation_batch_size, 32)
        self.assertEqual(self.diarizer.model.embedding_batch_size, 32)

    def test_stage_failure_releases_memory_and_stops_pipeline(self):
        self.wx.align.side_effect = RuntimeError('alignment failed')
        with self.assertRaisesRegex(RuntimeError, 'alignment failed'):
            worker.transcribe_audio('audio.wav', token='secret')
        self.assertEqual(self.torch.cuda.empty_cache.call_count, 2)
        self.diarize.DiarizationPipeline.assert_not_called()


class EndpointTests(unittest.TestCase):
    def test_model_selector_removed(self):
        page = app.app.test_client().get('/')
        self.assertEqual(page.status_code, 200)
        self.assertNotIn(b'<select', page.data)
        self.assertNotIn(b'codeswitch', page.data)

    def post(self):
        return app.app.test_client().post('/transcribe', data={
            'file': (io.BytesIO(b'audio'), 'audio.mp3'), 'min_speakers': '2'})

    @patch.dict(os.environ, {'HF_TOKEN': 'secret'})
    def test_json_job_and_offline_retry(self):
        calls = []

        def run(command, **kwargs):
            calls.append(kwargs['env'].copy())
            self.assertEqual(len(command), 3)
            self.assertNotIn('secret', command)
            job = json.loads(Path(command[2]).read_text())
            self.assertEqual(job['min_speakers'], 2)
            self.assertNotIn('model_name', job)
            self.assertNotIn('token', job)
            if len(calls) == 1:
                kwargs['stdout'].write('huggingface.co ConnectionError')
                return types.SimpleNamespace(returncode=1)
            Path(job['output_path']).write_text(json.dumps({'language': 'ru', 'segments': [
                {'text': ' Docker ', 'start': 0, 'end': 1, 'speaker': 'SPEAKER_00'}]}))
            return types.SimpleNamespace(returncode=0)

        with patch.object(app, 'run_worker', side_effect=run):
            response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['blocks'][0]['text'], 'Docker')
        self.assertEqual(calls[1]['HF_HUB_OFFLINE'], '1')
        self.assertFalse(app.transcribing.locked())

    @patch.dict(os.environ, {'HF_TOKEN': 'secret'})
    def test_timeout_releases_lock(self):
        with patch.object(app, 'run_worker', side_effect=subprocess.TimeoutExpired('worker', 1)):
            response = self.post()
        self.assertEqual(response.status_code, 504)
        self.assertFalse(app.transcribing.locked())


if __name__ == '__main__':
    unittest.main()
