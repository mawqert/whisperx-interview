import io
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch

import app
import transcribe_worker as worker


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
        download = patch.object(worker, 'snapshot_download', return_value='/cache/snapshot')
        self.download = download.start()
        self.addCleanup(download.stop)

    def test_pipeline_reuses_audio_and_preserves_settings(self):
        result = worker.transcribe_audio('audio.wav', token='secret', min_speakers=2, max_speakers=3)
        self.download.assert_called_once_with(repo_id=worker.MODEL_REPO,
                                              allow_patterns='ct2_int8_float16/*')
        self.wx.load_model.assert_called_once_with(
            '/cache/snapshot/ct2_int8_float16', device='cuda', compute_type='int8',
            language='ru', task='transcribe', use_auth_token='secret')
        self.wx.load_audio.assert_called_once_with('audio.wav')
        self.model.transcribe.assert_called_once_with(self.audio, batch_size=8)
        self.wx.align.assert_called_once_with(self.raw['segments'], 'aligner',
                                             {'language': 'ru'}, self.audio, 'cuda',
                                             return_char_alignments=False)
        self.diarizer.assert_called_once_with(self.audio, min_speakers=2, max_speakers=3)
        self.wx.assign_word_speakers.assert_called_once_with('speaker-segments', self.aligned)
        self.assertEqual(result['language'], 'ru')
        self.assertEqual(self.torch.cuda.empty_cache.call_count, 3)

    def test_silence_skips_alignment_and_diarization(self):
        self.model.transcribe.return_value = {'segments': []}
        self.assertEqual(worker.transcribe_audio('silence.wav', token='secret'),
                         {'segments': [], 'word_segments': [], 'language': 'ru'})
        self.wx.load_align_model.assert_not_called()
        self.diarize.DiarizationPipeline.assert_not_called()

    def test_stage_failure_releases_memory_and_stops_pipeline(self):
        self.wx.align.side_effect = RuntimeError('alignment failed')
        with self.assertRaisesRegex(RuntimeError, 'alignment failed'):
            worker.transcribe_audio('audio.wav', token='secret')
        self.assertEqual(self.torch.cuda.empty_cache.call_count, 2)
        self.diarize.DiarizationPipeline.assert_not_called()


class EndpointTests(unittest.TestCase):
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
            self.assertNotIn('token', job)
            if len(calls) == 1:
                kwargs['stdout'].write('huggingface.co ConnectionError')
                return types.SimpleNamespace(returncode=1)
            Path(job['output_path']).write_text(json.dumps({'language': 'ru', 'segments': [
                {'text': ' Docker ', 'start': 0, 'end': 1, 'speaker': 'SPEAKER_00'}]}))
            return types.SimpleNamespace(returncode=0)

        with patch.object(app.subprocess, 'run', side_effect=run):
            response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['blocks'][0]['text'], 'Docker')
        self.assertEqual(calls[1]['HF_HUB_OFFLINE'], '1')
        self.assertFalse(app.transcribing.locked())

    @patch.dict(os.environ, {'HF_TOKEN': 'secret'})
    def test_timeout_releases_lock(self):
        with patch.object(app.subprocess, 'run', side_effect=subprocess.TimeoutExpired('worker', 1)):
            response = self.post()
        self.assertEqual(response.status_code, 504)
        self.assertFalse(app.transcribing.locked())


if __name__ == '__main__':
    unittest.main()
