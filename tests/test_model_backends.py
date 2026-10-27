"""Verify the backend wire contract and a whole study against a local HTTP fixture."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from rsi_researcher.runtime.client import usage_summary, object_schema
from rsi_researcher.runtime.models import create_model, model_description
from rsi_researcher.research.benchmark import run_study
from tests.test_observed_study import coat_fixture


class ModelBackendTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.requests = []
        self.responses = []
        outer = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_GET(self):
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps({'data': [{'id': 'qwen3-8b'}]}).encode())
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                outer.requests.append((self.path, payload, dict(self.headers)))
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps({'model': 'qwen3-8b', 'choices': [
                    {'finish_reason': 'stop', 'message': {'content': json.dumps(outer.responses.pop(0), ensure_ascii=False)}}],
                    'usage': {'prompt_tokens': 20, 'completion_tokens': 10}}).encode())
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.config = {'llm': {'provider': 'vllm',
            'base_url': f'http://127.0.0.1:{self.server.server_port}/v1', 'model': 'qwen3-8b'}}

    def test_vllm_needs_no_deepseek_secret_and_uses_schema_not_deepseek_tools(self):
        self.responses = [{'strength': .02}]
        with patch.dict('os.environ', {}, clear=True):
            model = create_model(self.root, 3, config=self.config)
        self.assertTrue(model.health()['healthy'])
        answer = model.ask('Choose a valid strength', object_schema({'strength': {'type': 'number', 'enum': [.01, .02]}}), {})
        self.assertEqual(answer, {'strength': .02})
        path, payload, headers = self.requests[0]
        self.assertEqual(path, '/v1/chat/completions')
        self.assertEqual(payload['response_format']['type'], 'json_schema')
        self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking': False})
        self.assertNotIn('thinking', payload)
        self.assertNotIn('tools', payload)
        self.assertNotIn('Authorization', headers)
        self.assertIsNone(usage_summary(self.root)['usd_upper_estimate_for_reported_usage'])

    def test_discovery_episode_sampling_is_reproducible_and_scoped(self):
        config = {'llm': {**self.config['llm'], 'seed': 20261028,
                          'discovery_episode_sampling': True}}
        model = create_model(self.root, 4, config=config)
        schema = object_schema({'hypothesis': {'type': 'string'}})
        model.scope = {'scientific_role': 'open-discovery-scientist', 'episode': 'open-1'}
        seed1 = model.build_payload('investigate', schema)['seed']
        self.assertEqual(seed1, model.build_payload('investigate', schema)['seed'])
        model.scope['episode'] = 'open-2'
        self.assertNotEqual(seed1, model.build_payload('investigate', schema)['seed'])
        model.scope = {'scientific_role': 'ranking', 'episode': 'open-2'}
        self.assertEqual(model.build_payload('investigate', schema)['seed'], 20261028)

    def test_schema_retry_is_specific_and_remains_in_the_same_budget(self):
        self.responses = [{'strength': .03}, {'strength': .02}]
        model = create_model(self.root, 2, config=self.config)
        with patch('time.sleep'):
            answer = model.ask('Choose strength', object_schema({'strength': {'type': 'number', 'enum': [.01, .02]}}), {})
        self.assertEqual(answer, {'strength': .02})
        self.assertIn('allowed values: [0.01, 0.02]', self.requests[1][1]['messages'][0]['content'])
        self.assertEqual(usage_summary(self.root)['requests'], 2)

    def test_whole_study_uses_local_plan_execution_and_learning(self):
        self.responses = [
            {'hypothesis': '固定减分可能把有证据物品压到无证据物品下面。',
             'alternative_explanation': '热门可能代表真实相关性。',
             'operator': 'support_preserving', 'strength': .02,
             'prediction': '保留正分支持，检查质量与倾斜。', 'falsifier': '质量损失过大。', 'reason': '比较惩罚方式。'},
            {'lesson': '结论以程序实际输出为准。', 'next_question': '更弱惩罚有帮助吗？',
             'next_operator': 'retain', 'next_strength': 0, 'evidence_state': 'inconclusive'}]
        model = create_model(self.root, 2, config=self.config)
        strategy = {'version': 'fixture', 'strength_grid': [0, .02], 'evidence_focus': 'overall',
                    'preferred_operator': 'support_preserving', 'allowed_operators': ['retain', 'support_preserving']}
        result = run_study(self.root / 'study', model, coat_fixture(), rounds=1, strategy=strategy)
        self.assertEqual(result['execution_status'], 'completed')
        self.assertEqual(result['trials'][0]['policy'], {'operator': 'support_preserving', 'strength': .02})
        self.assertEqual(len(self.requests), 2)
        self.assertTrue((self.root / 'study/frozen_candidate.json').exists())

    def test_backend_identity_records_settings_and_rejects_unknown_providers(self):
        self.assertEqual(model_description()['provider'], 'deepseek')
        description = model_description(self.config)
        self.assertEqual(description['model_parameters']['max_tokens'], 2048)
        self.assertEqual(description['api_protocol'], 'vllm_json_schema_v1')
        with self.assertRaises(ValueError):
            model_description({'llm': {'provider': 'unknown'}})
