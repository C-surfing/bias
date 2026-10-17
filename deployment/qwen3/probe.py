"""Standalone service check for the deployment kit; no cloud backend fallback."""
import argparse
import json
from pathlib import Path
import urllib.request
from jsonschema import Draft202012Validator

p = argparse.ArgumentParser()
p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
p.add_argument('--output', default='/data/cjh/rsi/run/standalone-probe.json')
args = p.parse_args()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
base = args.base_url.rstrip('/')
with opener.open(base.removesuffix('/v1') + '/health', timeout=10) as response:
    assert response.status == 200
with opener.open(base + '/models', timeout=10) as response:
    models = json.load(response)
assert 'qwen3-8b' in [m['id'] for m in models['data']]
schema = {'type': 'object', 'properties': {
    'action': {'type': 'string', 'enum': ['retain', 'propose']},
    'strength': {'type': 'number', 'enum': [0, .01, .02]},
    'reason': {'type': 'string', 'pattern': '\\S'}},
    'required': ['action', 'strength', 'reason'], 'additionalProperties': False}
payload = {'model': 'qwen3-8b', 'messages': [{'role': 'user',
    'content': '这是连接检查，不执行实验。请用中文给出保留原版的动作、强度0和一句原因，只返回规定JSON。'}],
    'max_tokens': 256, 'temperature': .7, 'top_p': .8, 'top_k': 20, 'seed': 42,
    'chat_template_kwargs': {'enable_thinking': False},
    'response_format': {'type': 'json_schema', 'json_schema': {'name': 'probe', 'schema': schema}}}
request = urllib.request.Request(base + '/chat/completions', json.dumps(payload).encode(), {'Content-Type': 'application/json'})
with opener.open(request, timeout=180) as response:
    result = json.load(response)
choice = result['choices'][0]
assert choice['finish_reason'] == 'stop'
answer = json.loads(choice['message']['content'])
Draft202012Validator(schema).validate(answer)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({'status': 'completed', 'models': models, 'answer': answer,
    'usage': result.get('usage'), 'scope': 'service compatibility only'}, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': 'completed', 'answer': answer}, ensure_ascii=False))
