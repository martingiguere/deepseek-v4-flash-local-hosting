import json
import os
import subprocess
import threading
import time
import unittest
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import HTTPError


class MockHandler(BaseHTTPRequestHandler):
    MODE = 'ollama'
    requests = []

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        data = json.loads(body)
        self.__class__.requests.append({
            'path': self.path,
            'headers': {k.lower(): v for k, v in dict(self.headers).items()},
            'body': data,
        })

        is_stream = data.get('stream', False)

        if is_stream:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()

            has_tools = 'tools' in data and data['tools']
            if has_tools:
                self._stream_tool_call_response()
            else:
                self._stream_text_response()
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            has_tools = 'tools' in data and data['tools']
            if has_tools:
                response = self._tool_call_response(data)
            else:
                response = self._text_response()

            self.wfile.write(json.dumps(response).encode())

    def _text_response(self):
        return {
            'id': 'chatcmpl-123',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': 'deepseek-v4-flash',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'Hello from Mock',
                },
                'finish_reason': 'stop',
            }],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
        }

    def _tool_call_response(self, data):
        tool_name = data['tools'][0]['function']['name']
        return {
            'id': 'chatcmpl-456',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': 'deepseek-v4-flash',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': '',
                    'tool_calls': [{
                        'id': 'call_abc123',
                        'type': 'function',
                        'function': {
                            'name': tool_name,
                            'arguments': '{"query": "test"}',
                        },
                    }],
                },
                'finish_reason': 'tool_calls',
            }],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
        }

    def _write_sse(self, data_dict):
        self.wfile.write(f'data: {json.dumps(data_dict)}\n\n'.encode())
        self.wfile.flush()

    def _stream_text_response(self):
        self._write_sse({'choices': [{'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})
        self._write_sse({'choices': [{'delta': {'content': 'Hello'}, 'finish_reason': None}]})
        self._write_sse({'choices': [{'delta': {'content': ' from'}, 'finish_reason': None}]})
        self._write_sse({'choices': [{'delta': {'content': ' Mock'}, 'finish_reason': None}]})
        self._write_sse({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
        self.wfile.write(b'data: [DONE]\n\n')
        self.wfile.flush()

    def _stream_tool_call_response(self):
        self._write_sse({'choices': [{'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})
        tool_call_chunk = {
            'choices': [{
                'delta': {
                    'tool_calls': [{
                        'index': 0,
                        'id': 'call_abc123',
                        'type': 'function',
                        'function': {'name': 'get_weather', 'arguments': ''},
                    }],
                },
                'finish_reason': None,
            }],
        }
        self._write_sse(tool_call_chunk)
        arg_chunk = {
            'choices': [{
                'delta': {
                    'tool_calls': [{
                        'index': 0,
                        'function': {'arguments': '{"query": "test"}'},
                    }],
                },
                'finish_reason': None,
            }],
        }
        self._write_sse(arg_chunk)
        self._write_sse({'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}]})
        self.wfile.write(b'data: [DONE]\n\n')
        self.wfile.flush()

    def log_message(self, format, *args):
        pass


class TestCCP(unittest.TestCase):
    BASE = 'http://localhost:8080'

    def setUp(self):
        MockHandler.requests.clear()

    def _post(self, path, body, headers=None):
        hdrs = {
            'Content-Type': 'application/json',
            'x-api-key': 'test-key',
            'anthropic-version': '2023-06-01',
        }
        if headers:
            hdrs.update(headers)
        data = json.dumps(body).encode()
        req = Request(self.BASE + path, data=data, headers=hdrs, method='POST')
        try:
            resp = urlopen(req)
            return resp.status, json.loads(resp.read().decode())
        except HTTPError as e:
            body_bytes = e.read()
            try:
                return e.code, json.loads(body_bytes.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                return e.code, {'error': {'message': body_bytes.decode(errors='replace')}}

    def _get(self, path):
        req = Request(self.BASE + path, method='GET')
        resp = urlopen(req)
        return resp.status, json.loads(resp.read().decode())

    def _post_stream(self, path, body):
        hdrs = {
            'Content-Type': 'application/json',
            'x-api-key': 'test-key',
            'anthropic-version': '2023-06-01',
        }
        data = json.dumps(body).encode()
        req = Request(self.BASE + path, data=data, headers=hdrs, method='POST')
        resp = urlopen(req)
        return resp.status, resp.read().decode()

    def _last_mock_request(self):
        if MockHandler.requests:
            return MockHandler.requests[-1]
        return None

    def _mock_has_requests(self):
        return len(MockHandler.requests) > 0

    def test_turn1_simple_text(self):
        body = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 100,
            'messages': [{'role': 'user', 'content': 'Hello'}],
        }
        status, resp = self._post('/v1/messages', body)
        self.assertEqual(status, 200)
        self.assertEqual(resp['type'], 'message')
        self.assertEqual(resp['role'], 'assistant')
        self.assertEqual(resp['stop_reason'], 'end_turn')
        self.assertTrue(any(b['type'] == 'text' for b in resp['content']))

        if self._mock_has_requests():
            mock_req = self._last_mock_request()
            messages = mock_req['body']['messages']
            self.assertEqual(messages[-1]['role'], 'user')
            self.assertEqual(messages[-1]['content'], 'Hello')
            self.assertEqual(mock_req['body']['model'], 'deepseek-v4-flash')

    def test_turn1_tool_call(self):
        body = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 100,
            'messages': [{'role': 'user', 'content': 'What is the weather?'}],
            'tools': [{
                'name': 'get_weather',
                'description': 'Get weather',
                'input_schema': {'type': 'object', 'properties': {'location': {'type': 'string'}}},
            }],
        }
        status, resp = self._post('/v1/messages', body)
        self.assertEqual(status, 200)
        self.assertTrue(any(b['type'] == 'tool_use' for b in resp['content']))

        if self._mock_has_requests():
            mock_req = self._last_mock_request()
            tools = mock_req['body']['tools']
            self.assertEqual(tools[0]['type'], 'function')

    def test_turn2_tool_result(self):
        body = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 100,
            'messages': [
                {'role': 'user', 'content': 'What is the weather?'},
                {'role': 'assistant', 'content': [
                    {'type': 'text', 'text': 'Let me check'},
                    {'type': 'tool_use', 'id': 'toolu_abc', 'name': 'get_weather',
                     'input': {'location': 'NYC'}},
                ]},
                {'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': 'toolu_abc', 'content': 'Sunny'},
                ]},
            ],
        }
        status, resp = self._post('/v1/messages', body)
        self.assertEqual(status, 200)

        if self._mock_has_requests():
            mock_req = self._last_mock_request()
            messages = mock_req['body']['messages']
            tool_msg = [m for m in messages if m['role'] == 'tool']
            self.assertTrue(len(tool_msg) > 0)
            self.assertEqual(tool_msg[0]['tool_call_id'], 'toolu_abc')

    def test_model_opus(self):
        body = {
            'model': 'claude-opus-4-20250514',
            'max_tokens': 100,
            'messages': [{'role': 'user', 'content': 'Hi'}],
        }
        status, resp = self._post('/v1/messages', body)
        self.assertEqual(status, 200)
        self.assertEqual(resp['model'], 'deepseek-v4-pro')

        if self._mock_has_requests():
            mock_req = self._last_mock_request()
            self.assertEqual(mock_req['body']['model'], 'deepseek-v4-pro')

    def test_model_sonnet(self):
        body = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 100,
            'messages': [{'role': 'user', 'content': 'Hi'}],
        }
        status, resp = self._post('/v1/messages', body)
        self.assertEqual(status, 200)
        self.assertEqual(resp['model'], 'deepseek-v4-flash')

    def test_model_haiku(self):
        body = {
            'model': 'claude-haiku-3-20240307',
            'max_tokens': 100,
            'messages': [{'role': 'user', 'content': 'Hi'}],
        }
        status, resp = self._post('/v1/messages', body)
        self.assertEqual(status, 200)
        self.assertEqual(resp['model'], 'deepseek-v4-flash')

    def test_model_unknown(self):
        body = {
            'model': 'claude-some-random-model',
            'max_tokens': 100,
            'messages': [{'role': 'user', 'content': 'Hi'}],
        }
        status, resp = self._post('/v1/messages', body)
        self.assertEqual(status, 200)
        self.assertEqual(resp['model'], 'deepseek-v4-flash')

    def test_system_string(self):
        body = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 100,
            'system': 'You are a helpful assistant.',
            'messages': [{'role': 'user', 'content': 'Hi'}],
        }
        _, _ = self._post('/v1/messages', body)

        if self._mock_has_requests():
            mock_req = self._last_mock_request()
            messages = mock_req['body']['messages']
            self.assertEqual(messages[0]['role'], 'system')
            self.assertEqual(messages[0]['content'], 'You are a helpful assistant.')

    def test_system_array(self):
        body = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 100,
            'system': [{'type': 'text', 'text': 'Be helpful.'}, {'type': 'text', 'text': 'Be concise.'}],
            'messages': [{'role': 'user', 'content': 'Hi'}],
        }
        _, _ = self._post('/v1/messages', body)

        if self._mock_has_requests():
            mock_req = self._last_mock_request()
            messages = mock_req['body']['messages']
            system_msgs = [m for m in messages if m['role'] == 'system']
            combined = ' '.join(str(m.get('content', '')) for m in system_msgs)
            self.assertIn('Be helpful.', combined)
            self.assertIn('Be concise.', combined)

    def test_tool_choice_any(self):
        body = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 100,
            'messages': [{'role': 'user', 'content': 'Hi'}],
            'tools': [{'name': 'test_tool', 'description': 'A test', 'input_schema': {'type': 'object'}}],
            'tool_choice': {'type': 'any'},
        }
        _, _ = self._post('/v1/messages', body)

        if self._mock_has_requests():
            mock_req = self._last_mock_request()
            self.assertEqual(mock_req['body']['tool_choice'], 'required')

    def test_tool_choice_named(self):
        body = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 100,
            'messages': [{'role': 'user', 'content': 'Hi'}],
            'tools': [{'name': 'get_weather', 'description': 'Weather', 'input_schema': {'type': 'object'}}],
            'tool_choice': {'type': 'tool', 'name': 'get_weather'},
        }
        _, _ = self._post('/v1/messages', body)

        if self._mock_has_requests():
            mock_req = self._last_mock_request()
            tc = mock_req['body']['tool_choice']
            self.assertEqual(tc['type'], 'function')
            self.assertEqual(tc['function']['name'], 'get_weather')

    def test_image_block_rejected(self):
        body = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 100,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': 'What is this?'},
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': 'AAAA'}},
            ]}],
        }
        status, resp = self._post('/v1/messages', body)
        self.assertEqual(status, 400)
        self.assertIn('image', str(resp))

    def test_health_endpoint(self):
        status, resp = self._get('/health')
        self.assertEqual(status, 200)
        self.assertEqual(resp['status'], 'ok')

    def test_count_tokens_stub(self):
        body = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 100,
            'messages': [{'role': 'user', 'content': 'Hi'}],
        }
        status, resp = self._post('/v1/messages/count_tokens', body)
        self.assertEqual(status, 200)
        self.assertIn('input_tokens', resp)

    def test_stream_text(self):
        body = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 100,
            'stream': True,
            'messages': [{'role': 'user', 'content': 'Hello'}],
        }
        status, raw = self._post_stream('/v1/messages', body)
        self.assertEqual(status, 200)
        self.assertIn('event: message_start', raw)
        self.assertIn('event: message_delta', raw)
        self.assertIn('event: message_stop', raw)
        self.assertIn('event: content_block_start', raw)
        self.assertIn('event: content_block_delta', raw)
        self.assertIn('event: content_block_stop', raw)
        self.assertIn('"type":"text_delta"', raw)

    def test_stream_tool_call(self):
        body = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 100,
            'stream': True,
            'messages': [{'role': 'user', 'content': 'What is the weather?'}],
            'tools': [{
                'name': 'get_weather',
                'description': 'Get weather',
                'input_schema': {'type': 'object', 'properties': {'location': {'type': 'string'}}},
            }],
        }
        status, raw = self._post_stream('/v1/messages', body)
        self.assertEqual(status, 200)
        self.assertIn('event: content_block_start', raw)
        self.assertIn('"type":"tool_use"', raw)

    def test_cch_header_stripped(self):
        body = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 100,
            'messages': [{'role': 'user', 'content': 'Hello'}],
        }
        hdrs = {
            'x-anthropic-billing-header': 'some-billing-value',
        }
        _, _ = self._post('/v1/messages', body, headers=hdrs)

        if self._mock_has_requests():
            mock_req = self._last_mock_request()
            mock_headers = mock_req['headers']
            self.assertNotIn('x-anthropic-billing-header', mock_headers)


if __name__ == '__main__':
    backend = os.environ.get('CCP_BACKEND', 'ollama')
    MockHandler.MODE = backend

    mock = HTTPServer(('localhost', 9999), MockHandler)
    mock_thread = threading.Thread(target=mock.serve_forever, daemon=True)
    mock_thread.start()

    env = {**os.environ, 'CCP_PORT': '8080'}
    if backend == 'ollama':
        env.update({'CCP_BACKEND': 'ollama', 'CCP_OLLAMA_API_KEY': 'test',
                    'CCP_OLLAMA_BASE_URL': 'http://localhost:9999'})
    else:
        env.update({'CCP_BACKEND': 'bifrost', 'CCP_BIFROST_API_KEY': 'test',
                    'CCP_BIFROST_BASE_URL': 'http://localhost:9999'})

    proxy = subprocess.Popen(['./ccp'], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    time.sleep(0.5)

    # Check if proxy is alive
    if proxy.poll() is not None:
        stderr_output = proxy.stderr.read().decode()
        raise RuntimeError(f"Proxy exited immediately with code {proxy.returncode}. Stderr: {stderr_output}")

    try:
        unittest.main(argv=[''], exit=False)
    finally:
        proxy.terminate()
        mock.shutdown()