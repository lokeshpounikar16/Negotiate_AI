from fastapi.testclient import TestClient

from main import app


def test_negotiation_websocket_connects():
    with TestClient(app) as client:
        with client.websocket_connect('/ws/negotiation') as websocket:
            payload = websocket.receive_json()
            assert payload['type'] == 'connected'
            websocket.send_text('ping')
            pong = websocket.receive_text()
            assert pong == 'pong'
