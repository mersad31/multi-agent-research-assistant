from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")

    assert response.status_code == 200


def test_chat_resume_revise_without_note_returns_422():
    response = client.post(
        "/chat/resume",
        json={"thread_id": "abc", "action": "revise"},
    )

    assert response.status_code == 422


def test_chat_resume_invalid_action_returns_422():
    response = client.post(
        "/chat/resume",
        json={"thread_id": "abc", "action": "aproove"},
    )

    assert response.status_code == 422