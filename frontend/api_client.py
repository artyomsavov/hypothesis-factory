import requests

API_URL = "http://127.0.0.1:8000"


def generate_hypotheses(target_kpi: str, constraints: list) -> dict:
    payload = {"target_kpi": target_kpi, "constraints": constraints}
    response = requests.post(f"{API_URL}/generate_hypotheses", json=payload)
    response.raise_for_status()
    return response.json()


def upload_documents(files) -> dict:
    upload_data = [("files", (file.name, file.getvalue(), file.type)) for file in files]
    response = requests.post(f"{API_URL}/upload_documents", files=upload_data)
    response.raise_for_status()
    return response.json()
