import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.hypothesis_factory.base import BusinessRequest, HypothesisList

API_URL = "http://127.0.0.1:8000"


def generate_hypotheses(request: BusinessRequest) -> HypothesisList:
    response = requests.post(f"{API_URL}/generate", json=request.model_dump())
    response.raise_for_status()
    return HypothesisList.model_validate(response.json())
