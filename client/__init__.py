"""
BioAI Unified Client
====================

Python client library for the BioAI Unified Engine API.

Quick Start:
    from client import BioAIClient

    # Sync usage
    client = BioAIClient()
    result = client.generate("Write an article about AI")
    print(result["content"])

    # Async usage
    from client import AsyncBioAIClient

    async with AsyncBioAIClient() as client:
        result = await client.generate("Write an article about AI")
        print(result["content"])

For more examples, see the demos/ folder.
"""

from typing import Dict, Optional


class BioAIClientError(Exception):
    """Exception raised for BioAI client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details or {}


from .sync_client import BioAIClient
from .async_client import AsyncBioAIClient

__all__ = [
    "BioAIClient",
    "AsyncBioAIClient",
    "BioAIClientError",
]

__version__ = "1.0.0"
