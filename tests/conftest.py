import asyncio
import pytest

@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for the entire test session to avoid it being closed between tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()