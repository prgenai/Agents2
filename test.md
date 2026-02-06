cannot generate a downloadable binary .zip file directly in this chat interface. However, I have generated the complete, copy-pasteable codebase for your project below.

You can create a folder named tachyon and save these files according to the directory structure provided.

📂 Project Structure
Plaintext
project_root/
├── config.yaml                 # Configuration
├── requirements.txt            # Dependencies
├── README.md                   # The Documentation
└── tachyon/
    ├── __init__.py
    ├── router/
    │   ├── __init__.py
    │   ├── interfaces.py       # Contracts & Data Classes
    │   ├── router.py           # The Smart Router Logic
    │   └── session_manager.py  # Sticky Session Store
    └── clients/
        ├── __init__.py
        ├── adk_client.py       # Google ADK Adapter
        └── langchain_client.py # LangChain Adapter
1. README.md (Documentation)
Markdown
# Tachyon Framework 🚀

Tachyon is a unified resilience layer for AI Agents. It sits between your agent frameworks (Google ADK, LangChain) and LLM providers (Vertex AI, OpenAI) to provide **Sticky Sessions**, **Global Cooldowns**, **Circuit Breaking**, and **Automatic Failover**.

## Architecture
The `TachyonModelRouter` acts as a central gateway.
1.  **Stickiness:** Users are "glued" to a model (e.g., Gemini Pro) for a set duration.
2.  **Resilience:** If a model fails (e.g., 429 Error), Tachyon automatically retries.
3.  **Failover:** If retries fail, Tachyon marks the model as "Cooling Down" globally and switches the user to the next healthy model (e.g., GPT-4) instantly.

## Installation

```bash
pip install -r requirements.txt
Configuration (config.yaml)
Define your models and routing groups.

YAML
resilience:
  global_cooldown_seconds: 60
  max_retries: 3
  sticky_ttl_minutes: 30

models:
  - id: "gemini-1.5-pro"
    provider: "vertex_ai"
    model_name: "gemini-1.5-pro-preview-0409"
    api_key: "env:VERTEX_API_KEY"

  - id: "gpt-4"
    provider: "openai"
    model_name: "gpt-4-turbo"
    api_key: "env:OPENAI_API_KEY"

groups:
  default:
    - "gemini-1.5-pro"
    - "gpt-4"
Usage
1. With Google ADK
Python
from tachyon.router import TachyonModelRouter, InMemorySessionManager
from tachyon.clients.adk_client import TachyonADKClient
import yaml

# Init
config = yaml.safe_load(open("config.yaml"))
router = TachyonModelRouter(config, InMemorySessionManager())

# Create Client
client = TachyonADKClient(router)

# Use in Agent (Tachyon handles the rest!)
response = await client.async_generate_content("Hello", session_id="user_123")
2. With LangChain
Python
from tachyon.clients.langchain_client import TachyonLangChainClient

llm = TachyonLangChainClient(router=router)

# Standard LangChain usage
async for chunk in llm.astream("Tell me a joke", session_id="user_123"):
    print(chunk.content, end="")

---

### 2. `requirements.txt`

```text
google-genai
langchain-core
langchain-openai
litellm
tenacity
pyyaml
3. config.yaml
YAML
resilience:
  global_cooldown_seconds: 60
  max_retries: 3
  sticky_ttl_minutes: 30

# Define your available models
models:
  - id: "gemini-1.5-pro"
    provider: "vertex_ai"
    model_name: "gemini-1.5-pro-preview-0409"
    api_key: "MY_VERTEX_KEY" # Or use env vars in code

  - id: "gemini-1.5-flash"
    provider: "vertex_ai"
    model_name: "gemini-1.5-flash-preview-0514"
    api_key: "MY_VERTEX_KEY"

  - id: "gpt-4-turbo"
    provider: "openai"
    model_name: "gpt-4-turbo"
    api_key: "MY_OPENAI_KEY"

# Define routing logic (First available -> Failover)
groups:
  default:
    - "gemini-1.5-pro"
    - "gemini-1.5-flash"
    - "gpt-4-turbo"
4. tachyon/router/interfaces.py
Python
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable, Awaitable, AsyncIterator
from dataclasses import dataclass

@dataclass
class ModelConfig:
    id: str
    provider: str  # "vertex_ai", "openai"
    model_name: str
    api_key: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class BaseSessionManager(ABC):
    """Contract for Sticky Session Storage"""
    @abstractmethod
    async def get_affinity(self, session_id: str) -> Optional[str]: pass

    @abstractmethod
    async def set_affinity(self, session_id: str, model_id: str, ttl_seconds: int): pass

class BaseTachyonRouter(ABC):
    """Contract for the Core Router"""
    @abstractmethod
    async def route_and_execute(
        self, 
        session_id: str, 
        execution_callback: Callable[[ModelConfig], Awaitable[Any]],
        group: str = "default"
    ) -> Any: pass

    @abstractmethod
    async def route_and_stream(
        self,
        session_id: str,
        execution_callback: Callable[[ModelConfig], Awaitable[AsyncIterator[Any]]],
        group: str = "default"
    ) -> AsyncIterator[Any]: pass
5. tachyon/router/session_manager.py
Python
import time
import asyncio
from typing import Dict, Tuple, Optional
from .interfaces import BaseSessionManager

class InMemorySessionManager(BaseSessionManager):
    """
    In-Memory implementation for Cloud Run (Per-Instance).
    TODO: Swap with RedisSessionManager for multi-instance stickiness.
    """
    def __init__(self):
        # Structure: { session_id: (model_id, expiration_timestamp) }
        self._store: Dict[str, Tuple[str, float]] = {}
        self._lock = asyncio.Lock()

    async def get_affinity(self, session_id: str) -> Optional[str]:
        async with self._lock:
            if session_id not in self._store:
                return None
            
            model_id, expires_at = self._store[session_id]
            
            # Lazy Expiration Check
            if time.time() > expires_at:
                del self._store[session_id]
                return None
            
            return model_id

    async def set_affinity(self, session_id: str, model_id: str, ttl_seconds: int = 1800):
        async with self._lock:
            expires_at = time.time() + ttl_seconds
            self._store[session_id] = (model_id, expires_at)
6. tachyon/router/router.py (The Brain)
Python
import time
import random
import logging
import asyncio
from typing import Dict, List, Any, Callable, Awaitable, AsyncIterator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .interfaces import BaseTachyonRouter, BaseSessionManager, ModelConfig

logger = logging.getLogger("TachyonRouter")

class GlobalCooldownError(Exception):
    pass

class TachyonModelRouter(BaseTachyonRouter):
    def __init__(self, config: Dict[str, Any], session_manager: BaseSessionManager):
        self.session_manager = session_manager
        
        # Parse Models
        self.models = {m["id"]: ModelConfig(**m) for m in config.get("models", [])}
        self.groups = config.get("groups", {"default": list(self.models.keys())})
        
        # Resilience Settings
        resilience = config.get("resilience", {})
        self.cooldown_seconds = resilience.get("global_cooldown_seconds", 60)
        self.max_retries = resilience.get("max_retries", 3)
        
        # State
        self.cooldowns: Dict[str, float] = {}

    def _is_cooling_down(self, model_id: str) -> bool:
        if model_id in self.cooldowns:
            if time.time() < self.cooldowns[model_id]:
                return True
            else:
                del self.cooldowns[model_id]  # Expired
        return False

    def _trigger_cooldown(self, model_id: str):
        logger.warning(f"🔥 Triggering Cooldown for {model_id}")
        self.cooldowns[model_id] = time.time() + self.cooldown_seconds

    async def _get_candidate(self, session_id: str, group: str) -> ModelConfig:
        group_ids = self.groups.get(group, [])
        if not group_ids:
            raise ValueError(f"Group {group} is empty")

        # 1. Check Stickiness
        sticky_id = await self.session_manager.get_affinity(session_id)
        if sticky_id and sticky_id in group_ids and not self._is_cooling_down(sticky_id):
            return self.models[sticky_id]

        # 2. Select Healthy Candidate (Round Robin/Random)
        healthy_ids = [mid for mid in group_ids if not self._is_cooling_down(mid)]
        
        if not healthy_ids:
            raise GlobalCooldownError("All models are in cooldown!")
        
        selected_id = random.choice(healthy_ids)
        
        # 3. Update Stickiness
        await self.session_manager.set_affinity(session_id, selected_id)
        return self.models[selected_id]

    # --- EXECUTE (Request/Response) ---
    async def route_and_execute(self, session_id, execution_callback, group="default", _depth=0):
        if _depth > 3: raise GlobalCooldownError("Max failover depth reached.")

        target_model = await self._get_candidate(session_id, group)

        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_exponential(multiplier=1, min=1), reraise=True)
        async def _attempt():
            return await execution_callback(target_model)

        try:
            return await _attempt()
        except Exception as e:
            logger.error(f"❌ Execution failed on {target_model.id}: {e}")
            self._trigger_cooldown(target_model.id)
            return await self.route_and_execute(session_id, execution_callback, group, _depth + 1)

    # --- STREAM (Generators) ---
    async def route_and_stream(self, session_id, execution_callback, group="default", _depth=0):
        if _depth > 3: raise GlobalCooldownError("Max failover depth reached.")

        target_model = await self._get_candidate(session_id, group)

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1), reraise=True)
        async def _create_stream():
            return await execution_callback(target_model)

        try:
            # Only retry the connection setup
            stream_iterator = await _create_stream()
            async for chunk in stream_iterator:
                yield chunk
        except Exception as e:
            logger.error(f"❌ Stream failed on {target_model.id}: {e}")
            self._trigger_cooldown(target_model.id)
            # Recursive Failover for Streams
            async for chunk in self.route_and_stream(session_id, execution_callback, group, _depth + 1):
                yield chunk
7. tachyon/clients/adk_client.py (ADK Adapter)
Python
from typing import Any
from google.genai.types import GenerateContentResponse
from tachyon.router.interfaces import ModelConfig, BaseTachyonRouter

class TachyonADKClient:
    """
    Simulates a Google ADK Model but routes via Tachyon.
    """
    def __init__(self, router: BaseTachyonRouter, model_group: str = "default"):
        self.router = router
        self.group = model_group
        # Property required by some ADK agents
        self.model_name = "tachyon-router"

    async def async_generate_content(self, contents, config=None, **kwargs) -> GenerateContentResponse:
        
        # 1. Extract Session ID
        session_id = kwargs.get("session_id", "default")
        if config and hasattr(config, "metadata"):
            session_id = config.metadata.get("session_id", session_id)

        # 2. Define Callback
        async def _execute_model_logic(model_conf: ModelConfig):
            print(f"⚡ Tachyon executing with: {model_conf.id}")
            
            # --- INSERT YOUR ORIGINAL IMPLEMENTATION HERE ---
            # Call self._internal_google_call or your existing logic.
            # CRITICAL: Use `model_conf.model_name` instead of hardcoded values.
            
            # Mock return for demonstration:
            return await self._mock_google_call(model_conf, contents)

        # 3. Execute
        return await self.router.route_and_execute(
            session_id=session_id,
            execution_callback=_execute_model_logic,
            group=self.group
        )

    async def _mock_google_call(self, conf, contents):
        # In real code, call: await self.original_client.generate_content(...)
        from google.genai.types import Candidate, Content, Part
        return GenerateContentResponse(
            candidates=[Candidate(content=Content(parts=[Part(text=f"Response from {conf.id}")]))]
        )
8. tachyon/clients/langchain_client.py (LangChain Adapter)
Python
import asyncio
from typing import List, Any, AsyncIterator, Optional, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun

from tachyon.router.interfaces import BaseTachyonRouter, ModelConfig

class TachyonLangChainClient(BaseChatModel):
    router: BaseTachyonRouter
    model_group: str = "default"

    @property
    def _llm_type(self) -> str:
        return "tachyon-router"

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> AsyncIterator[ChatGenerationChunk]:
        
        session_id = kwargs.get("session_id", "default")
        params = self._get_model_parameters(stop, **kwargs)

        async def _streaming_callback(model_conf: ModelConfig) -> AsyncIterator[Any]:
            import litellm
            litellm_messages = self._convert_messages(messages)
            
            return await litellm.acompletion(
                model=model_conf.model_name,
                api_key=model_conf.api_key,
                messages=litellm_messages,
                stream=True,
                **params
            )

        stream_generator = self.router.route_and_stream(
            session_id=session_id,
            execution_callback=_streaming_callback,
            group=self.model_group
        )

        async for chunk in stream_generator:
            delta = chunk.choices[0].delta.content or ""
            lc_chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=delta)
            )
            if run_manager:
                await run_manager.on_llm_new_token(token=delta, chunk=lc_chunk)
            yield lc_chunk

    # --- Wrapper for Non-Streaming ---
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        final_content = ""
        async for chunk in self._astream(messages, stop, run_manager, **kwargs):
            final_content += chunk.message.content
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=final_content))])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        import asyncio
        return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))

    # --- Helpers ---
    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict]:
        formatted = []
        for m in messages:
            role = "user"
            if m.type == "ai": role = "assistant"
            elif m.type == "system": role = "system"
            formatted.append({"role": role, "content": m.content})
        return formatted

    def _get_model_parameters(self, stop, **kwargs):
        p = kwargs.copy()
        if stop: p["stop"] = stop
        return p
