"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import json
import asyncio

from . import storage
from .council import run_full_council, generate_conversation_title, stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final, calculate_aggregate_rankings
from .tavily_search import get_search_context
from .auth import verify_password, create_session, delete_session, get_session_token, require_auth, validate_session
from .config import AUTH_PASSWORD, CORS_ORIGINS

app = FastAPI(title="LLM Council API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    """Request to log in."""
    password: str


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    pass


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    message_count: int


class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}


@app.get("/api/auth/status")
async def auth_status(request: Request):
    """Check if user is authenticated and if auth is required."""
    auth_required = bool(AUTH_PASSWORD)
    token = get_session_token(request)
    authenticated = validate_session(token) if auth_required else True
    return {"authenticated": authenticated, "auth_required": auth_required}


@app.post("/api/auth/login")
async def login(request: LoginRequest, response: Response):
    """Log in with password."""
    if not verify_password(request.password):
        raise HTTPException(status_code=401, detail="Invalid password")
    token = create_session()
    # Use secure cookies if any origin uses HTTPS
    is_https = any(origin.startswith("https://") for origin in CORS_ORIGINS)
    response.set_cookie(
        key="session",
        value=token,
        httponly=True,
        secure=is_https,
        samesite="lax" if not is_https else "none",
        max_age=60 * 60 * 24 * 30,  # 30 days
    )
    return {"status": "ok"}


@app.post("/api/auth/logout")
async def logout(request: Request, response: Response):
    """Log out and clear session."""
    token = get_session_token(request)
    if token:
        delete_session(token)
    response.delete_cookie(key="session")
    return {"status": "ok"}


@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations(request: Request, _: None = Depends(require_auth)):
    """List all conversations (metadata only)."""
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: Request, body: CreateConversationRequest, _: None = Depends(require_auth)):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(request: Request, conversation_id: str, _: None = Depends(require_auth)):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(request: Request, conversation_id: str, body: SendMessageRequest, _: None = Depends(require_auth)):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    # Add user message
    storage.add_user_message(conversation_id, body.content)

    # If this is the first message, generate a title
    if is_first_message:
        title = await generate_conversation_title(body.content)
        storage.update_conversation_title(conversation_id, title)

    # Run the 3-stage council process
    stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
        body.content
    )

    # Add assistant message with all stages
    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result
    )

    # Return the complete response with metadata
    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata
    }


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(request: Request, conversation_id: str, body: SendMessageRequest, _: None = Depends(require_auth)):
    """
    Send a message and stream the 3-stage council process.
    Returns Server-Sent Events as each stage completes.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    async def event_generator():
        try:
            # Add user message
            storage.add_user_message(conversation_id, body.content)

            # Start title generation in parallel (don't await yet)
            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(generate_conversation_title(body.content))

            # Check if web search is needed
            yield f"data: {json.dumps({'type': 'web_search_start'})}\n\n"
            web_context = await get_search_context(body.content)
            web_search_used = web_context is not None
            yield f"data: {json.dumps({'type': 'web_search_complete', 'data': {'used': web_search_used}})}\n\n"

            # Stage 1: Collect responses (with web context if available)
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_results = await stage1_collect_responses(body.content, web_context)
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

            # Stage 2: Collect rankings
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2_results, label_to_model = await stage2_collect_rankings(body.content, stage1_results)
            aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings, 'web_search_used': web_search_used}})}\n\n"

            # Stage 3: Synthesize final answer
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3_result = await stage3_synthesize_final(body.content, stage1_results, stage2_results)
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

            # Wait for title generation if it was started
            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            # Save complete assistant message
            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result
            )

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
