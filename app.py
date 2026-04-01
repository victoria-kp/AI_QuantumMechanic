"""FastAPI wrapper for the AI Quantum Mechanic agent."""

import asyncio
import base64
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anthropic import Anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

import agent.graph as graph_module
from agent.graph import run_agent

app = FastAPI(
    title="AI Quantum Mechanic",
    description="Solves quantum mechanics problems using an AI agent with symbolic and numerical tools.",
    version="1.0.0",
)

_executor = ThreadPoolExecutor(max_workers=2)
_client_lock = threading.Lock()

AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "600"))


# --- Request / Response models ---


class SolveRequest(BaseModel):
    api_key: str = Field(
        ...,
        description="Your Anthropic API key. Used for this request only, never stored.",
        min_length=1,
    )
    problem: str = Field(
        ...,
        description="A quantum mechanics problem in natural language.",
        min_length=10,
        examples=[
            "Solve the Schrodinger equation for the infinite square well (particle in a box) "
            "with walls at x=0 and x=a. The energy eigenvalues are positive. "
            "Find the analytical wavefunctions for the first three energy levels (n=1,2,3). "
            "Normalize them and plot the probability densities |psi_n(x)|^2 "
            "for all three levels on a single figure."
        ],
    )


class SolveStep(BaseModel):
    role: str
    content: str


class SolveResponse(BaseModel):
    answer: str
    check_passed: bool
    stop_reason: str
    steps: list[SolveStep]
    tools_used: list[str]
    figures: list[str]
    elapsed_seconds: float


# --- Helpers ---


def _run_with_key(api_key: str, problem: str):
    """Swap the module-level Anthropic client, run the agent, restore."""
    with _client_lock:
        original_client = graph_module.client
        graph_module.client = Anthropic(api_key=api_key)
        try:
            return run_agent(problem)
        finally:
            graph_module.client = original_client


# --- Frontend ---

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Quantum Mechanic</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
  .container { max-width: 800px; margin: 0 auto; padding: 2rem 1rem; }
  h1 { text-align: center; font-size: 1.8rem; margin-bottom: 0.3rem; }
  .subtitle { text-align: center; color: #94a3b8; margin-bottom: 2rem; font-size: 0.95rem; }
  label { display: block; font-weight: 600; margin-bottom: 0.4rem; font-size: 0.9rem; }
  input, textarea { width: 100%; padding: 0.7rem; border: 1px solid #334155; border-radius: 6px;
    background: #1e293b; color: #e2e8f0; font-size: 0.95rem; font-family: inherit; }
  input:focus, textarea:focus { outline: none; border-color: #6366f1; }
  textarea { resize: vertical; min-height: 100px; }
  .field { margin-bottom: 1.2rem; }
  button { width: 100%; padding: 0.8rem; background: #6366f1; color: white; border: none;
    border-radius: 6px; font-size: 1rem; font-weight: 600; cursor: pointer; }
  button:hover { background: #4f46e5; }
  button:disabled { background: #475569; cursor: not-allowed; }
  .spinner { display: inline-block; width: 18px; height: 18px; border: 2px solid #fff;
    border-top-color: transparent; border-radius: 50%; animation: spin 0.8s linear infinite;
    vertical-align: middle; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  #result { margin-top: 2rem; }
  .error { background: #7f1d1d; border: 1px solid #991b1b; padding: 1rem; border-radius: 6px; }
  .answer-box { background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 1.2rem;
    margin-bottom: 1rem; white-space: pre-wrap; line-height: 1.6; font-size: 0.9rem; }
  .meta { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
  .tag { background: #334155; padding: 0.3rem 0.7rem; border-radius: 4px; font-size: 0.8rem; }
  .tag.pass { background: #14532d; }
  .tag.fail { background: #7f1d1d; }
  .section-title { font-weight: 600; margin-bottom: 0.6rem; font-size: 1.1rem; color: #c4b5fd; }
  .figures img { max-width: 100%; border-radius: 6px; margin-bottom: 1rem; border: 1px solid #334155; }
  .note { color: #94a3b8; font-size: 0.8rem; margin-top: 0.5rem; }
</style>
</head>
<body>
<div class="container">
  <h1>AI Quantum Mechanic</h1>
  <p class="subtitle">Solve quantum mechanics problems step-by-step</p>
  <div class="field">
    <label for="api_key">Anthropic API Key</label>
    <input type="password" id="api_key" placeholder="sk-ant-...">
    <p class="note">Used for this request only. Never stored or logged.</p>
  </div>
  <div class="field">
    <label for="problem">Problem</label>
    <textarea id="problem" placeholder="Solve the Schrodinger equation for the infinite square well..."></textarea>
  </div>
  <button id="submit" onclick="submitProblem()">Solve</button>
  <div id="result"></div>
</div>
<script>
async function submitProblem() {
  const apiKey = document.getElementById('api_key').value.trim();
  const problem = document.getElementById('problem').value.trim();
  const btn = document.getElementById('submit');
  const result = document.getElementById('result');

  if (!apiKey || problem.length < 10) {
    result.innerHTML = '<div class="error">Please enter your API key and a problem (at least 10 characters).</div>';
    return;
  }

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Solving... (this may take a few minutes)';
  result.innerHTML = '';

  try {
    const resp = await fetch('/solve', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({api_key: apiKey, problem: problem})
    });
    const data = await resp.json();

    if (!resp.ok) {
      result.innerHTML = '<div class="error">Error: ' + (data.detail || resp.statusText) + '</div>';
      return;
    }

    let html = '<div class="meta">';
    html += '<span class="tag ' + (data.check_passed ? 'pass' : 'fail') + '">' +
            (data.check_passed ? 'Checks passed' : 'Checks failed') + '</span>';
    html += '<span class="tag">' + data.stop_reason + '</span>';
    html += '<span class="tag">' + data.tools_used.length + ' tool calls</span>';
    html += '<span class="tag">' + data.elapsed_seconds + 's</span>';
    html += '</div>';

    html += '<p class="section-title">Final Answer</p>';
    html += '<div class="answer-box">' + escapeHtml(data.answer) + '</div>';

    if (data.figures && data.figures.length > 0) {
      html += '<p class="section-title">Figures</p>';
      html += '<div class="figures" id="figbox"></div>';
    }
    result.innerHTML = html;

    const figbox = document.getElementById('figbox');
    if (figbox && data.figures) {
      for (const fig of data.figures) {
        const img = document.createElement('img');
        img.src = fig;
        img.alt = 'Plot';
        figbox.appendChild(img);
      }
    }
  } catch (e) {
    result.innerHTML = '<div class="error">Network error: ' + e.message + '</div>';
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Solve';
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
</script>
</body>
</html>"""


# --- Endpoints ---


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse)
async def solve(request: SolveRequest):
    """Submit a quantum mechanics problem and receive the agent's solution."""
    loop = asyncio.get_event_loop()
    t0 = time.time()

    try:
        final_state = await asyncio.wait_for(
            loop.run_in_executor(
                _executor, _run_with_key, request.api_key, request.problem
            ),
            timeout=AGENT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Agent timed out after {AGENT_TIMEOUT}s. Try a simpler problem or increase AGENT_TIMEOUT.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {type(e).__name__}: {str(e)}",
        )

    elapsed = time.time() - t0

    # Extract the final answer (last AIMessage)
    answer = ""
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            answer = msg.content
            break

    # Build step-by-step trace
    steps = []
    for msg in final_state["messages"]:
        role = type(msg).__name__.replace("Message", "").lower()
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        steps.append(SolveStep(role=role, content=content))

    # Encode figures as base64 data URIs
    encoded_figures = []
    for fig_path in final_state.get("figures", []):
        p = Path(fig_path)
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            encoded_figures.append(f"data:image/png;base64,{b64}")

    return SolveResponse(
        answer=answer,
        check_passed=final_state.get("check_passed", False),
        stop_reason=final_state.get("stop_reason", ""),
        steps=steps,
        tools_used=final_state.get("tool_history", []),
        figures=encoded_figures,
        elapsed_seconds=round(elapsed, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
