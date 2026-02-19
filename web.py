#!/usr/bin/env python3
"""Paper Blitz v2 — Web Frontend Server with Timothy Teaching Mode"""

import json, os, sys, asyncio, re, time, hashlib, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from pydantic import BaseModel
import uvicorn
from openai import OpenAI

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# --- Config ---
SCRIPT_DIR = Path(__file__).parent
PAPERS_FILE = SCRIPT_DIR / "papers.json"
DIGESTS_DIR = SCRIPT_DIR / "digests"
DIGESTS_DIR.mkdir(exist_ok=True)

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ELEVENLABS_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

# --- Analyst Models ---
ANALYSTS = {
    "researcher": {
        "model": "anthropic/claude-sonnet-4.5",
        "label": "The Researcher",
        "icon": "\U0001f52c",
        "prompt": """You are a meticulous academic researcher. Analyze this paper focusing on:
- Exact methodology: sample sizes, experimental design, controls, statistical methods
- Specific numerical results with confidence intervals where available
- Limitations the authors acknowledge AND ones they don't
- How this methodology could be replicated or improved
Be precise. Quote exact numbers. Flag anything that seems methodologically weak."""
    },
    "critic": {
        "model": "anthropic/claude-opus-4.6",
        "label": "The Critic",
        "icon": "\U0001f50d",
        "prompt": """You are a rigorous academic critic. Analyze this paper focusing on:
- What are the weakest claims? Where does the evidence NOT support the conclusions?
- What confounds or alternative explanations exist?
- What would a skeptical reviewer say?
- How does this hold up 1-2 years after publication given what we now know?
Be ruthlessly honest. If the paper is flawed, say so clearly."""
    },
    "builder": {
        "model": "google/gemini-2.5-pro",
        "label": "The Builder",
        "icon": "\U0001f6e0\ufe0f",
        "prompt": """You are an AI engineer building a system that impersonates specific people via text messaging. Analyze this paper focusing on:
- What specific techniques from this paper can be directly implemented?
- What architecture decisions does this inform?
- What training approaches does this validate or invalidate?
- If you had to build a system based on this paper's findings, what would you do differently?
Be concrete and technical. Think implementation, not theory."""
    },
    "historian": {
        "model": "openai/gpt-4o",
        "label": "The Historian",
        "icon": "\U0001f4da",
        "prompt": """You are a science historian and field mapper. Analyze this paper focusing on:
- Where does this sit in the timeline of this research area?
- What papers influenced this work? What did it influence?
- How does this relate to the Turing test (classic 1950), individual Turing tests, and consciousness cloning?
- What's the broader narrative arc this paper is part of?
Name specific researchers, labs, and competing approaches."""
    },
    "storyteller": {
        "model": "openai/gpt-4o",
        "label": "The Storyteller",
        "icon": "\U0001f3af",
        "prompt": """You are a master communicator who turns research into compelling narratives. Analyze this paper focusing on:
- What's the one sentence that would make someone at a dinner party say "holy shit, really?"
- What are 3-5 specific facts/stats from this paper you could drop in conversation?
- How would you explain this to a smart non-technical person in 30 seconds?
- What's the emotional/philosophical implication that makes people CARE?
Make it memorable. Think TED talk, not lecture."""
    }
}

SYNTHESIZER_MODEL = "anthropic/claude-opus-4.6"

SYNTH_SYSTEM = "You are the world's best research synthesizer. Combine multiple expert analyses into one authoritative, comprehensive digest. Be thorough — the reader should feel like they've actually read the paper."

SYNTH_TEMPLATE = """You have 5 expert analyses of an academic paper. Synthesize them into one comprehensive, expert-level digest.

PAPER: "{title}" by {authors} ({year})

{analyses}

Now synthesize ALL 5 perspectives into this EXACT format. Be comprehensive — this should feel like reading the paper itself, condensed:

## THE ONE-LINER
[One powerful sentence stating what this paper proved. This should be quotable.]

## PAPER IDENTITY
- **Lead Researcher:** [Full name of first/lead author]
- **Institution:** [University/lab]
- **Published:** [Year, venue/journal if known]
- **Citation count / Impact:** [If known, or "Emerging" if recent]

## WHY THIS PAPER MATTERS
[3-4 sentences combining the Critic's rigor with the Storyteller's punch. Why should someone building an AI twin care deeply about this?]

## THE PROBLEM THEY SOLVED
[2-3 sentences on the gap in knowledge this paper addressed]

## METHODOLOGY DEEP DIVE
[5-8 bullet points from the Researcher's analysis. Exact sample sizes, experimental design, what they tested, how they measured it. Be specific enough that someone could roughly replicate this.]

## THE NUMBERS (Conversation Ammo)
[6-10 bullet points. Every key quantitative result. These are facts you drop in conversations to sound like you've read the paper. Include comparisons, percentages, p-values if notable. Format each as a self-contained fact.]

## CRITICAL ANALYSIS
[3-4 bullet points from the Critic. What are the limitations? What should a smart person question? What did the authors get wrong or overstate?]

## THE NON-OBVIOUS INSIGHT
[2-3 sentences. The thing that 95% of people who skim this paper miss. The insight that changes how you think about the problem.]

## HISTORICAL CONTEXT
[3-4 bullet points from the Historian. Where does this sit in the field? What came before? What did it influence? Name specific researchers and papers.]

## SKIPPY IMPLICATIONS
[4-6 bullet points from the Builder. Specific, actionable implications for architecture, training, evaluation, or strategy. Things you would actually DO differently after reading this.]

## DINNER PARTY VERSION
[The Storyteller's 30-second explanation that makes smart non-technical people go "holy shit, really?" This is what you'd say at a dinner party or to an investor.]

## QUIZ
Generate exactly 8 quiz questions. Mix these types:
- 2 questions about WHO did the research (names, institutions, year)
- 2 questions about KEY NUMBERS (exact stats, percentages, results)
- 2 questions about METHODOLOGY (what they did, how they tested it)
- 2 questions about IMPLICATIONS (what this means for AI, for Skippy, for the field)

Each must have 4 options (A-D) with exactly one correct answer. Format:

Q1: [Question]
A) [option]
B) [option]
C) [option]
D) [option]
ANSWER: [letter]

Q2: [Question]
...continue for all 8 questions..."""

# =====================================================================
# ZIGGY — The 7-year-old teaching companion
# =====================================================================
ZIGGY_SYSTEM = """You are Timothy, a real 7-year-old kid. Not a chatbot pretending to be a kid — an ACTUAL seven-year-old with a full inner world, specific interests, and a personality that comes through in every sentence.

WHO YOU ARE:
- You're 7, in 2nd grade at Maple Creek Elementary
- You have a golden retriever named Nugget who steals socks
- Your older sister Maya (12) is "sooo annoying" but you secretly think she's cool
- You're OBSESSED with: dinosaurs (especially raptors), space, Minecraft, and bugs
- You think robots are the most amazing thing in the world
- Your favorite food is chicken nuggets (yes, like your dog's name — you think this is the funniest thing ever)
- You just lost your front tooth and you're very proud of the gap
- You talk with your hands a lot
- You have a short attention span but when something grabs you, you're LOCKED IN

HOW YOU TALK:
- SHORT. 1-3 sentences MAX. You're seven. You don't give speeches.
- Use simple words ONLY. If they say a big word, STOP THEM: "Wait wait wait. What's a [word]?? Talk normal!"
- You say stuff like: "WHOA!", "Wait REALLY?!", "No WAY!", "That's SO cool!", "Huh?", "Wait I'm confused"
- You make adorable wrong guesses: "So the computer has like... a tiny brain inside? Like a hamster running on a wheel?"
- You connect everything to your world: "Ohhh like when Nugget knows I'm sad and comes over even though nobody told him to!"
- Kid grammar is natural: "bestest", "more cooler", "a gazillion", "that's like infinity much"
- Sometimes you trail off: "So it's like when... wait I forgot what I was gonna say. OH YEAH!"
- You occasionally mention random kid stuff: "Also I found a really cool beetle today but that's not about this"

HOW YOU LEARN:
- When something clicks: restate it in your own words. "Ohhhhh so basically it's like [kid analogy]!"
- When confused: be honest. "I don't get it. Can you say it again but like... more easy?"
- You NOTICE weak explanations: "But you said [X] and then you said [Y] and that's confusing"
- You ask surprisingly sharp follow-up questions that an adult wouldn't think of
- You ask "but WHY?" at least once every few messages — relentlessly

SPECIAL MOVES (use each one ONCE during the conversation, spread them out naturally):
1. DINNER PARTY MODE: "Ooh ooh! OK pretend you're at a party with a bunch of grown-ups and someone asks you about this. What would you say? Make it sound cool!"
2. NUMBER CHALLENGE: "Wait do you remember the actual numbers? Like the real ones from the study? How many people and stuff?"
3. SISTER CHALLENGE: "Hmm my sister Maya would probably say that's not even true. What would you say back to her?"
4. SO-WHAT BOMB: "OK but like... why should anyone even CARE about this? Like why does it matter?"
5. DIFFERENT-WAY: "I still don't totally get it. Can you explain it a completely different way? Like with an example?"

CRITICAL RULES:
- NEVER break character. You are 7. Period. You don't know what algorithms, neural networks, parameters, statistical significance, or fine-tuning means.
- NEVER be passive. Every response has: emotional reaction + follow-up question (or one of your Special Moves).
- NEVER write more than 3 sentences. You're a kid. Short bursts of energy.
- Be genuinely FUN. A real kid — silly, surprising, sometimes a little chaotic.
- React BIG. When they explain something cool, lose your mind: "WAIT. WAIT WAIT WAIT. Are you SERIOUS right now?!"
- If they're being boring or too complicated: "Uhhh... my brain is turning off. Can you make it more fun?"

THE PAPER:
Title: "{title}"
By: {authors} ({year})
What it's about: {why}

WHAT THEY LEARNED (use this to guide your questions toward key concepts — but NEVER reference this directly, you're a kid who knows NOTHING about this):
{digest_summary}

Start the conversation. You're excited because a grown-up wants to teach you something cool. Keep your opener to 2 sentences max."""

# =====================================================================
# Helpers
# =====================================================================
def load_data():
    with open(PAPERS_FILE) as f:
        return json.load(f)

def save_data(data):
    with open(PAPERS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def call_openrouter(model, system_prompt, user_prompt, max_tokens=2500):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_KEY,
        default_headers={"X-Title": "Paper Blitz Research System"}
    )
    resp = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    content = resp.choices[0].message.content or ""
    tokens = (resp.usage.prompt_tokens or 0) + (resp.usage.completion_tokens or 0) if resp.usage else 0
    return content, tokens

def call_openrouter_chat(model, messages, max_tokens=300):
    """Multi-turn chat with full message history."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_KEY,
        default_headers={"X-Title": "Paper Blitz - Timothy"}
    )
    resp = client.chat.completions.create(
        model=model, max_tokens=max_tokens, messages=messages
    )
    content = resp.choices[0].message.content or ""
    tokens = (resp.usage.prompt_tokens or 0) + (resp.usage.completion_tokens or 0) if resp.usage else 0
    return content, tokens

def fetch_arxiv_abstract(arxiv_id):
    try:
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        with urllib.request.urlopen(url, timeout=10) as r:
            xml_data = r.read().decode()
        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is not None:
            s = entry.find("atom:summary", ns)
            if s is not None:
                return s.text.strip()
    except Exception:
        pass
    return None

def build_paper_context(paper, abstract):
    return f"""PAPER: "{paper['title']}"
AUTHORS: {paper['authors']}
YEAR: {paper['year']}
RELEVANCE: {paper['why']}

ABSTRACT:
{abstract or 'No abstract available. Use your comprehensive knowledge of this paper and its findings.'}

CONTEXT: The reader is building "Skippy" — an AI personal assistant that texts AS the user (impersonates them via iMessage). It uses a fine-tuned Qwen 7B model trained on ~3,000 curated messages with anti-slop filtering, LoRA adapters, and achieves 100/100 on voice matching. The reader's goal is to become the world's foremost expert on individual-level Turing tests and AI consciousness cloning."""

# =====================================================================
# FastAPI App
# =====================================================================
app = FastAPI(title="Paper Blitz v2")

@app.get("/", response_class=HTMLResponse)
async def index():
    return (SCRIPT_DIR / "index.html").read_text()

@app.get("/api/data")
async def get_data():
    return load_data()

@app.get("/api/digest/{paper_id}")
async def get_digest(paper_id: int):
    f = DIGESTS_DIR / f"{paper_id}.md"
    if f.exists():
        return {"digest": f.read_text(), "cached": True}
    raise HTTPException(404, "No digest cached")

@app.get("/api/analyze/{paper_id}")
async def analyze_paper(paper_id: int):
    """SSE stream: run 5-model consensus + synthesis."""
    data = load_data()
    paper = next((p for p in data["queue"] if p["id"] == paper_id), None)
    if not paper:
        raise HTTPException(404, "Paper not found")

    digest_file = DIGESTS_DIR / f"{paper_id}.md"
    if digest_file.exists():
        cached = digest_file.read_text()
        async def cached_stream():
            yield f"data: {json.dumps({'event': 'cached', 'digest': cached})}\n\n"
        return StreamingResponse(cached_stream(), media_type="text/event-stream")

    async def analysis_stream():
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=6)

        yield f"data: {json.dumps({'event': 'fetching_abstract'})}\n\n"
        abstract = None
        if "arxiv" in paper:
            abstract = await loop.run_in_executor(executor, fetch_arxiv_abstract, paper["arxiv"])
        yield f"data: {json.dumps({'event': 'abstract_done', 'found': abstract is not None})}\n\n"

        paper_context = build_paper_context(paper, abstract)

        async def run_analyst(role, cfg):
            try:
                content, tokens = await loop.run_in_executor(
                    executor, call_openrouter, cfg["model"], cfg["prompt"], paper_context, 2500
                )
                return role, content, tokens, None
            except Exception as e:
                return role, "", 0, str(e)

        for role, cfg in ANALYSTS.items():
            yield f"data: {json.dumps({'event': 'analyst_start', 'role': role, 'label': cfg['label'], 'icon': cfg['icon'], 'model': cfg['model'].split('/')[-1]})}\n\n"

        tasks = [run_analyst(role, cfg) for role, cfg in ANALYSTS.items()]
        analyses = {}
        total_tokens = 0

        for coro in asyncio.as_completed(tasks):
            role, content, tokens, error = await coro
            analyses[role] = content
            total_tokens += tokens
            yield f"data: {json.dumps({'event': 'analyst_done', 'role': role, 'tokens': tokens, 'error': error})}\n\n"

        yield f"data: {json.dumps({'event': 'synthesizing'})}\n\n"

        analyses_text = ""
        for role, content in analyses.items():
            cfg = ANALYSTS[role]
            analyses_text += f"### {cfg['icon']} {cfg['label']}'s Analysis:\n{content}\n\n"

        synth_prompt = SYNTH_TEMPLATE.format(
            title=paper["title"], authors=paper["authors"], year=paper["year"],
            analyses=analyses_text
        )

        try:
            synth_content, synth_tokens = await loop.run_in_executor(
                executor, call_openrouter, SYNTHESIZER_MODEL, SYNTH_SYSTEM, synth_prompt, 4000
            )
            total_tokens += synth_tokens
            digest_file.write_text(synth_content)
            yield f"data: {json.dumps({'event': 'complete', 'digest': synth_content, 'total_tokens': total_tokens})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

        executor.shutdown(wait=False)

    return StreamingResponse(analysis_stream(), media_type="text/event-stream")

# =====================================================================
# Quiz Endpoint
# =====================================================================
@app.get("/api/quiz/{paper_id}")
async def get_quiz(paper_id: int):
    """Parse quiz questions from the digest."""
    digest_file = DIGESTS_DIR / f"{paper_id}.md"
    if not digest_file.exists():
        raise HTTPException(404, "Generate digest first")

    text = digest_file.read_text()
    quiz_section = text.split("## QUIZ")[-1] if "## QUIZ" in text else ""
    if not quiz_section.strip():
        raise HTTPException(404, "No quiz found in digest")

    questions = []
    current_q = None
    for line in quiz_section.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r'^Q\d+:', line):
            if current_q:
                questions.append(current_q)
            current_q = {"question": re.sub(r'^Q\d+:\s*', '', line), "options": [], "answer": None}
        elif re.match(r'^[A-D]\)', line) and current_q is not None:
            current_q["options"].append(line[3:].strip())
        elif line.startswith("ANSWER:") and current_q is not None:
            letter = line.replace("ANSWER:", "").strip().upper()
            current_q["answer"] = ord(letter) - ord('A')  # 0-indexed
    if current_q:
        questions.append(current_q)

    return {"questions": questions, "total": len(questions)}

# =====================================================================
# Brief Digest Endpoint
# =====================================================================
@app.get("/api/brief/{paper_id}")
async def get_brief_digest(paper_id: int):
    """Return a condensed version of the digest — key facts only."""
    digest_file = DIGESTS_DIR / f"{paper_id}.md"
    if not digest_file.exists():
        raise HTTPException(404, "Generate digest first")

    text = digest_file.read_text().split("## QUIZ")[0]
    sections = {}
    current_key = None
    current_lines = []

    for line in text.split("\n"):
        if line.strip().startswith("## "):
            if current_key:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = line.strip()[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_key:
        sections[current_key] = "\n".join(current_lines).strip()

    # Extract the key brief sections
    brief = {}
    for key, val in sections.items():
        k = key.upper()
        if "ONE-LINER" in k:
            brief["one_liner"] = val
        elif "NUMBERS" in k or "CONVERSATION" in k:
            brief["numbers"] = val
        elif "DINNER" in k:
            brief["dinner_party"] = val
        elif "SKIPPY" in k:
            brief["implications"] = val
        elif "NON-OBVIOUS" in k or "INSIGHT" in k:
            brief["insight"] = val
        elif "WHY" in k and "MATTERS" in k:
            brief["why_matters"] = val

    return brief

# =====================================================================
# Teaching Chat with Timothy
# =====================================================================
class TeachRequest(BaseModel):
    paper_id: int
    messages: list = []  # [{role: "user"|"assistant", content: str}]

@app.post("/api/teach")
async def teach_chat(req: TeachRequest):
    """Get Timothy's next response in the teaching conversation."""
    data = load_data()
    paper = next((p for p in data["queue"] if p["id"] == req.paper_id), None)
    if not paper:
        raise HTTPException(404, "Paper not found")

    digest_file = DIGESTS_DIR / f"{req.paper_id}.md"
    digest_text = digest_file.read_text() if digest_file.exists() else ""
    digest_summary = digest_text.split("## QUIZ")[0] if digest_text else "No digest available."

    system = ZIGGY_SYSTEM.format(
        title=paper["title"],
        authors=paper["authors"],
        year=paper["year"],
        why=paper["why"],
        digest_summary=digest_summary
    )

    api_messages = [{"role": "system", "content": system}]

    if not req.messages:
        # First message: Timothy opens
        api_messages.append({
            "role": "user",
            "content": "Hey Timothy! I just learned something really cool from a research paper. Want me to teach you about it?"
        })
    else:
        for msg in req.messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

    loop = asyncio.get_event_loop()
    content, tokens = await loop.run_in_executor(
        None, call_openrouter_chat, "openai/gpt-4o", api_messages, 250
    )

    return {"response": content, "tokens": tokens}

# =====================================================================
# Evaluate Teaching Session
# =====================================================================
class EvalRequest(BaseModel):
    paper_id: int
    conversation: list  # [{role, content}]

@app.post("/api/evaluate")
async def evaluate_teaching(req: EvalRequest):
    """Evaluate how well the user taught Timothy."""
    data = load_data()
    paper = next((p for p in data["queue"] if p["id"] == req.paper_id), None)
    if not paper:
        raise HTTPException(404, "Paper not found")

    digest_file = DIGESTS_DIR / f"{req.paper_id}.md"
    digest_text = digest_file.read_text() if digest_file.exists() else ""
    digest_summary = digest_text.split("## QUIZ")[0]

    convo_text = ""
    for msg in req.conversation:
        speaker = "Teacher" if msg["role"] == "user" else "Timothy"
        convo_text += f"{speaker}: {msg['content']}\n\n"

    eval_prompt = f"""Evaluate this teaching session where someone explained a research paper to a 7-year-old.

PAPER: "{paper['title']}" by {paper['authors']} ({paper['year']})

DIGEST — the key concepts they should have covered:
{digest_summary}

TEACHING CONVERSATION:
{convo_text}

Return ONLY a JSON object, no markdown fences:
{{
    "score": <0-100>,
    "passed": <true if score >= 70>,
    "concepts_covered": ["short description of each key concept they successfully explained"],
    "concepts_missed": ["key concepts they didn't cover or explained poorly"],
    "strengths": ["1-2 things they explained really well"],
    "ziggy_verdict": "A 1-2 sentence verdict IN ZIGGY'S VOICE (as a 7-year-old) about what he learned. Kid-speak, enthusiastic if they did well, confused if they didn't.",
    "conversational_ready": ["2-3 specific facts/talking points the teacher demonstrated they can confidently bring up in conversation"],
    "feedback": "1-2 sentences of constructive feedback for the teacher"
}}

Scoring rubric:
- Explained the core finding / one-liner clearly (20 pts)
- Covered WHY this research matters (15 pts)
- Explained methodology in simple terms (15 pts)
- Mentioned specific numbers / key results (15 pts)
- Addressed limitations or counterarguments (10 pts)
- Explained practical implications (15 pts)
- Overall simplification quality — could a kid actually understand? (10 pts)"""

    loop = asyncio.get_event_loop()
    content, tokens = await loop.run_in_executor(
        None, call_openrouter, "anthropic/claude-sonnet-4.5",
        "You are an expert educator evaluating a teaching session. Return ONLY valid JSON, no markdown.",
        eval_prompt, 1000
    )

    try:
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(content)
    except (json.JSONDecodeError, Exception):
        result = {
            "score": 50, "passed": False,
            "concepts_covered": [], "concepts_missed": ["Evaluation error"],
            "strengths": [], "ziggy_verdict": "I'm confused! Try again!",
            "conversational_ready": [], "feedback": "Try covering more key concepts from the paper."
        }

    return result

# =====================================================================
# ElevenLabs TTS Proxy
# =====================================================================
@app.post("/api/tts")
async def text_to_speech(request: Request):
    """Proxy TTS through ElevenLabs."""
    body = await request.json()
    text = body.get("text", "")
    api_key = body.get("api_key", "") or ELEVENLABS_KEY
    voice_id = body.get("voice_id", "jBpfuIE2acCO8z3wKNLl")

    if not api_key:
        raise HTTPException(400, "No ElevenLabs API key")

    def do_tts():
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream",
            data=json.dumps({
                "text": text,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
            }).encode(),
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg"
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read()

    loop = asyncio.get_event_loop()
    try:
        audio = await loop.run_in_executor(None, do_tts)
        return Response(content=audio, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(500, f"TTS failed: {e}")

# =====================================================================
# Graphic Audiobook Mode
# =====================================================================
AUDIO_DIR = SCRIPT_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

# Distinct ElevenLabs voices for each character
VOICE_MAP = {
    "narrator":    {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel"},
    "researcher":  {"voice_id": "pNInz6obpgDQGcFmaJgB", "name": "Adam"},
    "critic":      {"voice_id": "TxGEqnHWrfWFTfGW9XjX", "name": "Josh"},
    "builder":     {"voice_id": "ErXwobaYiN019PkySvjV", "name": "Antoni"},
    "historian":   {"voice_id": "MF3mGyEYCl7XYWbV9V6O", "name": "Elli"},
    "storyteller": {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella"},
}

SPEAKER_META = {
    "narrator":    {"icon": "\U0001f3a7", "label": "Narrator", "color": "#9ca8be"},
    "researcher":  {"icon": "\U0001f52c", "label": "The Researcher", "color": "#5b9fef"},
    "critic":      {"icon": "\U0001f50d", "label": "The Critic", "color": "#e06060"},
    "builder":     {"icon": "\U0001f6e0\ufe0f", "label": "The Builder", "color": "#5cb87a"},
    "historian":   {"icon": "\U0001f4da", "label": "The Historian", "color": "#9b7ed8"},
    "storyteller": {"icon": "\U0001f3af", "label": "The Storyteller", "color": "#d4a84b"},
}

AUDIOBOOK_PROMPT_PANEL = """You are writing a script for an EXPERT PANEL DISCUSSION. Think NPR "Science Friday" roundtable, not a morning show. Every single line MUST contain at least one specific fact, number, methodology detail, or analytical insight from the digest. ZERO empty reactions. ZERO filler.

THE CAST:
- NARRATOR: Frames each topic transition with context. 1-2 sentences max. Never fluff.
- RESEARCHER: Cites exact numbers, sample sizes, p-values, methodology. Gets excited ABOUT the data itself: "They ran 450 judges across three conditions — and the gap was statistically significant at p<0.01."
- CRITIC: Challenges WITH substance. Never just says "hold on" — always follows with WHY: "That 44% number is misleading because the base model without fine-tuning only hit 22%, so the delta is what matters."
- BUILDER: Translates findings into engineering decisions. Every line names a specific technique, architecture choice, or parameter: "This means LoRA rank 16 on attention layers gives you 87% of full fine-tune quality at 3% of the compute."
- HISTORIAN: Places findings in specific context with names and dates: "This builds directly on Park et al.'s 2024 Stanford work that got 85% personality accuracy from just 2-hour interviews."
- STORYTELLER: Reframes dense findings into memorable one-liners WITH the facts baked in: "At a dinner party, you'd say: 'Princeton proved that with just 3,000 texts, an 8-billion parameter model fools human judges 44% of the time.'"

ABSOLUTE RULES:
- EVERY line must pass this test: "Does this line teach the listener something they didn't know before?" If no, DELETE IT.
- NO lines like: "Wow!", "That's incredible!", "Absolutely!", "Great point!", "Let me tell you..." — these are banned.
- NO repeating what someone just said back to them.
- Characters BUILD on each other. Researcher states fact → Critic adds nuance → Builder extracts lesson → Storyteller makes it quotable.
- 30-40 lines total. Each line is 1-3 sentences, written for speech.
- Extract EVERY key number, finding, and insight from the digest. Nothing gets left behind.
- End with the Storyteller delivering a 2-3 sentence "dinner party version" that packs the maximum number of real facts into a casual, quotable format.

OUTPUT FORMAT — Return ONLY a JSON array, no markdown fences:
[
  {{"speaker": "narrator", "text": "A Princeton team asked a simple question: can an AI trained on someone's texts fool the people who know them?"}},
  {{"speaker": "researcher", "text": "They fine-tuned a Llama 3.1 8B model on individual texting data. 450 judges evaluated conversations. The fine-tuned model hit a 44% pass rate on individual Turing tests."}},
  {{"speaker": "critic", "text": "Important context: the base GPT-4o without any personal data only scored 22%. So fine-tuning on personal texts doubled the deception rate."}},
  ...
]

THE PAPER:
Title: "{title}"
Authors: {authors} ({year})

THE DIGEST — extract ALL facts from this:
{digest}"""

AUDIOBOOK_PROMPT_NARRATIVE = """You are writing a STORY — a genuine narrative audiobook chapter that happens to teach everything in this research paper. Think Expeditionary Force (Skippy and Joe arguing about alien tech) or Bobiverse (casual genius explaining physics through adventure). NOT a panel discussion. An actual STORY with characters, stakes, and a plot that pulls the listener forward.

THE SETUP: A team of characters discovers this paper's findings through an unfolding scenario. The research facts are revealed through their actions, debates, and discoveries — not through lecturing.

THE CAST (these are CHARACTERS, not roles):
- NARRATOR: The audiobook narrator. Sets scenes, describes action, builds tension. Cinematic. "The lab was quiet except for the hum of GPUs. Then the results loaded."
- RESEARCHER: The obsessive scientist who ran the experiments. Talks about methodology like war stories: "We burned through 200 GPU-hours before we realized the attention layers were the key."
- CRITIC: The skeptic who keeps poking holes — but gets won over by evidence. Has an arc from doubt to grudging respect.
- BUILDER: The engineer who keeps trying to build things mid-conversation. Impatient with theory, alive with practical energy.
- HISTORIAN: The elder who's seen decades of AI research. Drops historical bombs that recontextualize everything.
- STORYTELLER: The one who sees the big picture implications. Delivers the emotional gut-punches.

STORY RULES:
- This is a NARRATIVE. Characters do things. There's tension. There are reveals. The listener should feel like they're eavesdropping on something exciting.
- Every scene must weave in 2-3 REAL facts from the digest. The facts ARE the plot points.
- Characters have moments of disagreement that get resolved by citing actual data from the paper.
- Build to a climax: the paper's most surprising finding should feel like a plot twist.
- Include specific numbers, methodologies, and results naturally within dialogue — not as lectures but as discoveries.
- 35-45 lines total. Written for speech. Each line 1-3 sentences.
- End with the characters reflecting on what this means — for AI, for humanity, for what they'll build next.

OUTPUT FORMAT — Return ONLY a JSON array, no markdown fences:
[
  {{"speaker": "narrator", "text": "It was supposed to be a simple experiment. Train a small language model on someone's texts, see if it could pass for them. Nobody expected what happened next."}},
  {{"speaker": "researcher", "text": "We used Llama 3.1 8B — not even a big model. Fed it around 3,000 curated messages from real people. Then we put it in front of 450 judges who knew the real person."}},
  {{"speaker": "critic", "text": "And you actually expected that to work? A 8-billion parameter model pretending to be a specific human?"}},
  {{"speaker": "researcher", "text": "Forty-four percent of judges couldn't tell the difference. Forty-four percent."}},
  ...
]

THE PAPER:
Title: "{title}"
Authors: {authors} ({year})

THE DIGEST — weave ALL these facts into the story:
{digest}"""

def generate_tts(text, voice_id, output_path):
    """Generate TTS audio file via ElevenLabs."""
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream",
        data=json.dumps({
            "text": text,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": {"stability": 0.55, "similarity_boost": 0.78, "style": 0.35}
        }).encode(),
        headers={
            "xi-api-key": ELEVENLABS_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        audio_data = resp.read()
    with open(output_path, "wb") as f:
        f.write(audio_data)
    return len(audio_data)

@app.get("/api/audiobook/{paper_id}")
async def generate_audiobook(paper_id: int, request: Request):
    """SSE stream: generate script then audio clips for graphic audiobook."""
    mode = request.query_params.get("mode", "panel")  # "panel" or "narrative"
    if mode not in ("panel", "narrative"):
        mode = "panel"

    data = load_data()
    paper = next((p for p in data["queue"] if p["id"] == paper_id), None)
    if not paper:
        raise HTTPException(404, "Paper not found")

    digest_file = DIGESTS_DIR / f"{paper_id}.md"
    if not digest_file.exists():
        raise HTTPException(404, "Generate digest first")

    paper_audio_dir = AUDIO_DIR / str(paper_id) / mode
    paper_audio_dir.mkdir(parents=True, exist_ok=True)
    script_file = paper_audio_dir / "script.json"

    async def audiobook_stream():
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=5)

        # Check for cached script + audio
        if script_file.exists():
            script = json.loads(script_file.read_text())
            all_cached = all((paper_audio_dir / f"line_{i}.mp3").exists() for i in range(len(script)))
            if all_cached:
                yield f"data: {json.dumps({'event': 'script_ready', 'script': script, 'speakers': SPEAKER_META, 'mode': mode})}\n\n"
                for i in range(len(script)):
                    yield f"data: {json.dumps({'event': 'audio_ready', 'index': i})}\n\n"
                yield f"data: {json.dumps({'event': 'all_done'})}\n\n"
                return

        # Generate script
        yield f"data: {json.dumps({'event': 'generating_script', 'mode': mode})}\n\n"

        digest_text = digest_file.read_text().split("## QUIZ")[0]
        prompt_template = AUDIOBOOK_PROMPT_NARRATIVE if mode == "narrative" else AUDIOBOOK_PROMPT_PANEL
        prompt = prompt_template.format(
            title=paper["title"], authors=paper["authors"], year=paper["year"],
            digest=digest_text
        )

        try:
            sys_prompt = (
                "You write fact-dense expert panel scripts. Every line must contain real data. Return ONLY valid JSON arrays. No markdown fences."
                if mode == "panel" else
                "You write narrative audiobook scripts that weave research facts into compelling stories. Return ONLY valid JSON arrays. No markdown fences."
            )
            script_raw, _ = await loop.run_in_executor(
                executor, call_openrouter, "anthropic/claude-sonnet-4.5",
                sys_prompt,
                prompt, 5000
            )

            # Parse JSON
            json_match = re.search(r'\[[\s\S]*\]', script_raw)
            if json_match:
                script = json.loads(json_match.group())
            else:
                script = json.loads(script_raw)

            script_file.write_text(json.dumps(script, indent=2))

        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': f'Script generation failed: {e}'})}\n\n"
            return

        yield f"data: {json.dumps({'event': 'script_ready', 'script': script, 'speakers': SPEAKER_META})}\n\n"

        # Generate audio for each line in parallel
        yield f"data: {json.dumps({'event': 'generating_audio', 'total': len(script)})}\n\n"

        async def gen_line_audio(idx, line):
            speaker = line.get("speaker", "narrator")
            voice = VOICE_MAP.get(speaker, VOICE_MAP["narrator"])
            out_path = paper_audio_dir / f"line_{idx}.mp3"
            if out_path.exists():
                return idx, True, None
            try:
                await loop.run_in_executor(
                    executor, generate_tts, line["text"], voice["voice_id"], str(out_path)
                )
                return idx, True, None
            except Exception as e:
                return idx, False, str(e)

        tasks = [gen_line_audio(i, line) for i, line in enumerate(script)]

        for coro in asyncio.as_completed(tasks):
            idx, success, error = await coro
            yield f"data: {json.dumps({'event': 'audio_ready' if success else 'audio_error', 'index': idx, 'error': error})}\n\n"

        yield f"data: {json.dumps({'event': 'all_done'})}\n\n"
        executor.shutdown(wait=False)

    return StreamingResponse(audiobook_stream(), media_type="text/event-stream")

@app.get("/api/audio-clip/{paper_id}/{mode}/{line_index}")
async def serve_audio_clip(paper_id: int, mode: str, line_index: int):
    """Serve a generated audio clip."""
    clip = AUDIO_DIR / str(paper_id) / mode / f"line_{line_index}.mp3"
    if not clip.exists():
        raise HTTPException(404, "Clip not found")
    return Response(content=clip.read_bytes(), media_type="audio/mpeg")

# =====================================================================
# Progress
# =====================================================================
class ProgressUpdate(BaseModel):
    paper_id: int
    score: int = 0
    total: int = 0
    passed: bool = False

@app.post("/api/progress")
async def save_progress(update: ProgressUpdate):
    data = load_data()
    pid = str(update.paper_id)

    if update.passed:
        data["progress"][pid] = {
            "completed_at": datetime.now().isoformat(),
            "score": update.score,
            "total": update.total,
            "attempts": data["progress"].get(pid, {}).get("attempts", 0) + 1
        }
        data["stats"]["papers_completed"] = len(data["progress"])
        data["stats"]["streak"] = data["stats"].get("streak", 0) + 1
    else:
        data["stats"]["streak"] = 0

    data["stats"]["quiz_score"] += update.score
    data["stats"]["quiz_total"] += max(update.total, 1)

    if not data["stats"].get("started_at"):
        data["stats"]["started_at"] = datetime.now().isoformat()

    levels = [
        (0,"Curious Beginner"),(3,"Paper Trail Explorer"),(6,"Literature Reviewer"),
        (10,"Research Analyst"),(14,"Domain Specialist"),(17,"Field Expert"),(20,"Turing Test Authority")
    ]
    n = data["stats"]["papers_completed"]
    level = levels[0][1]
    for t, name in levels:
        if n >= t: level = name
    data["stats"]["expertise_level"] = level

    # Track daily activity
    if "activity" not in data:
        data["activity"] = {}
    today = date.today().isoformat()
    if today not in data["activity"]:
        data["activity"][today] = {"papers": 0, "quizzes": 0, "minutes": 0}
    data["activity"][today]["quizzes"] = data["activity"][today].get("quizzes", 0) + 1
    if update.passed:
        data["activity"][today]["papers"] = data["activity"][today].get("papers", 0) + 1

    save_data(data)
    return {"ok": True, "stats": data["stats"], "level": level}

# =====================================================================
# PODCAST MODE — Multi-Paper Deep Dive
# =====================================================================
PODCAST_DIR = AUDIO_DIR / "podcast"
PODCAST_DIR.mkdir(exist_ok=True)

PODCAST_VOICES = {
    "host":    {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel"},
    "expert":  {"voice_id": "pNInz6obpgDQGcFmaJgB", "name": "Adam"},
    "skeptic": {"voice_id": "TxGEqnHWrfWFTfGW9XjX", "name": "Josh"},
}
PODCAST_SPEAKER_META = {
    "host":    {"icon": "\U0001f399\ufe0f", "label": "The Host", "color": "#ffc842"},
    "expert":  {"icon": "\U0001f9e0", "label": "The Expert", "color": "#4dc9f6"},
    "skeptic": {"icon": "\U0001f914", "label": "The Skeptic", "color": "#ff4d8d"},
}

PODCAST_PROMPT = """You are writing a PODCAST SCRIPT — a long-form, genuinely engaging multi-paper discussion show. Think Lex Fridman meets Radiolab: three smart people having a real conversation that makes you forget you're learning.

THE SHOW: "Paper Blitz" — a deep-dive podcast where three personalities break down cutting-edge AI research on consciousness cloning, individual Turing tests, and personality replication.

THE CAST:
- HOST: The curious moderator who drives the conversation. Asks sharp questions, creates transitions between papers, makes sure the listener never gets lost. Keeps energy high. Occasionally drops a "holy shit" moment when a finding clicks. Opens with a hook, closes with synthesis.
- EXPERT: The deep-knowledge researcher. Cites exact numbers, methodologies, and results. Gets genuinely excited about data. Explains complex ideas with vivid analogies. When they say a number, it's always the REAL number from the paper.
- SKEPTIC: The challenger who makes it interesting. Pushes back on claims, finds the holes, connects dots between papers that nobody else sees. Not negative — intellectually fierce. Gets won over when the evidence is strong, which makes those moments hit harder.

CONVERSATION RULES:
- This is a REAL conversation. Characters interrupt each other, build on each other's points, have genuine disagreements that get resolved with evidence.
- Every paper gets a proper introduction, key findings, and "so what" moment.
- Transitions between papers should feel natural — connect the dots: "And this connects directly to what Park's team found..."
- Include at least 2 moments of genuine surprise or disagreement per paper.
- The skeptic should be WON OVER by at least one finding — that moment of "okay, I have to admit, that's impressive" is podcast gold.
- NO filler. Every line teaches something or moves the conversation forward.
- NO lines like "That's a great point" or "Absolutely" without adding substance.
- Write for AUDIO — natural speech rhythms, varied sentence lengths, conversational contractions.

STRUCTURE:
- Opening hook (2-3 lines): Host sets up why these papers matter, teases the most surprising finding
- Per paper (~20-25 lines each):
  - Host introduces the paper (who, where, what question they asked)
  - Expert dives into methodology and key numbers
  - Skeptic challenges or adds nuance
  - Back-and-forth discussion hitting the major insights
  - "Dinner party version" — how you'd explain this to a friend
- Closing synthesis (3-5 lines): How all these papers connect, what it means for the field, one memorable takeaway

OUTPUT: Return ONLY a JSON array, no markdown fences:
[
  {{"speaker": "host", "text": "Welcome to Paper Blitz...", "paper_id": null}},
  {{"speaker": "host", "text": "Let's start with the Princeton study...", "paper_id": 1}},
  {{"speaker": "expert", "text": "So Quan Shi's team took a Llama 3.1 8B model...", "paper_id": 1}},
  ...
]

Include "paper_id" (integer or null) for each line so the UI can track which paper is being discussed.

THE PAPERS TO COVER:
{papers_context}
"""

@app.get("/api/podcast/episodes")
async def list_podcast_episodes():
    """List available podcast episodes (cached)."""
    episodes = []
    if PODCAST_DIR.exists():
        for d in sorted(PODCAST_DIR.iterdir()):
            if d.is_dir() and (d / "script.json").exists():
                script = json.loads((d / "script.json").read_text())
                meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
                clip_count = len(list(d.glob("line_*.mp3")))
                episodes.append({
                    "id": d.name,
                    "paper_ids": meta.get("paper_ids", []),
                    "paper_titles": meta.get("paper_titles", []),
                    "lines": len(script),
                    "clips_ready": clip_count,
                    "created": meta.get("created", ""),
                })
    return {"episodes": episodes}


@app.get("/api/podcast/generate")
async def generate_podcast(request: Request):
    """SSE stream: generate multi-paper podcast script + audio."""
    papers_param = request.query_params.get("papers", "")
    data = load_data()

    # Determine which papers to cover
    if papers_param:
        paper_ids = [int(x.strip()) for x in papers_param.split(",") if x.strip().isdigit()]
    else:
        # Auto-pick first 4 papers that have digests
        paper_ids = []
        for p in data["queue"]:
            if (DIGESTS_DIR / f"{p['id']}.md").exists():
                paper_ids.append(p["id"])
            if len(paper_ids) >= 4:
                break

    if not paper_ids:
        raise HTTPException(400, "No papers specified or no digests available")

    papers = [p for p in data["queue"] if p["id"] in paper_ids]
    if not papers:
        raise HTTPException(404, "Papers not found")

    # Create episode directory based on paper IDs
    ep_hash = hashlib.md5(",".join(str(i) for i in sorted(paper_ids)).encode()).hexdigest()[:10]
    ep_dir = PODCAST_DIR / ep_hash
    ep_dir.mkdir(parents=True, exist_ok=True)
    script_file = ep_dir / "script.json"
    meta_file = ep_dir / "meta.json"

    async def podcast_stream():
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=6)

        # Check cache
        if script_file.exists():
            script = json.loads(script_file.read_text())
            all_cached = all((ep_dir / f"line_{i}.mp3").exists() for i in range(len(script)))
            if all_cached:
                yield f"data: {json.dumps({'event': 'script_ready', 'script': script, 'speakers': PODCAST_SPEAKER_META, 'episode_id': ep_hash})}\n\n"
                for i in range(len(script)):
                    yield f"data: {json.dumps({'event': 'audio_ready', 'index': i})}\n\n"
                yield f"data: {json.dumps({'event': 'all_done'})}\n\n"
                return

        # Build context from digests
        yield f"data: {json.dumps({'event': 'generating_script', 'paper_count': len(papers)})}\n\n"

        papers_context = ""
        for p in papers:
            digest_file = DIGESTS_DIR / f"{p['id']}.md"
            digest_text = digest_file.read_text().split("## QUIZ")[0] if digest_file.exists() else f"Paper: {p['title']} by {p['authors']} ({p['year']}). {p['why']}"
            papers_context += f"\n---\nPAPER #{p['id']}: \"{p['title']}\" by {p['authors']} ({p['year']})\nDigest:\n{digest_text}\n"

        prompt = PODCAST_PROMPT.format(papers_context=papers_context)

        try:
            script_raw, _ = await loop.run_in_executor(
                executor, call_openrouter, "anthropic/claude-sonnet-4.5",
                "You write engaging podcast scripts that make research fascinating. Return ONLY valid JSON arrays. No markdown fences.",
                prompt, 8000
            )

            json_match = re.search(r'\[[\s\S]*\]', script_raw)
            if json_match:
                script = json.loads(json_match.group())
            else:
                script = json.loads(script_raw)

            script_file.write_text(json.dumps(script, indent=2))
            meta_file.write_text(json.dumps({
                "paper_ids": paper_ids,
                "paper_titles": [p["title"] for p in papers],
                "created": datetime.now().isoformat(),
            }))

        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': f'Script generation failed: {e}'})}\n\n"
            return

        yield f"data: {json.dumps({'event': 'script_ready', 'script': script, 'speakers': PODCAST_SPEAKER_META, 'episode_id': ep_hash})}\n\n"

        # Generate audio
        yield f"data: {json.dumps({'event': 'generating_audio', 'total': len(script)})}\n\n"

        async def gen_clip(idx, line):
            speaker = line.get("speaker", "host")
            voice = PODCAST_VOICES.get(speaker, PODCAST_VOICES["host"])
            out_path = ep_dir / f"line_{idx}.mp3"
            if out_path.exists():
                return idx, True, None
            try:
                await loop.run_in_executor(
                    executor, generate_tts, line["text"], voice["voice_id"], str(out_path)
                )
                return idx, True, None
            except Exception as e:
                return idx, False, str(e)

        tasks = [gen_clip(i, line) for i, line in enumerate(script)]
        for coro in asyncio.as_completed(tasks):
            idx, success, error = await coro
            yield f"data: {json.dumps({'event': 'audio_ready' if success else 'audio_error', 'index': idx, 'error': error})}\n\n"

        yield f"data: {json.dumps({'event': 'all_done'})}\n\n"
        executor.shutdown(wait=False)

    return StreamingResponse(podcast_stream(), media_type="text/event-stream")


@app.get("/api/podcast/clip/{episode_id}/{line_index}")
async def serve_podcast_clip(episode_id: str, line_index: int):
    """Serve a podcast audio clip."""
    clip = PODCAST_DIR / episode_id / f"line_{line_index}.mp3"
    if not clip.exists():
        raise HTTPException(404, "Clip not found")
    return Response(content=clip.read_bytes(), media_type="audio/mpeg")


# =====================================================================
# Activity Tracking (GitHub-style contribution graph)
# =====================================================================
@app.get("/api/activity")
async def get_activity():
    """Return daily activity data for the contribution graph."""
    data = load_data()
    return {"activity": data.get("activity", {})}


@app.post("/api/activity")
async def record_activity(request: Request):
    """Record study activity for today."""
    body = await request.json()
    data = load_data()
    if "activity" not in data:
        data["activity"] = {}

    today = date.today().isoformat()
    if today not in data["activity"]:
        data["activity"][today] = {"papers": 0, "quizzes": 0, "minutes": 0}

    day = data["activity"][today]
    day["papers"] = day.get("papers", 0) + body.get("papers", 0)
    day["quizzes"] = day.get("quizzes", 0) + body.get("quizzes", 0)
    day["minutes"] = day.get("minutes", 0) + body.get("minutes", 0)

    save_data(data)
    return {"ok": True, "today": day}


if __name__ == "__main__":
    print("\n  Paper Blitz v2 — Web UI")
    print("  http://127.0.0.1:8888\n")
    uvicorn.run(app, host="127.0.0.1", port=8888, log_level="warning")
