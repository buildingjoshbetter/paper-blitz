#!/usr/bin/env python3
"""
Paper Blitz v2 — Multi-model consensus research system
5 AI models analyze each paper in parallel, then a 6th synthesizes.
8-10 quiz questions. Forced re-review on failure. Cumulative review every 3 papers.

Usage:
    python3 blitz.py              # Start next paper in queue
    python3 blitz.py --status     # Show progress dashboard
    python3 blitz.py --quiz       # Random review from completed papers
    python3 blitz.py --map        # Show knowledge graph
    python3 blitz.py --pick N     # Jump to paper #N
    python3 blitz.py --review     # Re-review a failed paper
"""

import json, os, sys, textwrap, random, time, asyncio, concurrent.futures
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# --- Config ---
SCRIPT_DIR = Path(__file__).parent
PAPERS_FILE = SCRIPT_DIR / "papers.json"
DIGESTS_DIR = SCRIPT_DIR / "digests"
DIGESTS_DIR.mkdir(exist_ok=True)

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

PASS_THRESHOLD = 0.7  # Must get 70%+ to pass a paper

# The 5 analyst models + 1 synthesizer (same architecture as K-LLM)
ANALYSTS = {
    "researcher": {
        "model": "anthropic/claude-sonnet-4.5",
        "label": "The Researcher",
        "icon": "🔬",
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
        "icon": "🔍",
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
        "icon": "🛠️",
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
        "icon": "📚",
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
        "icon": "🎯",
        "prompt": """You are a master communicator who turns research into compelling narratives. Analyze this paper focusing on:
- What's the one sentence that would make someone at a dinner party say "holy shit, really?"
- What are 3-5 specific facts/stats from this paper you could drop in conversation?
- How would you explain this to a smart non-technical person in 30 seconds?
- What's the emotional/philosophical implication that makes people CARE?
Make it memorable. Think TED talk, not lecture."""
    }
}

SYNTHESIZER = {
    "model": "anthropic/claude-opus-4.6",
    "label": "Synthesizer",
    "max_tokens": 4000
}

# Colors
class C:
    GOLD = "\033[38;2;201;169;98m"
    GREEN = "\033[38;2;107;127;94m"
    RED = "\033[38;2;197;48;48m"
    BLUE = "\033[38;2;100;149;237m"
    PURPLE = "\033[38;2;160;120;200m"
    DIM = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    WHITE = "\033[97m"

LEVELS = [
    (0, "Curious Beginner"),
    (3, "Paper Trail Explorer"),
    (6, "Literature Reviewer"),
    (10, "Research Analyst"),
    (14, "Domain Specialist"),
    (17, "Field Expert"),
    (20, "Turing Test Authority"),
]

def get_level(n):
    level = LEVELS[0][1]
    for t, name in LEVELS:
        if n >= t: level = name
    return level

def load_data():
    with open(PAPERS_FILE) as f: return json.load(f)

def save_data(data):
    with open(PAPERS_FILE, "w") as f: json.dump(data, f, indent=2)

def clear():
    os.system("clear")

def wrap(text, width=78, indent="  "):
    lines = text.split("\n")
    out = []
    for line in lines:
        if line.strip() == "": out.append("")
        else: out.extend(textwrap.wrap(line, width=width, initial_indent=indent, subsequent_indent=indent))
    return "\n".join(out)

def header():
    print(f"""
{C.GOLD}  ╔══════════════════════════════════════════════════════════╗
  ║  {C.WHITE}P A P E R   B L I T Z   v 2{C.GOLD}                             ║
  ║  {C.DIM}Multi-Model Consensus Research System{C.GOLD}                   ║
  ╚══════════════════════════════════════════════════════════╝{C.RESET}
""")

# --- OpenRouter API ---
def call_openrouter(model, system_prompt, user_prompt, max_tokens=2500):
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY,
                    default_headers={"X-Title": "Paper Blitz Research System"})
    resp = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    )
    content = resp.choices[0].message.content or ""
    tokens = (resp.usage.prompt_tokens or 0) + (resp.usage.completion_tokens or 0) if resp.usage else 0
    return content, tokens

def fetch_arxiv_abstract(arxiv_id):
    import urllib.request, xml.etree.ElementTree as ET
    try:
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        with urllib.request.urlopen(url, timeout=10) as r:
            xml_data = r.read().decode()
        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is not None:
            s = entry.find("atom:summary", ns)
            if s is not None: return s.text.strip()
    except: pass
    return None

# --- Multi-model consensus digest ---
def run_consensus_digest(paper, abstract):
    """Run 5 analyst models in parallel, then synthesize into mega-digest."""
    paper_context = f"""PAPER: "{paper['title']}"
AUTHORS: {paper['authors']}
YEAR: {paper['year']}
RELEVANCE: {paper['why']}

ABSTRACT:
{abstract or 'No abstract available. Use your comprehensive knowledge of this paper and its findings.'}

CONTEXT: The reader is building "Skippy" — an AI personal assistant that texts AS the user (impersonates them via iMessage). It uses a fine-tuned Qwen 7B model trained on ~3,000 curated messages with anti-slop filtering, LoRA adapters, and achieves 100/100 on voice matching. The reader's goal is to become the world's foremost expert on individual-level Turing tests and AI consciousness cloning."""

    total_tokens = 0
    analyses = {}

    # Run all 5 analysts in parallel
    print(f"\n  {C.BOLD}Running 5-model consensus analysis...{C.RESET}\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for role, cfg in ANALYSTS.items():
            futures[executor.submit(call_openrouter, cfg["model"], cfg["prompt"], paper_context)] = role
            print(f"  {cfg['icon']} {cfg['label']:20s} {C.DIM}({cfg['model'].split('/')[-1]}){C.RESET}  ⏳", flush=True)

        for future in concurrent.futures.as_completed(futures):
            role = futures[future]
            cfg = ANALYSTS[role]
            try:
                content, tokens = future.result()
                analyses[role] = content
                total_tokens += tokens
                # Reprint with done status
                print(f"  {cfg['icon']} {cfg['label']:20s} {C.GREEN}done{C.RESET}  ({tokens} tokens)")
            except Exception as e:
                analyses[role] = f"[Error: {e}]"
                print(f"  {cfg['icon']} {cfg['label']:20s} {C.RED}failed: {e}{C.RESET}")

    # Synthesize
    print(f"\n  {C.GOLD}Synthesizing via {SYNTHESIZER['model'].split('/')[-1]}...{C.RESET}", end="", flush=True)

    synth_prompt = f"""You have 5 expert analyses of an academic paper. Synthesize them into one comprehensive, expert-level digest.

PAPER: "{paper['title']}" by {paper['authors']} ({paper['year']})

"""
    for role, content in analyses.items():
        cfg = ANALYSTS[role]
        synth_prompt += f"### {cfg['icon']} {cfg['label']}'s Analysis:\n{content}\n\n"

    synth_prompt += """Now synthesize ALL 5 perspectives into this EXACT format. Be comprehensive — this should feel like reading the paper itself, condensed:

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

    synth_content, synth_tokens = call_openrouter(SYNTHESIZER["model"],
        "You are the world's best research synthesizer. Combine multiple expert analyses into one authoritative, comprehensive digest. Be thorough — the reader should feel like they've actually read the paper.",
        synth_prompt, max_tokens=SYNTHESIZER["max_tokens"])
    total_tokens += synth_tokens

    est_cost = total_tokens * 0.000015  # rough average across models
    print(f" {C.GREEN}done{C.RESET}")
    print(f"\n  {C.DIM}Total: {total_tokens:,} tokens | ~${est_cost:.3f}{C.RESET}")

    return synth_content

def parse_quiz(text):
    questions = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Q") and ":" in line[:4]:
            q = {"question": line.split(":", 1)[1].strip(), "options": [], "answer": ""}
            i += 1
            while i < len(lines):
                l = lines[i].strip()
                if l.startswith(("A)", "B)", "C)", "D)")):
                    q["options"].append(l)
                elif l.startswith("ANSWER:"):
                    q["answer"] = l.split(":")[1].strip().upper()
                    break
                elif l == "": pass
                elif l.startswith("Q") and ":" in l[:4]: break
                else: pass
                i += 1
            if q["options"] and q["answer"]:
                questions.append(q)
        i += 1
    return questions

def run_quiz_interactive(questions, paper_title, is_review=False):
    """Run quiz. Returns (score, total, wrong_indices)."""
    score = 0
    total = len(questions)
    wrong = []

    label = "REVIEW QUIZ" if is_review else "QUIZ TIME"
    print(f"\n  {C.GOLD}{'─' * 60}{C.RESET}")
    print(f"  {C.BOLD}{label}{C.RESET} — {total} questions on: {paper_title[:40]}")
    if not is_review:
        print(f"  {C.DIM}You need {int(PASS_THRESHOLD*100)}% to pass. Below that = forced re-review.{C.RESET}")
    print(f"  {C.GOLD}{'─' * 60}{C.RESET}\n")

    for idx, q in enumerate(questions):
        print(f"  {C.BOLD}Q{idx+1}/{total}:{C.RESET} {q['question']}")
        for opt in q["options"]:
            print(f"    {opt}")
        while True:
            answer = input(f"\n  {C.GOLD}Your answer (A/B/C/D): {C.RESET}").strip().upper()
            if answer in ["A", "B", "C", "D"]: break
            print(f"  {C.DIM}Enter A, B, C, or D{C.RESET}")

        correct = q["answer"][0]
        if answer == correct:
            score += 1
            print(f"  {C.GREEN}Correct!{C.RESET} ✓")
        else:
            wrong.append(idx)
            print(f"  {C.RED}Wrong.{C.RESET} Answer was {C.BOLD}{correct}{C.RESET}")
            # Show the correct option text
            for opt in q["options"]:
                if opt.startswith(f"{correct})"):
                    print(f"  {C.DIM}→ {opt}{C.RESET}")
        print()

    pct = int((score / total) * 100) if total > 0 else 0
    passed = pct >= int(PASS_THRESHOLD * 100)

    if pct == 100:
        msg = "PERFECT. You own this paper."
    elif passed:
        msg = "Passed. The key ideas stuck."
    else:
        msg = f"FAILED. You need {int(PASS_THRESHOLD*100)}%+. Time to re-review."

    color = C.GREEN if passed else C.RED
    print(f"  {C.GOLD}{'─' * 60}{C.RESET}")
    print(f"  {color}{C.BOLD}Score: {score}/{total} ({pct}%){C.RESET} — {msg}")
    print(f"  {C.GOLD}{'─' * 60}{C.RESET}")

    return score, total, wrong, passed

def display_digest(paper, digest_text, show_quiz_section=False):
    """Display the digest content nicely."""
    parts = digest_text.split("## QUIZ")
    content = parts[0]

    print(f"  {C.BOLD}Paper #{paper['id']}{C.RESET} — {C.GOLD}{paper['tier']}-tier{C.RESET}")
    print(f"  {paper['title']}")
    print(f"  {C.DIM}{paper['authors']} ({paper['year']}){C.RESET}")
    print(f"  {C.DIM}{paper.get('url', '')}{C.RESET}")
    print()

    for line in content.split("\n"):
        s = line.strip()
        if s.startswith("## "):
            print(f"\n  {C.GOLD}{C.BOLD}{s[3:]}{C.RESET}")
        elif s.startswith("- **"):
            # Bold key in bullet
            print(f"  {C.WHITE}  {s}{C.RESET}")
        elif s.startswith("- "):
            print(f"  {C.WHITE}  {s}{C.RESET}")
        elif s.startswith("**") and s.endswith("**"):
            print(f"\n  {C.BOLD}{s.strip('*')}{C.RESET}")
        elif s:
            print(wrap(s))

def run_cumulative_review(data):
    """Every 3 papers, run a cumulative review pulling questions from all completed papers."""
    completed = list(data["progress"].keys())
    if len(completed) < 3 or len(completed) % 3 != 0:
        return

    clear()
    header()
    print(f"  {C.PURPLE}{C.BOLD}╔══════════════════════════════════════════╗{C.RESET}")
    print(f"  {C.PURPLE}{C.BOLD}║   CUMULATIVE REVIEW — {len(completed)} Papers Done    ║{C.RESET}")
    print(f"  {C.PURPLE}{C.BOLD}╚══════════════════════════════════════════╝{C.RESET}")
    print(f"\n  {C.DIM}Testing retention across everything you've learned so far.{C.RESET}\n")

    all_q = []
    for pid in completed:
        df = DIGESTS_DIR / f"{pid}.md"
        if df.exists():
            paper = get_paper_by_id(data, int(pid))
            questions = parse_quiz(df.read_text())
            for q in questions:
                q["paper_id"] = pid
                q["paper_title"] = paper["title"] if paper else f"#{pid}"
            all_q.extend(questions)

    if len(all_q) < 5:
        print(f"  {C.DIM}Not enough questions for cumulative review.{C.RESET}")
        return

    # Pick 10 random questions (or all if less)
    selected = random.sample(all_q, min(10, len(all_q)))
    score = 0

    for idx, q in enumerate(selected):
        print(f"  {C.DIM}From: {q['paper_title'][:45]}{C.RESET}")
        print(f"  {C.BOLD}Q{idx+1}/{len(selected)}:{C.RESET} {q['question']}")
        for opt in q["options"]:
            print(f"    {opt}")
        while True:
            answer = input(f"\n  {C.GOLD}Answer (A/B/C/D): {C.RESET}").strip().upper()
            if answer in ["A", "B", "C", "D"]: break

        if answer == q["answer"][0]:
            score += 1
            print(f"  {C.GREEN}Correct!{C.RESET} ✓")
        else:
            print(f"  {C.RED}Wrong.{C.RESET} Answer: {C.BOLD}{q['answer'][0]}{C.RESET}")
            for opt in q["options"]:
                if opt.startswith(f"{q['answer'][0]})"):
                    print(f"  {C.DIM}→ {opt}{C.RESET}")
        print()

    pct = int((score / len(selected)) * 100)
    print(f"  {C.PURPLE}{'━' * 50}{C.RESET}")
    print(f"  {C.BOLD}Cumulative Score: {score}/{len(selected)} ({pct}%){C.RESET}")

    if pct >= 80:
        print(f"  {C.GREEN}Excellent retention. Your knowledge is building.{C.RESET}")
    elif pct >= 60:
        print(f"  {C.GOLD}Good, but some gaps. Consider re-reviewing weak papers.{C.RESET}")
    else:
        print(f"  {C.RED}Significant gaps. Use --review to revisit specific papers.{C.RESET}")
    print(f"  {C.PURPLE}{'━' * 50}{C.RESET}")

    data["stats"]["quiz_score"] += score
    data["stats"]["quiz_total"] += len(selected)
    save_data(data)
    input(f"\n  {C.GOLD}Press Enter to continue →{C.RESET} ")

def run_paper(data, paper, is_re_review=False):
    """Full flow: fetch → consensus → display → quiz → save (or force re-review)."""
    clear()
    header()

    pid = str(paper["id"])
    digest_file = DIGESTS_DIR / f"{pid}.md"

    if digest_file.exists() and not is_re_review:
        digest_text = digest_file.read_text()
        print(f"  {C.DIM}Loaded cached consensus digest for paper #{paper['id']}{C.RESET}")
    else:
        if digest_file.exists():
            digest_text = digest_file.read_text()
            print(f"  {C.DIM}Loaded cached digest for re-review{C.RESET}")
        else:
            print(f"  {C.BOLD}Paper #{paper['id']}{C.RESET} — {C.GOLD}{paper['tier']}-tier{C.RESET}")
            print(f"  {paper['title']}")
            print(f"  {C.DIM}{paper['authors']} ({paper['year']}){C.RESET}\n")

            # Fetch abstract
            print(f"  {C.DIM}Fetching abstract...{C.RESET}", end="", flush=True)
            abstract = None
            if "arxiv" in paper:
                abstract = fetch_arxiv_abstract(paper["arxiv"])
                print(f" {C.GREEN}got it{C.RESET}" if abstract else f" {C.DIM}using AI knowledge{C.RESET}")
            else:
                print(f" {C.DIM}non-arxiv, using AI knowledge{C.RESET}")

            # Run consensus
            try:
                digest_text = run_consensus_digest(paper, abstract)
                digest_file.write_text(digest_text)
            except Exception as e:
                print(f"\n  {C.RED}Error: {e}{C.RESET}")
                print(f"  {C.DIM}Check OPENROUTER_API_KEY{C.RESET}")
                return

    # Display
    clear()
    header()
    display_digest(paper, digest_text)

    print()
    input(f"  {C.GOLD}Press Enter for quiz ({8} questions, need {int(PASS_THRESHOLD*100)}% to pass) →{C.RESET} ")

    # Quiz
    parts = digest_text.split("## QUIZ")
    quiz_section = parts[1] if len(parts) > 1 else ""
    questions = parse_quiz("## QUIZ\n" + quiz_section)

    if not questions:
        print(f"\n  {C.DIM}No quiz questions parsed. Marking as complete.{C.RESET}")
        quiz_score, quiz_total, wrong, passed = 0, 0, [], True
    else:
        quiz_score, quiz_total, wrong, passed = run_quiz_interactive(questions, paper["title"], is_review=is_re_review)

    if not passed:
        # Force re-review
        print(f"\n  {C.RED}{C.BOLD}RE-REVIEW REQUIRED{C.RESET}")
        print(f"  {C.DIM}You missed {len(wrong)} questions. Let's review what you got wrong.{C.RESET}\n")

        for idx in wrong:
            q = questions[idx]
            print(f"  {C.RED}✗ Q{idx+1}:{C.RESET} {q['question']}")
            correct_letter = q["answer"][0]
            for opt in q["options"]:
                if opt.startswith(f"{correct_letter})"):
                    print(f"  {C.GREEN}  → {opt}{C.RESET}")
            print()

        input(f"  {C.GOLD}Press Enter to re-read the digest →{C.RESET} ")
        clear()
        header()
        display_digest(paper, digest_text)

        print()
        input(f"  {C.GOLD}Press Enter for re-quiz →{C.RESET} ")

        # Re-quiz with just the missed questions + 2 random others
        retry_qs = [questions[i] for i in wrong]
        other_qs = [questions[i] for i in range(len(questions)) if i not in wrong]
        if other_qs:
            retry_qs.extend(random.sample(other_qs, min(2, len(other_qs))))
        random.shuffle(retry_qs)

        quiz_score2, quiz_total2, _, passed2 = run_quiz_interactive(retry_qs, paper["title"], is_review=True)
        quiz_score += quiz_score2
        quiz_total += quiz_total2
        passed = passed2 or (quiz_score2 / quiz_total2 >= PASS_THRESHOLD if quiz_total2 > 0 else False)

        if not passed:
            print(f"\n  {C.DIM}Still struggling. This paper is marked incomplete — come back with --review{C.RESET}")

    # Save progress
    if passed:
        data["progress"][pid] = {
            "completed_at": datetime.now().isoformat(),
            "quiz_score": quiz_score,
            "quiz_total": quiz_total,
            "attempts": data["progress"].get(pid, {}).get("attempts", 0) + 1
        }
        data["stats"]["papers_completed"] = len(data["progress"])

    data["stats"]["quiz_score"] += quiz_score
    data["stats"]["quiz_total"] += quiz_total

    # Streak
    if passed and quiz_total > 0:
        data["stats"]["streak"] = data["stats"].get("streak", 0) + 1
    else:
        data["stats"]["streak"] = 0

    if not data["stats"].get("started_at"):
        data["stats"]["started_at"] = datetime.now().isoformat()

    old_level = data["stats"].get("expertise_level", "")
    new_level = get_level(data["stats"]["papers_completed"])
    data["stats"]["expertise_level"] = new_level
    save_data(data)

    # Level up
    if new_level != old_level and passed:
        print(f"\n  {C.GOLD}{'★' * 30}{C.RESET}")
        print(f"  {C.BOLD}LEVEL UP!{C.RESET} → {C.GOLD}{new_level}{C.RESET}")
        print(f"  {C.GOLD}{'★' * 30}{C.RESET}")

    # Cumulative review every 3 papers
    if passed and data["stats"]["papers_completed"] % 3 == 0 and data["stats"]["papers_completed"] > 0:
        print(f"\n  {C.PURPLE}{C.BOLD}Cumulative review unlocked!{C.RESET}")
        cont = input(f"  {C.GOLD}Take the review now? (y/n): {C.RESET}").strip().lower()
        if cont == "y":
            run_cumulative_review(data)

    # Next paper
    print()
    next_paper = get_next_paper(data)
    if next_paper and passed:
        print(f"  {C.DIM}Next: #{next_paper['id']} — {next_paper['title'][:50]}{C.RESET}")
        print(f"  {C.DIM}Progress: {data['stats']['papers_completed']}/{data['stats']['total_papers']} papers{C.RESET}")
        cont = input(f"\n  {C.GOLD}Continue to next paper? (y/n): {C.RESET}").strip().lower()
        if cont == "y":
            run_paper(data, next_paper)
    elif not next_paper:
        print(f"\n  {C.GREEN}{C.BOLD}ALL PAPERS COMPLETE. You are a {new_level}.{C.RESET}\n")

def get_next_paper(data):
    for p in data["queue"]:
        if str(p["id"]) not in data["progress"]: return p
    return None

def get_paper_by_id(data, pid):
    for p in data["queue"]:
        if p["id"] == pid: return p
    return None

def show_status(data):
    clear()
    header()
    s = data["stats"]
    c = s["papers_completed"]
    t = s["total_papers"]
    level = get_level(c)
    pct = int((c/t)*100) if t>0 else 0
    bar_len = 40
    filled = int(bar_len*c/t) if t>0 else 0
    bar = f"{C.GOLD}{'█'*filled}{C.DIM}{'░'*(bar_len-filled)}{C.RESET}"
    qpct = int((s["quiz_score"]/s["quiz_total"])*100) if s["quiz_total"]>0 else 0

    print(f"  {C.BOLD}Level:{C.RESET} {C.GOLD}{level}{C.RESET}")
    print(f"  {C.BOLD}Papers:{C.RESET} {c}/{t}  [{bar}] {pct}%")
    print(f"  {C.BOLD}Quiz Score:{C.RESET} {s['quiz_score']}/{s['quiz_total']} ({qpct}%)")
    print(f"  {C.BOLD}Streak:{C.RESET} {s.get('streak',0)} {'🔥' if s.get('streak',0)>=3 else ''}")
    print()
    print(f"  {C.BOLD}{'#':>3}  {'Tier':4} {'Status':10} Title{C.RESET}")
    print(f"  {C.DIM}{'─'*68}{C.RESET}")

    for p in data["queue"]:
        pid = str(p["id"])
        done = pid in data["progress"]
        tc = C.GOLD if p["tier"]=="S" else C.BLUE if p["tier"]=="A" else C.DIM
        status = f"{C.GREEN}  PASSED " if done else f"{C.DIM}         "
        print(f"  {p['id']:>3}  {tc}{p['tier']:4}{C.RESET} {status}{C.RESET} {p['title'][:48]}")

    print()
    np = get_next_paper(data)
    if np:
        print(f"  {C.GOLD}Next:{C.RESET} #{np['id']} — {np['title']}")
        print(f"  {C.DIM}Run: python3 blitz.py{C.RESET}")
    else:
        print(f"  {C.GREEN}All papers completed! You are a {level}.{C.RESET}")
    print()

def show_map(data):
    clear()
    header()
    print(f"  {C.BOLD}KNOWLEDGE MAP{C.RESET}\n")
    tags = {}
    for p in data["queue"]:
        for tag in p.get("tags",[]):
            tags.setdefault(tag,[]).append(p)
    for tag, papers in sorted(tags.items(), key=lambda x:-len(x[1]))[:12]:
        dc = sum(1 for p in papers if str(p["id"]) in data["progress"])
        bar = f"{C.GREEN}{'●'*dc}{C.DIM}{'○'*(len(papers)-dc)}{C.RESET}"
        print(f"  {C.GOLD}{tag:30s}{C.RESET} {bar}")
        for p in papers:
            d = str(p["id"]) in data["progress"]
            print(f"    {C.GREEN+'✓' if d else C.DIM+'○'}{C.RESET} #{p['id']:2d} {p['title'][:52]}")
        print()

def random_quiz(data):
    clear()
    header()
    completed = list(data["progress"].keys())
    if len(completed) < 2:
        print(f"  {C.DIM}Complete at least 2 papers first.{C.RESET}")
        return
    all_q = []
    for pid in completed:
        df = DIGESTS_DIR / f"{pid}.md"
        if df.exists():
            paper = get_paper_by_id(data, int(pid))
            qs = parse_quiz(df.read_text())
            for q in qs:
                q["paper_title"] = paper["title"] if paper else f"#{pid}"
            all_q.extend(qs)
    if len(all_q)<3:
        print(f"  {C.DIM}Not enough questions yet.{C.RESET}")
        return
    selected = random.sample(all_q, min(10, len(all_q)))
    print(f"  {C.BOLD}RANDOM REVIEW{C.RESET} — {len(selected)} questions from {len(completed)} papers\n")
    score = 0
    for idx, q in enumerate(selected):
        print(f"  {C.DIM}From: {q['paper_title'][:45]}{C.RESET}")
        print(f"  {C.BOLD}Q{idx+1}:{C.RESET} {q['question']}")
        for opt in q["options"]: print(f"    {opt}")
        while True:
            a = input(f"\n  {C.GOLD}Answer: {C.RESET}").strip().upper()
            if a in ["A","B","C","D"]: break
        if a == q["answer"][0]:
            score += 1; print(f"  {C.GREEN}Correct!{C.RESET} ✓")
        else:
            print(f"  {C.RED}Wrong.{C.RESET} → {C.BOLD}{q['answer'][0]}{C.RESET}")
        print()
    pct = int((score/len(selected))*100)
    print(f"  {C.GOLD}Score: {score}/{len(selected)} ({pct}%){C.RESET}")
    data["stats"]["quiz_score"] += score
    data["stats"]["quiz_total"] += len(selected)
    save_data(data)

def main():
    data = load_data()
    if "--status" in sys.argv: show_status(data); return
    if "--quiz" in sys.argv: random_quiz(data); return
    if "--map" in sys.argv: show_map(data); return
    if "--review" in sys.argv:
        # Re-review a specific paper
        try:
            idx = sys.argv.index("--review")
            pid = int(sys.argv[idx+1])
        except (IndexError, ValueError):
            # Show incomplete papers
            print("Papers available for review:")
            for p in data["queue"]:
                if str(p["id"]) not in data["progress"]:
                    print(f"  #{p['id']} — {p['title']}")
            print("\nUsage: python3 blitz.py --review N")
            return
        paper = get_paper_by_id(data, pid)
        if paper: run_paper(data, paper, is_re_review=True)
        else: print(f"Paper #{pid} not found.")
        return
    if "--pick" in sys.argv:
        try:
            idx = sys.argv.index("--pick")
            pid = int(sys.argv[idx+1])
            paper = get_paper_by_id(data, pid)
            if paper: run_paper(data, paper)
            else: print(f"Paper #{pid} not found.")
        except (IndexError, ValueError):
            print("Usage: python3 blitz.py --pick N")
        return

    paper = get_next_paper(data)
    if paper: run_paper(data, paper)
    else: show_status(data)

if __name__ == "__main__":
    main()
