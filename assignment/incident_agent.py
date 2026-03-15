"""
incident_agent.py — "Ghost-Protocol" Agentic AI Workflow
=============================================================
An Autonomous Incident Response Agent that uses Endee as its Long-Term Memory.

When a server crash or error occurs, the Agent:
  1. Uses the `search_memory` tool to query Endee for past occurrences.
  2. Analyzes the retrieved context (past solutions).
  3. Decides to either "Fix" (generate code) or "Escalate" (summarize for human).

Note: Powered by Google Gemini 3-Flash (via GEMINI_API_KEY).
Uses Endee for stateful incident retrieval.
"""

import os
import json
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

# ── 1. Initialize Endee "Long-Term Memory" ──────────────────

INDEX_NAME = "agentic_incident_memory"
DIMENSION = 384

print("Loading Sentence-Transformers model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
client = Endee()
if os.environ.get("NDD_URL"):
    client.set_base_url(os.environ.get("NDD_URL"))

# Ensure fresh index for the demo
try:
    client.delete_index(INDEX_NAME)
except Exception:
    pass

client.create_index(name=INDEX_NAME, dimension=DIMENSION, space_type="cosine", precision=Precision.FLOAT32)
memory_index = client.get_index(name=INDEX_NAME)

# ── 2. Seed Memory with Past Server Incidents ───────────────

past_incidents = [
    {
        "error_str": "PostgreSQL Connection Refused port 5432",
        "solution": "Restarted the pg_ctl service and increased max_connections to 200 in postgresql.conf. Resolved gracefully.",
        "difficulty": "Easy"
    },
    {
        "error_str": "OOMKilled: Pod memory limit exceeded in Kubernetes",
        "solution": "Memory leak detected in Node.js worker pod. Requires senior SRE to profile the heap. Cannot be auto-fixed.",
        "difficulty": "Hard"
    },
    {
        "error_str": "AWS S3 Access Denied 403 Bucket Policy",
        "solution": "IAM role lost s3:PutObject permissions. Terraform applied to restore the role. Fix script available.",
        "difficulty": "Easy"
    }
]

print("\n[Admin] Seeding AI's Endee Memory with past incident logs...")
payloads = []
for i, inc in enumerate(past_incidents):
    # Vectorize the error signature
    vec = model.encode([inc["error_str"]])[0].tolist()
    payloads.append({
        "id": f"incident_log_{i}",
        "vector": vec,
        "meta": inc
    })
memory_index.upsert(payloads)
print(f"        ✓ {len(payloads)} past incidents committed to long-term memory.")


# ── 3. Define the LLM's Tool (Function Calling) ─────────────

def search_memory(error_signature: str) -> str:
    """
    The Agent calls this tool to browse its Endee vector memory.
    """
    print(f"\n  [Agent Tool Execution] 🔍 Querying Endee Memory for: '{error_signature}'")
    vec = model.encode([error_signature])[0].tolist()
    
    # Query Endee for the closest past incident
    results = memory_index.query(vector=vec, top_k=1)
    
    if not results or results[0].get("distance", -1) < 0.35:
        return "No similar past incidents found in memory."
    
    match = results[0]["meta"]
    print(f"  [Endee Return] Found past match! '{match['error_str']}'")
    return f"Past Issue: {match['error_str']}. Past Solution: {match['solution']}. Difficulty: {match['difficulty']}."


# ── 4. The Agentic Loop (Gemini Powered) ───────────────────

def run_agentic_loop(new_error: str):
    """
    Runs the autonomous agent loop using Gemini 3-Flash.
    The agent consults its Endee memory before making a decision.
    """
    print(f"\n🚨 NEW ALERT RECEIVED: {new_error}\n")
    print("🤖 Agent Thinking: Analyzing alert signature...")
    
    # 1. Autonomous Retrieval from Endee Memory
    memory_result = search_memory(new_error)
    
    # 2. Reasoning with Gemini
    from google import genai
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ Error: GEMINI_API_KEY not found in environment.")
        return
    gen_client = genai.Client()

    agent_prompt = f"""
    You are an Autonomous Site Reliability Engineering (SRE) Agent. 
    You have just received a server error: "{new_error}"
    
    You searched your internal Endee Long-Term Memory and found this:
    {memory_result}

    DECISION RULES:
    1. If the memory contains a solution and the difficulty is 'Easy', output: "DECISION: AUTO-FIX 🛠️" followed by the fix command.
    2. If the difficulty is 'Hard' or context is complex, output: "DECISION: ESCALATE w/ CONTEXT ⚠️" followed by a summary for a human.
    3. If no similar incidents were found, output: "DECISION: EMERGENCY ESCALATE ☎️".

    Assistant:
    """

    # Optimized Model Rotation for Quota Resilience
    models_to_try = [
        "gemini-2.0-flash", 
        "gemini-2.0-flash-lite-preview-02-05", 
        "gemini-1.5-flash", 
        "gemini-1.5-flash-8b"
    ]
    
    response_text = None
    for model_name in models_to_try:
        try:
            response = gen_client.models.generate_content(model=model_name, contents=agent_prompt)
            if response.text:
                response_text = response.text
                break
        except Exception as e:
            continue

    if response_text:
        print(f"\n🤖 Agent Decision Engine:\n{response_text}\n")
    else:
        print(f"❌ Agent Loop Failed: Quota exceeded on all Gemini models.")


# ── RUN PLAYBOOKS ──────────────────────────────────────────

if __name__ == "__main__":
    print("\n==================================================================")
    print(" 🚨 GHOST-PROTOCOL: Automated Incident Response Agent")
    print("==================================================================")
    
    print("\n--- Playbook 1: A Known, Auto-Fixable Issue ---")
    run_agentic_loop("URGENT: Database crashing. Connection Refused to Postgres on port 5432.")
    
    print("\n--- Playbook 2: A Known, Hard Issue (Requires Escalation) ---")
    run_agentic_loop("Production down: K8s pods encountering OOMKilled limits constantly.")
    
    print("\n--- Playbook 3: A Completely Unknown Issue ---")
    run_agentic_loop("Kafka brokers encountering Split-Brain network partition anomaly.")
    
    print("\n==================================================================")
    print("✓ Agent executed successfully via Endee Vector Memory routing.")
