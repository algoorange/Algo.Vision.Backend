from app.utils.helpers import build_summary_prompt
from app.services.query_engine import query_ollama

async def generate_summary(tracks):
    summary_prompt = build_summary_prompt(tracks)
    summary = await query_ollama(summary_prompt)
    return summary
