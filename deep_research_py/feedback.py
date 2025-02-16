from typing import List, Optional
import openai
import ollama
import json
from .prompt import system_prompt
from .common.logging import log_error, log_event
from .ai.providers import generate_completions
from deep_research_py.utils import ModelConfig
from pydantic import BaseModel


class FeedbackResponse(BaseModel):
    questions: List[str]


async def generate_feedback(
    query: str,
    client: Optional[openai.OpenAI | ollama.Client],
    model: str,
    max_feedbacks: int = 5,
) -> List[str]:
    """Generates follow-up questions to clarify research direction."""

    prompt = f"Given this research topic: {query}, generate at most {max_feedbacks} follow-up questions to better understand the user's research needs, but feel free to return none questions if the original query is clear. Return the response as a JSON object with a 'questions' array field."

    response = await generate_completions(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": prompt,
            },
        ],
        format=FeedbackResponse.model_json_schema(),
    )

    # Parse the JSON response
    try:
        if ModelConfig.get_service() == "ollama":
            result = json.loads(response.message.content)
        else:
            # OpenAI compatible API
            result = json.loads(response.choices[0].message.content)
        log_event(
            f"Generated {len(result.get('questions', []))} follow-up questions for query: {query}"
        )
        log_event(f"Got follow-up questions: {result.get('questions', [])}")
        return result.get("questions", [])
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        log_error(f"Failed to parse JSON response for query: {query}")
        return []
