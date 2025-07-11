from agentcore.base_agent import BaseAgent
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
import torch

class PlannerAgent(BaseAgent):
    def __init__(self, name, memory):
        super().__init__(name=name, role="Planner", memory=memory)

        model_id = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32  # ⬅️ Add this to avoid meta tensor issue
        )

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            device=-1  # ⬅️ Force CPU (MPS can cause meta tensor issues)
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

    def act(self, goal: str, context: dict = {}) -> dict:
        prompt = f"""
You are a planner agent in a multi-agent system.

User's Goal:
{goal}

Decide the next agent to call. Choose one of: ['ToolUser', 'SearchAgent', 'SummaryAgent']

You can also specify a short subtask for the agent.

Respond in JSON like:
{{"agent": "ToolUser", "subtask": "Search for top AI jobs"}}
"""
        response = self.llm.invoke(prompt)
        try:
            result = eval(response[0]['generated_text'])  # Safe eval for controlled output
            return result
        except:
            return {"agent": "ToolUser", "subtask": goal}