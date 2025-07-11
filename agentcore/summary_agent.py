# agentcore/summary_agent.py

from agentcore.base_agent import BaseAgent
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from newspaper import Article
import torch

class SummaryAgent(BaseAgent):
    def __init__(self, name, memory):
        super().__init__(name=name, role="SummaryAgent", memory=memory)
        model_id = "google/flan-t5-base"

        # Force CPU usage for compatibility
        device = 0 if torch.cuda.is_available() else -1

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        self.pipe = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=256
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    def scrape_article(self, url):
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()

    def act(self, url: str, context: dict = {}) -> str:
        try:
            raw_text = self.scrape_article(url)
            if not raw_text or len(raw_text.split()) < 50:
                return f"[SummaryAgent] Article too short or empty to summarize: {url}"

            summary = self.llm.invoke(f"summarize: {raw_text}")
            return summary[0]['generated_text'] if isinstance(summary, list) else summary
        except Exception as e:
            return f"[Summary Error] Could not summarize {url}: {str(e)}"