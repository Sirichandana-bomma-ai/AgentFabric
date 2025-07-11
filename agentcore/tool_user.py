from agentcore.base_agent import BaseAgent
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from newspaper import Article
from transformers import pipeline

# Load summarization pipeline once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class ToolUserAgent(BaseAgent):
    def __init__(self, name, memory):
        super().__init__(name=name, role="ToolUser", memory=memory)
        self.search = DuckDuckGoSearchAPIWrapper()

    def summarize_article(self, url):
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
            summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            return f"⚠️ Error summarizing article: {e}"

    def act(self, task: str, context: dict = {}) -> str:
        try:
            results = self.search.results(task, max_results=3)
            top_url = results[0]['link']
            summary = self.summarize_article(top_url)
            return f"📝 Summary of top result for '{task}':\n\n{summary}\n\n🔗 {top_url}"
        except Exception as e:
            return f"⚠️ Error performing search or summarization: {e}"