# agentcore/base_agent.py
class BaseAgent:
    def __init__(self, name, role, memory):
        self.name = name
        self.role = role
        self.memory = memory

    def act(self, goal: str, context: dict = {}) -> str:
        raise NotImplementedError("Each agent must define its own 'act' method.")