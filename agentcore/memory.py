# agentcore/memory.py

class SharedMemory:
    def __init__(self):
        self.logs = []

    def add(self, agent_name, action):
        self.logs.append({"agent": agent_name, "action": action})

    def get_history(self):
        return self.logs[-5:]  # return last 5 actions