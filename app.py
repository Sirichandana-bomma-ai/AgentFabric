# app.py

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from agentcore.memory import SharedMemory
from agentcore.planner import PlannerAgent
from agentcore.tool_user import ToolUserAgent
from agentcore.summary_agent import SummaryAgent

st.set_page_config(page_title="AgentFabric", layout="centered")
st.title("🤖 AgentFabric: Agentic AI System")

# Initialize memory + agents
memory = SharedMemory()
planner = PlannerAgent(name="PlannerAgent", memory=memory)
tool_user = ToolUserAgent(name="ToolUserAgent", memory=memory)
summary_agent = SummaryAgent(name="SummaryAgent", memory=memory)

# UI: User enters a goal
goal = st.text_input("Enter your goal", placeholder="e.g., Find top remote AI jobs")

if st.button("Run Agent System") and goal:
    st.subheader("🧠 Planner Thinking...")
    plan = planner.act(goal)
    next_agent = plan.get("agent", "ToolUser")
    subtask = plan.get("subtask", goal)
    st.success(f"Planner says: Use **{next_agent}** for subtask: *{subtask}*")

    st.markdown("### 🤖 Agent Output:")
    
    # ToolUser Agent
    if next_agent == "ToolUser":
        result = tool_user.act(subtask)
        st.code(result)

        # Extract and summarize URLs if any
        st.markdown("### 📝 Auto-Summary of Top URLs:")
        for line in result.splitlines():
            url = line.strip()
            if url.startswith("http"):
                st.markdown(f"🔗 [Source Link]({url})")
                try:
                    summary = summary_agent.act(url)
                    st.success(summary)
                except Exception as e:
                    st.error(f"Summary failed: {e}")

    # SummaryAgent direct usage
    elif next_agent == "SummaryAgent":
        result = summary_agent.act(subtask)
        if "summary" in result:
            st.markdown("### 📝 Summarized Content:")
            st.success(result)
        else:
            st.warning("No summary returned.")

    # Fallback
    else:
        st.warning(f"No agent implemented yet for: {next_agent}")

    # Memory log
    st.markdown("### 🧠 Recent Memory:")
    for item in memory.get_history():
        st.markdown(f"- **{item['agent']}** did: `{item['action']}`")