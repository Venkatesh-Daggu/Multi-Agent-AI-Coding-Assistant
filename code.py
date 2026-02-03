# ======================================================
# ALL-IN-ONE STREAMLIT MULTI-AGENT PIPELINE
# ======================================================
import json
import os
import re
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM

# ======================================================
# API KEYS
# ======================================================
GEMINI_API_KEY = "key1"
GEMINI_API_KEY2 = "Key2"

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
os.environ["GOOGLE_API_KEY2"] = GEMINI_API_KEY2

# ======================================================
# LLMs
# ======================================================

llm = LLM(model="gemini/gemini-2.5-flash", api_key=GEMINI_API_KEY)
llm2 = LLM(model="gemini/gemini-2.5-flash-lite", api_key=GEMINI_API_KEY2)

# ======================================================
# UTILS
# ======================================================
def extract_json(text: str):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON found in output")
    return json.loads(match.group())

# ======================================================
# AGENTS
# ======================================================
developer_agent = Agent(
    role="Software Developer",
    goal="Write correct, clean, production-ready code, no comments.",
    backstory="Senior developer. Output ONLY executable code.",
    llm=llm,
    tools=[],
    verbose=True
)

qa_agent = Agent(
    role="QA Engineer",
    goal="Find bugs and runtime issues",
    backstory="Aggressive QA engineer",
    llm=llm,
    tools=[],
    verbose=True
)

use_case_agent = Agent(
    role="Use Case Analysis Agent",
    goal="Generate exhaustive test cases",
    backstory="QA architect focused on edge cases",
    llm=llm2,
    verbose=True
)

testing_agent = Agent(
    role="Automated Testing Agent",
    goal="Execute test cases and validate outputs",
    backstory="Automated test runner",
    llm=llm,
    tools=[],
    verbose=True
)

reviewer_agent = Agent(
    role="Code Reviewer",
    goal="Approve only production-ready code",
    backstory="Principal engineer",
    llm=llm2,
    verbose=True
)

# ======================================================
# PIPELINE FUNCTION
# ======================================================
def run_full_pipeline(user_request, max_iterations):
    code_output = ""
    final_code = ""
    test_results_summary = []
  
    for iteration in range(max_iterations):
        dev_task = Task(
            description=f"Write code for the requirement below.\nReturn ONLY executable code.\n\nRequirement:\n{user_request}",
            agent=developer_agent,
            expected_output="Executable Python source code only"
        )

        qa_task = Task(
            description="Analyze the code for bugs, logical errors, and runtime issues.\nFix if needed.\nReturn ONLY corrected executable code.",
            agent=qa_agent,
            expected_output="Bug-free executable Python code"
        )

        use_case_task = Task(
            description="""
Generate exhaustive test cases.
Return ONLY JSON in the format:
{
  "tests": [
    {"description": "", "call": "", "expected_output": ""}
  ]
}
""",
            agent=use_case_agent,
            expected_output="Valid JSON object containing test cases"
        )

        testing_task = Task(
            description="""
Execute the test cases against the code.
Return ONLY JSON with actual_output and status.
""",
            agent=testing_agent,
            expected_output="""
{
  "tests": [
    {"description": "", "call": "", "expected_output": "", "actual_output": "", "status": "pass/fail"}
  ]
}
"""
        )
        review_task = Task(
            description="Approve ONLY if code is correct and production-ready.\nReturn APPROVED or REJECTED with reason.",
            agent=reviewer_agent,
            expected_output="APPROVED or REJECTED"
        )

        crew = Crew(
            agents=[developer_agent, qa_agent, use_case_agent, testing_agent, reviewer_agent],
            tasks=[dev_task, qa_task, use_case_task, testing_task, review_task],
            process=Process.sequential
        )

        result = crew.kickoff()
        code_output = str(result)
        final_code = dev_task.output.raw

        # Parse test results
        try:
            testing_json = extract_json(testing_task.output.raw)
            for idx, test in enumerate(testing_json.get("tests", []), start=1):
                status = test.get("status", "fail").upper()
                test_results_summary.append(f"Test Case {idx}: {status}")
        except Exception as e:
            test_results_summary.append(f"Test results parsing failed: {e}")

        if "APPROVED" in code_output.upper():
            break

    return {
        "final_code": final_code,
        "pipeline_output": code_output,
        "test_summary": test_results_summary
    }

# ======================================================
# STREAMLIT FRONTEND
# ======================================================
st.set_page_config(page_title="AI Coding Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AI Multi-Agent Coding Assistant")
st.markdown("""
Enter your programming requirement below.   
""")

user_task = st.text_area("Enter your programming task:", placeholder="E.g., Write a Python function to check if a string is a palindrome")
max_iterations = st.number_input("Max Iterations (for refining code if needed):", min_value=1, max_value=5, value=1, step=1)

if st.button("Run Pipeline"):
    if not user_task.strip():
        st.warning("Please enter a programming task!")
    else:
        with st.spinner("Running AI pipeline..."):
            output = run_full_pipeline(user_task, max_iterations=max_iterations)

        st.subheader("✅ Final Code")
        st.code(output["final_code"], language="python")

        st.subheader("🧪 Test Case Results")
        for line in output["test_summary"]:
            if "PASS" in line:
                st.success(line)
            elif "FAIL" in line:
                st.error(line)
            else:
                st.warning(line)

        st.subheader("📄 Full Pipeline Output")
        st.text_area("Pipeline Output", output["pipeline_output"], height=300) 