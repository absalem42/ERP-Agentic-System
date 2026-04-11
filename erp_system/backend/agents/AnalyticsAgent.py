"""
Analytics & Reporting Agent using LangChain
============================================
An agentic AI system for answering quantitative and reasoning-based
executive questions using SQL and contextual explanations.
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import tool

from config.llm import get_groq_model, get_llm, has_llm_credentials

METRICS_PATH = Path(__file__).with_name("metrics.md")


def execute_sql(query: str, params: tuple = ()) -> List[Dict]:
    conn = sqlite3.connect(os.getenv("DB_PATH", ""))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_table_schema(table_name: str) -> str:
    schema = execute_sql(f"PRAGMA table_info({table_name})")
    return f"Table: {table_name}\n" + "\n".join([f"  - {col['name']} ({col['type']})" for col in schema])


def get_all_tables() -> List[str]:
    tables = execute_sql("SELECT name FROM sqlite_master WHERE type='table'")
    return [table["name"] for table in tables]


def _extract_text(response) -> str:
    if hasattr(response, "content"):
        return response.content.strip()
    return str(response).strip()


@tool
def text_to_sql(question: str, context: Optional[str] = None) -> str:
    """Convert a natural language analytics question to a read-only SQL query and execute it."""
    tables = get_all_tables()
    schema_info = "\n".join([get_table_schema(table) for table in tables])
    prompt = f"""
    Given the following database schema:
    {schema_info}

    Convert this question to a SQL query: {question}
    Additional context: {context if context else 'None'}
    Return only one read-only SQL query without explanation.
    SQL Query:
    """

    llm = get_llm()
    sql_query = _extract_text(llm.invoke(prompt)).replace("```sql", "").replace("```", "").strip()
    if not sql_query.upper().startswith("SELECT"):
        return f"Generated SQL was not read-only: {sql_query}"

    try:
        results = execute_sql(sql_query)
        if results:
            df = pd.DataFrame(results)
            return f"Query executed successfully. Results:\n{df.to_string()}\n\nSQL: {sql_query}"
        return f"Query executed but returned no results.\nSQL: {sql_query}"
    except Exception as exc:
        return f"Error executing SQL query: {exc}\nGenerated SQL: {sql_query}"


@tool
def rag_definition(query: str) -> str:
    """Return metric definitions or business context from the local metrics markdown file."""
    if not METRICS_PATH.exists():
        return "Metrics reference file is unavailable."

    content = METRICS_PATH.read_text(encoding="utf-8")
    paragraphs = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    query_terms = [term for term in query.lower().split() if len(term) > 2]

    matches = []
    for paragraph in paragraphs:
        lowered = paragraph.lower()
        if any(term in lowered for term in query_terms):
            matches.append(paragraph)

    if matches:
        return "\n\n".join(matches[:3])
    return "No relevant metric definition found in the local knowledge base."


@tool
def analytics_reporting(input_data):
    """Summarize, aggregate, or create a visualization spec from structured data."""
    try:
        if isinstance(input_data, str):
            cleaned = input_data.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            input_data = json.loads(cleaned.strip())

        data = input_data.get("data", [])
        if not data:
            return "No data provided"

        df = pd.DataFrame(data)
        if df.empty:
            return "Empty dataset"

        operation = input_data.get("operation", "summarize")
        params = input_data.get("params", {})

        if operation == "visualize":
            viz_type = params.get("type", "bar")
            x_col = params.get("x", df.columns[0])
            y_col = params.get("y", df.columns[1] if len(df.columns) > 1 else df.columns[0])
            viz_spec = {
                "type": viz_type,
                "data": df.to_dict("records"),
                "x": x_col,
                "y": y_col,
                "title": params.get("title", f"{viz_type} chart"),
            }
            if viz_type == "pie":
                viz_spec["label"] = x_col
                viz_spec["value"] = y_col
            elif viz_type == "histogram":
                viz_spec["column"] = x_col
            elif viz_type == "box":
                viz_spec["column"] = y_col
                viz_spec["category"] = x_col
            return json.dumps(viz_spec, indent=2)

        if operation == "aggregate":
            group_col = params.get("group_by")
            value_col = params.get("value_col")
            agg_func = params.get("agg_func", "sum")
            if group_col and value_col:
                return df.groupby(group_col)[value_col].agg(agg_func).to_string()
            return df.describe().to_string()

        summary = {"rows": len(df), "columns": list(df.columns), "sample": df.head(3).to_dict("records")}
        return json.dumps(summary, indent=2)
    except Exception as exc:
        return f"Error: {exc}"


ANALYTICS_AGENT_SYSTEM = """You are the Analytics & Reporting Agent for Helios Dynamics.

Your responsibilities:
- Retrieve business definitions and metrics using the rag_definition tool when needed for business context.
- Answer executive questions using SQL and contextual explanations.
- Use the text_to_sql tool for natural language to SQL conversion. Only use read-only queries.
- Use analytics_reporting to summarize data or create visualization specs.

After you finish, return:
- data retrieved in table format, if any
- json output of any visualizations you created, if any
- your insights and analysis of the data

Available tools: {tools}
Tool names: {tool_names}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def create_analytics_agent():
    llm = get_llm()
    tools = [text_to_sql, rag_definition, analytics_reporting]
    memory = ConversationBufferMemory()
    prompt = PromptTemplate.from_template(ANALYTICS_AGENT_SYSTEM)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        memory=memory,
    )

if __name__ == "__main__":
    executor = create_analytics_agent()
    print("Analytics Agent Ready")
    print(f"Provider: {'Groq' if has_llm_credentials() else 'Fallback'}")
    if has_llm_credentials():
        print(f"Model: {get_groq_model()}")
    try:
        while True:
            user_input = input("Analytics Agent > ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            try:
                result = executor.invoke({"input": user_input})
                print(f"\n{result['output']}\n")
            except Exception as exc:
                print(f"Error: {exc}")
    except KeyboardInterrupt:
        print("\nGoodbye!")
