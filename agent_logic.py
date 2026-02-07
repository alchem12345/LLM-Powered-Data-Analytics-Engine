import pandas as pd
import yaml
from langchain_classic.agents import AgentExecutor,create_tool_calling_agent
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import openai
import os


load_dotenv(dotenv_path='example.env')
openai_api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai_api_key)


with open("config.yaml" , 'r') as f:
    config = yaml.safe_load(f)

# creating the dataset
df = pd.read_csv(config['dataset']['path'],sep = config['dataset']['sep'])

# some manual tools to be used for quick computation and fast retreival of data from the dataframe
@tool
def get_column_correlation(col1 , col2):
    """Calculates the pearson correlation between the two columns"""
    if col1 not in df.columns or col2 not in df.columns:
        return f"the columns must be from the list of the following columns{df.columns}"
    else:
        correl = df[col1].corr(df[col2])

    return correl
@tool
def get_data_summary():
    """run this function for knowing what the numbers in each column of the dataframe stand for"""
    return config['metadata']
#..............................................................................
def generate_pandas_logic(user_request):
    """Direct OpenAI call to convert natural language to Pandas code"""
    metadata_str = "\n".join([f"- {col}: {desc}" for col, desc in config['metadata'].items()])
    system_prompt = f"""
    You are a Senior Data Analyst. 
    DATASET SCHEMA:
    {metadata_str}
    
    TASK: Write a 1-line Python expression using a dataframe named 'df' to answer: "{user_request}"
    
    RULES:
    1. For 'best', 'smartest', or 'top', use: df.sort_values(by=['grade', 'grade_previous'], ascending=False).head(5)
    2. For filtering, use standard Pandas syntax: df[df['age'] == 1]
    3. Return ONLY the code. No markdown, no 'python' tag, no explanations.
    """

    # now making the actual llm and its call for the best low latency llm call for the query
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip().replace("`", "").replace("python", "")


@tool
def analytical_tool(user_request):
    """use this tool for analysing the data using pandas and its query system where other tools may not be used the input to this tool shoud be the best description of the user query you think is best for creating a pandas query"""

    try :
        code = generate_pandas_logic(user_request)
        result = eval(code, {"df" : df , "pd" : pd})

        if isinstance(result, pd.DataFrame):
            if result.empty:
                return "No students found for that specific criteria."
            return result.to_string()

        return f"The result is: {str(result)}"

    except Exception as e:
        return f"Analysis Error: I couldn't process that logic. Try asking for specific columns like 'grade' or 'age'."
#..............................................................................


# now that we have made the tools for the AGent lets make the LLM module or the Brains of the Agent itself
metadata_context = "\n".join([f"- {k}: {v}" for k, v in config['metadata'].items()])

prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are VeriScholar AI, a data analyst. 
    DATA SCHEMA:
    {metadata_context}

    INSTRUCTIONS:
    - Use the schema above to understand column meanings (e.g., G3 is the final grade).
    - If a user asks for 'at-risk' students, look for low G3 and high absences.
and Any questions unrelated to the Data analysis or any random out of context should be ignored and replied back to in an appropriate manner"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

def get_agent():
    llm = ChatOpenAI(model = config['llm']['model'] , temperature = config['llm']['temperature'] , api_key = openai_api_key)
    tools = [get_column_correlation, analytical_tool ,get_data_summary]
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True , max_iteration = 5)







