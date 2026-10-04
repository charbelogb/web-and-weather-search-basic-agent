
import os
import certifi

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.tools import tool
import requests


from langchain.agents import create_react_agent, AgentExecutor


#===================
# LOAD ENV VARIABLES
#===================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


search_tools = TavilySearchResults(max_results=2)


@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city
    """
    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    ) 

    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}."

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%\n"   
    )



result = search_tools.invoke("Donne moi les dernières actualités sur Kemi Seba")
result


#============
# LLM
#============
llm = ChatOpenAI(
    model="gpt-5.4-mini",
    api_key=OPENAI_API_KEY,
    temperature=1,
)


response = llm.invoke("Quelle date sommes-nous aujourd'hui ?")
response


#============
# PROMPT
#============
prompt = hub.pull("hwchase17/react") #React predefined prompt



#============
# TOOLS
#============
tools = [search_tools, get_weather_data]


#============
# CREATE AGENT
#============
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=False
)



#============
# EXECUTOR
#============
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)


#============
# RUN
#============
response = agent_executor.invoke({
    "input": (
        "Trouve la capitale du Bénin"
        "Et donne la météo actuelle."
    )
})
print(response["output"])





