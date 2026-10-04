import os
import json
import certifi
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Agentic AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ENVIRONMENT
# ============================================================

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


missing_keys = []

if not OPENAI_API_KEY:
    missing_keys.append("OPENAI_API_KEY")

if not TAVILY_API_KEY:
    missing_keys.append("TAVILY_API_KEY")

if not WEATHERSTACK_API_KEY:
    missing_keys.append("WEATHERSTACK_API_KEY")

if missing_keys:
    st.error(
        "Clé(s) API manquante(s) : "
        + ", ".join(missing_keys)
    )
    st.stop()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #F8FAFC;
    }

    .block-container {
        max-width: 1050px;
        padding-top: 3rem;
        padding-bottom: 6rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    h1 {
        color: #0F172A !important;
        letter-spacing: -1.6px !important;
        font-weight: 750 !important;
    }

    h2, h3 {
        color: #0F172A !important;
    }

    p {
        color: #475569;
    }

    /* SIDEBAR */

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    [data-testid="stSidebar"] h1 {
        font-size: 1.35rem !important;
        letter-spacing: -0.5px !important;
    }

    /* CONTAINERS */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border-color: #E2E8F0 !important;
        border-radius: 14px !important;
    }

    /* BUTTONS */

    .stButton > button {
        border-radius: 13px;
        border: 1px solid #E2E8F0;
        background: #FFFFFF;
        color: #334155;
        min-height: 62px;
        font-weight: 550;

        transition:
            border-color .15s ease,
            box-shadow .15s ease,
            transform .15s ease;
    }

    .stButton > button:hover {
        border-color: #818CF8;
        color: #4338CA;
        background: #FAFAFF;

        box-shadow:
            0 6px 20px rgba(79, 70, 229, .08);

        transform: translateY(-1px);
    }

    /* INPUT */

    [data-testid="stTextInput"] input {
        min-height: 55px;
        border-radius: 13px;
        border: 1px solid #CBD5E1;
        background: #FFFFFF;
        font-size: 15px;
    }

    [data-testid="stTextInput"] input:focus {
        border-color: #818CF8;
        box-shadow:
            0 0 0 3px rgba(99, 102, 241, .10);
    }

    /* CHAT */

    [data-testid="stChatMessage"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 14px 18px;
        margin-bottom: 12px;

        box-shadow:
            0 1px 3px rgba(15, 23, 42, .03);
    }

    [data-testid="stChatInput"] {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 15px;

        box-shadow:
            0 10px 35px rgba(15, 23, 42, .07);
    }

    /* STATUS */

    [data-testid="stStatusWidget"] {
        border-radius: 12px;
    }

    @media (max-width: 768px) {

        .block-container {
            padding-top: 2rem;
        }

        h1 {
            font-size: 2rem !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOOLS
# ============================================================

search_tool = TavilySearchResults(
    max_results=3
)


@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.

    Use this tool whenever the user asks about current
    weather conditions in a location.
    """

    url = (
        "https://api.weatherstack.com/current"
        f"?access_key={WEATHERSTACK_API_KEY}"
        f"&query={city}"
    )

    try:

        response = requests.get(
            url,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        if "current" not in data:

            error = data.get(
                "error",
                {}
            )

            error_message = error.get(
                "info",
                "Unknown Weatherstack error"
            )

            return (
                f"Could not fetch weather data "
                f"for {city}. "
                f"Reason: {error_message}"
            )

        current = data["current"]

        descriptions = current.get(
            "weather_descriptions",
            []
        )

        description = (
            descriptions[0]
            if descriptions
            else "Unknown"
        )

        return (
            f"City: {city}\n"
            f"Temperature: {current.get('temperature')}°C\n"
            f"Weather: {description}\n"
            f"Humidity: {current.get('humidity')}%"
        )

    except requests.RequestException as error:

        return (
            "Weather service error: "
            f"{str(error)}"
        )


tools = [
    search_tool,
    get_weather_data,
]


# Map permettant d'exécuter un tool à partir de son nom
tools_by_name = {
    tool_item.name: tool_item
    for tool_item in tools
}


# ============================================================
# LLM + NATIVE TOOL CALLING
# ============================================================

@st.cache_resource
def create_llm():

    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        api_key=OPENAI_API_KEY,
        temperature=1,
    )

    # C'est ici que les tools sont exposés nativement au modèle.
    return llm.bind_tools(tools)


llm_with_tools = create_llm()


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an intelligent AI research assistant.

You have access to external tools.

Rules:

1. Use Tavily Search whenever the user asks for information
   that may be recent, current, time-sensitive, political,
   news-related, or likely to have changed.

2. Use get_weather_data whenever the user asks for current
   weather information.

3. Never pretend that your internal knowledge is current
   when the question requires up-to-date information.

4. When a tool can provide a more reliable answer than your
   internal knowledge, use the tool.

5. You may use several tools for the same request.

6. Answer in the same language as the user unless asked
   otherwise.

7. Do not mention internal tool mechanics unless useful.

8. Base current factual claims on the information returned
   by the tools.
"""


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ============================================================
# RESET
# ============================================================

def reset_conversation():

    st.session_state.messages = []
    st.session_state.pending_prompt = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("✦ Agentic AI")

    st.caption(
        "Intelligent research assistant"
    )

    st.success(
        "● Agent opérationnel"
    )

    st.divider()

    st.caption(
        "CAPACITÉS"
    )

    with st.container(border=True):

        st.markdown(
            "#### 🌐 Recherche Web"
        )

        st.caption(
            "Recherche d'informations et "
            "d'actualités récentes sur Internet."
        )

    with st.container(border=True):

        st.markdown(
            "#### 🌤️ Météo en temps réel"
        )

        st.caption(
            "Consultation des conditions "
            "météorologiques actuelles."
        )

    st.divider()

    st.caption(
        "CONFIGURATION"
    )

    with st.container(border=True):

        st.markdown(
            "**GPT-5.4 mini**"
        )

        st.caption(
            "Native Tool Calling · 2 outils"
        )

    st.write("")

    st.button(
        "＋ Nouvelle conversation",
        use_container_width=True,
        on_click=reset_conversation,
    )


# ============================================================
# CONVERT CHAT HISTORY
# ============================================================

def build_llm_messages():

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        )
    ]

    for message in st.session_state.messages:

        if message["role"] == "user":

            messages.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        elif message["role"] == "assistant":

            messages.append(
                AIMessage(
                    content=message["content"]
                )
            )

    return messages


# ============================================================
# FORMAT TOOL INPUT
# ============================================================

def format_tool_input(args):

    if isinstance(args, dict):

        return ", ".join(
            f"{key}={value}"
            for key, value in args.items()
        )

    return str(args)


# ============================================================
# RUN AGENT
# ============================================================

def run_agent(user_prompt: str):

    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    # --------------------------------------------------------
    # Display user
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            user_prompt
        )

    # --------------------------------------------------------
    # Assistant
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        tool_count = 0
        tools_used = []

        with st.status(
            "🧠 Analyse de votre demande...",
            expanded=True,
        ) as status:

            try:

                # ============================================
                # Build conversation
                # ============================================

                messages = build_llm_messages()

                # ============================================
                # Agent loop
                # ============================================

                max_iterations = 6

                final_answer = None

                for iteration in range(
                    max_iterations
                ):

                    # ----------------------------------------
                    # Ask LLM
                    # ----------------------------------------

                    ai_message = (
                        llm_with_tools.invoke(
                            messages
                        )
                    )

                    messages.append(
                        ai_message
                    )

                    # ----------------------------------------
                    # No tool calls => final answer
                    # ----------------------------------------

                    if not ai_message.tool_calls:

                        final_answer = (
                            ai_message.content
                        )

                        break

                    # ----------------------------------------
                    # Execute requested tools
                    # ----------------------------------------

                    for tool_call in (
                        ai_message.tool_calls
                    ):

                        tool_name = (
                            tool_call["name"]
                        )

                        tool_args = (
                            tool_call["args"]
                        )

                        tool_call_id = (
                            tool_call["id"]
                        )

                        tool_count += 1

                        # ====================================
                        # UI
                        # ====================================

                        if (
                            "tavily"
                            in tool_name.lower()
                        ):

                            status.write(
                                "🌐 Recherche sur le web"
                            )

                            tools_used.append(
                                "Tavily"
                            )

                        elif (
                            "weather"
                            in tool_name.lower()
                        ):

                            status.write(
                                "🌤️ Consultation "
                                "de la météo"
                            )

                            tools_used.append(
                                "Weatherstack"
                            )

                        else:

                            status.write(
                                f"⚡ Utilisation de "
                                f"{tool_name}"
                            )

                            tools_used.append(
                                tool_name
                            )

                        formatted_input = (
                            format_tool_input(
                                tool_args
                            )
                        )

                        if formatted_input:

                            status.caption(
                                formatted_input
                            )

                        # ====================================
                        # Find tool
                        # ====================================

                        selected_tool = (
                            tools_by_name.get(
                                tool_name
                            )
                        )

                        if (
                            selected_tool
                            is None
                        ):

                            tool_result = (
                                f"Unknown tool: "
                                f"{tool_name}"
                            )

                        else:

                            try:

                                # ============================
                                # Execute tool
                                # ============================

                                tool_result = (
                                    selected_tool.invoke(
                                        tool_args
                                    )
                                )

                                status.write(
                                    "✓ Données récupérées"
                                )

                            except Exception as error:

                                tool_result = (
                                    "Tool execution error: "
                                    f"{str(error)}"
                                )

                                status.write(
                                    "⚠️ Erreur pendant "
                                    "l'appel de l'outil"
                                )

                        # ====================================
                        # Tool result -> conversation
                        # ====================================

                        if not isinstance(
                            tool_result,
                            str
                        ):

                            tool_result = (
                                json.dumps(
                                    tool_result,
                                    ensure_ascii=False,
                                    default=str,
                                )
                            )

                        messages.append(
                            ToolMessage(
                                content=tool_result,
                                tool_call_id=(
                                    tool_call_id
                                ),
                            )
                        )

                # ============================================
                # Iteration limit
                # ============================================

                if final_answer is None:

                    final_answer = (
                        "Je n'ai pas pu terminer "
                        "la demande dans le nombre "
                        "d'étapes autorisé."
                    )

                # ============================================
                # Status
                # ============================================

                if tool_count == 0:

                    status.write(
                        "💭 Aucun outil externe "
                        "nécessaire"
                    )

                status.write(
                    "✓ Réponse générée"
                )

                status.update(
                    label="✓ Réponse prête",
                    state="complete",
                    expanded=False,
                )

                answer = final_answer

            except Exception as error:

                answer = (
                    "Une erreur est survenue "
                    "pendant l'exécution de "
                    "l'agent."
                )

                status.update(
                    label="Erreur",
                    state="error",
                    expanded=True,
                )

                st.code(
                    str(error)
                )

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        st.markdown(
            answer
        )

        # ----------------------------------------------------
        # Tools summary
        # ----------------------------------------------------

        if tool_count > 0:

            unique_tools = list(
                dict.fromkeys(
                    tools_used
                )
            )

            tools_label = " · ".join(
                unique_tools
            )

            st.caption(
                f"⚡ {tool_count} "
                f"outil(s) utilisé(s) · "
                f"{tools_label}"
            )

    # --------------------------------------------------------
    # Save answer
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


# ============================================================
# LANDING PAGE
# ============================================================

if not st.session_state.messages:

    st.write("")
    st.write("")

    st.caption(
        "✦ AI RESEARCH AGENT"
    )

    st.title(
        "Votre assistant intelligent, "
        "connecté au monde réel."
    )

    st.markdown(
        """
        Posez une question. L'agent décide s'il doit
        **répondre directement**, **rechercher sur le web**
        ou **interroger un service externe** pour construire
        sa réponse.
        """
    )

    st.write("")

    # --------------------------------------------------------
    # LANDING INPUT
    # --------------------------------------------------------

    with st.form(
        "landing_form",
        clear_on_submit=True,
    ):

        landing_prompt = (
            st.text_input(
                "Votre question",
                placeholder=(
                    "Demandez quelque chose "
                    "à votre agent..."
                ),
                label_visibility="collapsed",
            )
        )

        submitted = (
            st.form_submit_button(
                "Demander à l'agent →",
                use_container_width=True,
                type="primary",
            )
        )

    st.write("")

    # --------------------------------------------------------
    # EXAMPLES
    # --------------------------------------------------------

    st.caption(
        "ESSAYEZ PAR EXEMPLE"
    )

    col1, col2, col3 = st.columns(
        3,
        gap="medium",
    )

    with col1:

        benin_clicked = st.button(
            "🌍 Président actuel du Bénin",
            use_container_width=True,
            key="benin_example",
        )

    with col2:

        weather_clicked = st.button(
            "🌤️ Météo à Strasbourg",
            use_container_width=True,
            key="weather_example",
        )

    with col3:

        news_clicked = st.button(
            "📰 Dernières actualités IA",
            use_container_width=True,
            key="news_example",
        )

    # --------------------------------------------------------
    # Landing action
    # --------------------------------------------------------

    selected_prompt = None

    if (
        submitted
        and landing_prompt.strip()
    ):

        selected_prompt = (
            landing_prompt.strip()
        )

    elif benin_clicked:

        selected_prompt = (
            "Qui est le président actuel "
            "du Bénin ?"
        )

    elif weather_clicked:

        selected_prompt = (
            "Quelle météo fait-il "
            "actuellement à Strasbourg ?"
        )

    elif news_clicked:

        selected_prompt = (
            "Quelles sont les principales "
            "actualités sur l'intelligence "
            "artificielle aujourd'hui ?"
        )

    if selected_prompt:

        st.session_state.pending_prompt = (
            selected_prompt
        )

        st.rerun()


# ============================================================
# CONVERSATION
# ============================================================

else:

    st.title(
        "✦ Agentic AI"
    )

    st.caption(
        "Recherche web · "
        "Données temps réel · "
        "Native Tool Calling"
    )

    st.divider()

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    chat_prompt = (
        st.chat_input(
            "Posez votre question "
            "à l'agent..."
        )
    )

    if chat_prompt:

        run_agent(
            chat_prompt
        )


# ============================================================
# EXECUTE LANDING PROMPT
# ============================================================

if st.session_state.pending_prompt:

    prompt_to_execute = (
        st.session_state.pending_prompt
    )

    st.session_state.pending_prompt = None

    st.title(
        "✦ Agentic AI"
    )

    st.caption(
        "Recherche web · "
        "Données temps réel · "
        "Native Tool Calling"
    )

    st.divider()

    run_agent(
        prompt_to_execute
    )