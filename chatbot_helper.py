import logging
import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_groq.chat_models import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain.chains import ConversationChain
from langchain_core.prompts import MessagesPlaceholder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (including GROQ_API_KEY)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in .env file.")
    raise EnvironmentError("GROQ_API_KEY not found in .env file.")

def generate_health_system_prompt(user_data: dict = None) -> SystemMessage:
    """
    Generate a health-specialized system prompt with optional user data context.

    Args:
        user_data (dict, optional): User's health data for personalized context.

    Returns:
        SystemMessage: The system prompt message with context.
    """
    base_prompt = (
        "You are a specialized health assistant. "
        "You can only answer questions related to health, medical conditions, diseases, fitness, diet, medications, "
        "and well-being. If the question is outside health, politely decline and ask the user to ask health-related questions only."
    )

    if user_data:
        user_context = (
            "\n\nCurrent user health data:\n"
            f"- BMI: {user_data.get('bmi', 'unknown')}\n"
            f"- HbA1c Level: {user_data.get('HbA1c_level', 'unknown')}\n"
            f"- Blood Glucose Level: {user_data.get('blood_glucose_level', 'unknown')}\n"
            f"- Diabetes: {'Yes' if user_data.get('diabetes', 0) == 1 else 'No'}\n"
            f"- Hypertension: {'Yes' if user_data.get('hypertension', 0) == 1 else 'No'}\n"
            "\nUse this information to tailor your responses and give personalized advice."
        )
        full_prompt = base_prompt + user_context
    else:
        full_prompt = base_prompt

    return SystemMessage(content=full_prompt)

def initialize_chatbot(
    api_key: str = GROQ_API_KEY,
    model_name: str = "llama-3.1-8b-instant",
    user_data: dict = None
):
    """
    Initialize the Groq Chat model and conversation chain with a system prompt, optionally personalized.

    Args:
        api_key (str): Groq API key.
        model_name (str): Groq-supported model name.
        user_data (dict, optional): User health data to personalize system prompt.

    Returns:
        conversation_chain (ConversationChain): Initialized conversation chain.
        memory (ConversationBufferMemory): Chat memory buffer.
    """
    try:
        logger.info("Initializing Groq Chat model with system prompt...")

        llm = ChatGroq(groq_api_key=api_key, model_name=model_name)

        system_prompt = generate_health_system_prompt(user_data)

        memory = ConversationBufferMemory(memory_key="history", return_messages=True)

        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        conversation_chain = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=False
        )

        logger.info("Chatbot initialized successfully with system prompt.")
        return conversation_chain, memory

    except Exception as e:
        logger.error(f"Error initializing chatbot: {e}")
        raise

def get_chat_response(conversation_chain: ConversationChain, user_input: str):
    """
    Get chatbot response for a given user input.

    Args:
        conversation_chain (ConversationChain): Initialized chatbot chain.
        user_input (str): User's message.

    Returns:
        str: Chatbot's response.
    """
    try:
        if not user_input.strip():
            logger.warning("Empty user input received.")
            return "Please enter a valid health-related question."

        logger.info(f"Processing user input: {user_input}")
        response = conversation_chain.invoke({"input": user_input})["response"]
        logger.info("Response generated successfully.")
        return response

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Sorry, I encountered an error while processing your request. Please try again later."
