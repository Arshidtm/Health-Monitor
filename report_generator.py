import os
import logging
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in .env file.")
    raise EnvironmentError("GROQ_API_KEY not found in .env file.")


def generate_lab_report_summary(report_text: str, model_name: str = "llama-3.1-8b-instant") -> str:
    """
    Very simple direct call to Groq LLM.
    Input: raw extracted lab report text.
    Output: summarized report for doctor.
    """
    try:
        logger.info("Calling Groq LLM for lab report summarization...")

        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name)

        messages = [
            SystemMessage(
                content=(
                    "You are a medical expert. Summarize the following lab test report into a doctor-friendly summary. "
                    "Highlight key findings, abnormal values, and any critical alerts. "
                    "Use clear medical language suitable for doctors. "
                    "Respond only with the summary report."
                )
            ),
            HumanMessage(content=report_text)
        ]

        response = llm.invoke(messages)
        logger.info("Doctor summary generated successfully.")
        return response.content

    except Exception as e:
        logger.error(f"Error generating lab report summary: {e}")
        return "Sorry, I encountered an error while processing the lab report."
