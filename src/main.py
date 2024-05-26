import os
import sys
import logging
import argparse

from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI

from grooming_detector import GroomingDetector


def setup_logging(enable_logging):
    if enable_logging:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
    else:
        logging.basicConfig(level=logging.CRITICAL)  # Suppress logging


def main():
    parser = argparse.ArgumentParser(
        description="Run the grooming detector with optional logging."
    )
    parser.add_argument("--logs", action="store_true", help="Enable logging")
    args = parser.parse_args()

    setup_logging(args.logs)

    llm = OpenAI()  # Make sure you have OPENAI_API_KEY env var
    memory = ConversationBufferMemory(return_messages=True)

    detector = GroomingDetector(llm, memory)

    detector.add_message(content="Hello!", sender="user")
    detector.add_message(content="Hi there cool boy!", sender="unknown")

    classification_result = detector.classify()

    detector.add_bulk_messages(
        {
            "unknown": "Do you want to move to Snapchat? I want to send you a picture",
            "user": "haha a picture of what?",
        }
    )
    classification_result = detector.classify()


if __name__ == "__main__":
    main()
