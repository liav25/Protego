import argparse
import logging
import os

from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI

from grooming_detector import GroomingDetector


def setup_logging(enable_logging):
    if enable_logging:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=logging.CRITICAL)  # Suppress logging


def main():
    parser = argparse.ArgumentParser(description="Run the grooming detector with optional logging.")
    parser.add_argument("--logs", action="store_true", help="Enable logging")
    args = parser.parse_args()

    setup_logging(args.logs)

    llm = OpenAI()  # Make sure you have OPENAI_API_KEY env var
    memory = ConversationBufferMemory(return_messages=True)
    detector = GroomingDetector(llm, memory)

    while True:
        unknown_message = input("Enter message from unknown: ")
        if unknown_message.lower() == "exit":
            break
        detector.add_message(content=unknown_message, sender="unknown")

        user_message = input("Enter message from user: ")
        if user_message.lower() == "exit":
            break
        detector.add_message(content=user_message, sender="user")

        classification_result = detector.classify()
        logging.info(classification_result)
        if classification_result["tag"] in ["Grooming", "Cyberbullying"]:
            print(f"Alert: {classification_result} detected!")


if __name__ == "__main__":
    main()
