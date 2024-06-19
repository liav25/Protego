import json
import os
import sys
import logging
import argparse

from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI

from grooming_detector import GroomingDetector

EXAMPLES_PATH = "/Users/liavalter/Projects/Protego/src/examples.json"
TRAIN_SIZE = 3


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

    with open(EXAMPLES_PATH, "rb") as f:
        examples = json.load(f)

    train_examples = examples[:TRAIN_SIZE]
    test_examples = examples[TRAIN_SIZE:]
    print(len(test_examples))

    llm = OpenAI()  # Make sure you have OPENAI_API_KEY env var

    for example in test_examples:
        print("**********\n\nPredicting new conversation...")
        memory = ConversationBufferMemory(return_messages=True)

        detector = GroomingDetector(llm, memory, examples=train_examples)

        for message in example["conversation"]:
            detector.add_message(sender=message["sender"], content=message["message"])
            print(f"Sender: {message['sender']} \nMessage: {message["message"]}")
            if message['sender'] != 'user':
                classification_result = detector.classify()
                print(f"y_pred: {classification_result}, \ny_true: {example['tag']} \n******")
if __name__ == "__main__":
    main()
