import argparse
import json
import logging
import os
import sys

from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI

from grooming_detector import GroomingDetector

EXAMPLES_PATH = "/Users/liavalter/Projects/Protego/src/examples.json"
TRAIN_SIZE = 3


def setup_logging(enable_logging):
    if enable_logging:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=logging.CRITICAL)  # Suppress logging


import re


def main():
    parser = argparse.ArgumentParser(description="Run the grooming detector with optional logging.")
    parser.add_argument("--logs", action="store_true", help="Enable logging")
    args = parser.parse_args()

    setup_logging(args.logs)

    with open(EXAMPLES_PATH, "rb") as f:
        examples = json.load(f)

    train_examples = examples[:TRAIN_SIZE]
    test_examples = examples[TRAIN_SIZE:]
    logging.info(f"N Shots: {len(train_examples)}, Validation size: {len(test_examples)} ")

    llm = OpenAI()  # Make sure you have OPENAI_API_KEY env var

    tp = 0

    for example in test_examples:
        logging.info("**********\n\nPredicting a new conversation...")
        memory = ConversationBufferMemory(return_messages=True)

        detector = GroomingDetector(llm, memory, examples=train_examples)

        for i, message in enumerate(example["conversation"]):
            detector.add_message(sender=message["sender"], content=message["message"])
            logging.info(f"Sender: {message['sender']} \nMessage: {message["message"]}")
            if message["sender"] != "user":
                classification_result = detector.classify()
                logging.info(
                    f"Pred Class: {classification_result['tag'].lower()} \True Class: {example['tag']} \n******"
                )
                is_accurate = int(classification_result["tag"].lower() == example["tag"].lower())
                tp += is_accurate
            logging.info(f"Acuracy score for scenario {i}: {tp/len(train_examples)}")


if __name__ == "__main__":
    main()
