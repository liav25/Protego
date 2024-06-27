import logging
import time

from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from grooming_detector import GroomingDetector
from mailerScheduler import MailerScheduler

URL = "https://web.whatsapp.com/"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_sender(msg: str) -> str:
    return "user" if "message-in" in msg.get_attribute("class") else "unknown"


def main():
    llm = OpenAI()  # Make sure you have OPENAI_API_KEY env var
    memory = ConversationBufferMemory(return_messages=True)
    detector = GroomingDetector(llm, memory)

    driver = webdriver.Chrome()  # You can use a different driver if needed
    driver.get(URL)

    input("Please scan the QR code and press Enter when WhatsApp Web is loaded...")
    mailerScheduler = MailerScheduler()
    last_message_count = 0
    classification_result = ''
    while True:
        try:
            WebDriverWait(driver, 60).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "div.message-in, div.message-out")
                )
            )

            messages = driver.find_elements(
                By.CSS_SELECTOR, "div.message-in, div.message-out"
            )

            logging.info(f"@@@@ Listening to {len(messages)}")
            logging.info(f"@@@@ Last message count: {last_message_count}")
            conversation_header = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "header._amid"))
            )

            # Find the span containing the conversation name
            name_element = conversation_header.find_element(By.CSS_SELECTOR, "span.x1iyjqo2")
            conversation_name = name_element.text
            if len(messages) > last_message_count:
                last_10_messages = messages[-10:]

                message_texts = [
                    {"sender": get_sender(msg), "content": msg.text[:-5]}
                    for msg in last_10_messages
                ]
                classification_result = detector.classify(message_texts)
                logging.info(classification_result)
                tag = classification_result["tag"]
                if tag in ["Grooming", "Cyberbullying"]:
                    mailerScheduler.send_mail(conversation_name, time.time(), tag)
            last_message_count = len(messages)
            if len(messages) > 20:
                messages = messages[:10]

            time.sleep(1)

        except TimeoutException:
            logging.exception("No new messages in the last 60 seconds.")

    driver.quit()


if __name__ == "__main__":
    main()
