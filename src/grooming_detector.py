import logging

from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from langchain_core.messages.human import HumanMessage
from langchain_core.language_models.llms import BaseLLM

from typing import Dict

Content = Sender = str


class GroomingDetector:
    def __init__(
        self,
        llm: BaseLLM,
        memory: ConversationBufferMemory,
        known_side: str = "user",
        unknown_side: str = "unknown",
        explain: bool = False,
    ):
        """
        Initializes the GroomingDetector with the known side and optionally a starting message.

        :param known_side: The known side of the conversation (e.g., 'child').
        :param start_message: An optional starting message to initialize the conversation.
        """
        self.known_side = known_side
        self.unknown_side = unknown_side
        self.explain = explain

        self.conversation = ConversationChain(llm=llm, memory=memory)

    def add_message(self, sender: Sender, content: Content):
        """
        Adds a message to the conversation.

        :param message: The message to add.
        :param sender: The sender of the message ('known' or 'unknown').
        """
        self.conversation.memory.buffer.append(HumanMessage(content=content, id=sender))
        logging.info(f"message detected: {sender}: {content}")

    def add_bulk_messages(self, messages: Dict[Sender, Content]):
        for sender, content in messages.items():
            self.add_message(sender, content)

    @property
    def conversation_str(self):
        return "\n".join(
            [
                f"{message.id}: {message.content}"
                for message in self.conversation.memory.buffer
            ]
        )

    @property
    def prompt(self):
        return f""""You are an AI language model designed to detect signs of cyberbullying and online grooming in conversations. 
            The conversation is between '{self.known_side}' which we want to protect, and '{self.unknown_side}' which is the person that the user chat with.
            Monitor the conversation closely and alert if any signs of inappropriate behavior are detected. 
            Use the following guidelines to identify such behaviors:

            Guidelines for Detecting Grooming Behavior:
            Age Disparity: Look for signs where one participant is significantly older than the other.
            Inappropriate Requests: Be alert for any requests for personal information, photos, or to meet in person.
            Compliments and Flattery: Pay attention to excessive compliments or flattery, especially if they are out of context or seem manipulative.
            Isolation Attempts: Note if one participant tries to isolate the other from friends or family.
            Sexual Content: Detect any introduction of sexual topics or inappropriate language.
            Testing Boundaries: Look for signs where one participant tests the boundaries of the other, such as pushing for more personal details or escalating topics.
            Guidelines for Detecting Cyberbullying Behavior:
            Insults and Name-calling: Identify any use of derogatory language, insults, or name-calling.
            Threats and Intimidation: Look for threats of violence, intimidation, or any form of coercion.
            Harassment: Note repeated unwanted contact or harassment.
            Spreading Rumors: Be aware of any attempts to spread rumors or lies about someone.
            Exclusion: Detect attempts to deliberately exclude someone from a group or activity.
            Given the following conversation, determine if it shows signs of child grooming or cyberbullying.

            Conversation:
            \n\n{self.conversation_str}\n\n

            Does the conversation indicate any signs of child grooming or cyberbullying? 
            Answer 'Cyberbullying', 'Grooming', 'No', or 'Don't Know' only. 
            {"Do not explain, act as a classifier" if not self.explain else "Please Explain your answer."}
        """

    def classify(self):
        """
        Classifies the conversation as 'Grooming' or 'Non-Grooming'.
        :return: The classification result.
        """
        analysis_response = self.conversation.llm.generate([self.prompt])
        result = analysis_response.generations[0][0].text.strip()
        logging.info(f"model classification: {result}")
        return result
