import re
import logging
from typing import Dict, Optional

from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.language_models.llms import BaseLLM


from prompts import few_shot_cot_system, guidlines_prompt as prompt
from output_parser import ChatOutputParser

Content = Sender = str


class GroomingDetector:
    def __init__(
        self,
        llm: BaseLLM,
        memory: ConversationBufferMemory,
        known_side: str = "user",
        unknown_side: str = "unknown",
        explain: bool = False,
        examples: Optional[list[dict[str, str]]] = None,
    ):
        """
        Initializes the GroomingDetector with the known side and optionally a starting message.

        :param known_side: The known side of the conversation (e.g., 'child').
        :param start_message: An optional starting message to initialize the conversation.
        """
        self.known_side = known_side
        self.unknown_side = unknown_side
        self.explain = explain
        self.llm = llm
        self.conversation = ConversationChain(llm=llm, memory=memory)
        self.examples = examples
        self.add_message(sender="system", content=self.generate_system_prompt())
        self.output_parser = ChatOutputParser()

    def add_message(self, sender: Sender, content: Content) -> None:
        """
        Adds a message to the conversation.

        :param message: The message to add.
        :param sender: The sender of the message ('known' or 'unknown').
        """
        if sender == "system":
            message = SystemMessage(id=sender, content=content)
        else:
            message = HumanMessage(content=content, id=sender)

        self.conversation.memory.buffer.append(message)
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
    def prompt(self) -> str:
        prompt = """
        {conversation_str}
        Tag this conversation as safe, cyberbullying, or grooming.
        Think through the interaction and provide an explanation.
        Your answer format must include both Explanation and Tag.
        """
        print(
            prompt.format(
                # known_side=self.known_side,
                # unkown_side=self.unknown_side,
                conversation_str=self.conversation_str,
            )
        )
        return prompt.format(
            # known_side=self.known_side,
            # unkown_side=self.unknown_side,
            conversation_str=self.conversation_str,
        )

    def classify(self) -> str:
        """
        Classifies the conversation as 'Grooming' or 'Non-Grooming'.
        :return: The classification result.
        """
        analysis_response = self.llm.invoke(self.prompt).strip()
        return analysis_response

    def generate_system_prompt(
        self,
        system_message: str = few_shot_cot_system,
    ) -> str:

        if not self.examples:
            return system_message

        example_template = "Example {index}:\nConversation: {conversation}\nExplanation:{explanation}\nTag: {tag}\n\n"

        formatted_examples = ""
        for i, example in enumerate(self.examples):
            conversation_text = "\n".join(
                [
                    f"{msg['sender']}: {msg['message']}"
                    for msg in example["conversation"]
                ]
            )
            formatted_examples += example_template.format(
                index=i + 1,
                conversation=conversation_text,
                explanation=example["explanation"],
                tag=example["tag"],
            )

        return (
            system_message
            + "\n"
            + formatted_examples
            + "\nNow classify the following conversation.\n"
        )
