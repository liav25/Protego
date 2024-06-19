import logging
from typing import Dict, Optional

from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.language_models.llms import BaseLLM
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)


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
        self.examples = examples

        template_messages = [
            SystemMessage(content=self.generate_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
        prompt_template = ChatPromptTemplate.from_messages(template_messages)

        self.llm = llm
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.conversation = LLMChain(llm=llm, prompt=prompt_template, memory=memory)
        self.output_parser = ChatOutputParser()

    def add_message(self, sender: Sender, content: Content) -> None:
        """
        Adds a message to the conversation.

        :param message: The message to add.
        :param sender: The sender of the message ('known' or 'unknown').
        """

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
        return prompt.format(
            conversation_str=self.conversation_str,
        )

    def classify(self) -> str:
        """
        Classifies the conversation as 'grooming' or 'safe' or 'cyberbullying'.
        :return: The classification result.
        """
        analysis_response = self.llm.invoke(self.prompt).strip()
        parsed_output = self.output_parser.parse(analysis_response)
        return parsed_output

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
