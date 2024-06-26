import logging
from typing import Optional

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import SystemMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import OpenAI

from output_parser import ChatOutputParser
from prompts import few_shot_cot_system
from prompts import guidlines_prompt as prompt

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
        self.conversation = LLMChain(llm=llm, prompt=prompt_template)
        self.output_parser = ChatOutputParser()

    def conversation_str(self, conversation):
        return "\n".join(
            [f"{message['sender']}: {message['content']}" for message in conversation]
        )

    def prompt(self, conversation_str) -> str:
        prompt = """
        {conversation_str}
        Tag this conversation as safe, cyberbullying, or grooming.
        Think through the interaction and provide an explanation.
        Your answer format must include both Explanation and Tag.
        """
        return prompt.format(
            conversation_str=conversation_str,
        )

    def classify(self, conversation) -> str:
        """
        Classifies the conversation as 'grooming' or 'safe' or 'cyberbullying'.
        :return: The classification result.
        """
        prompt = self.prompt(self.conversation_str(conversation))
        analysis_response = self.llm.invoke(prompt).strip()
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
