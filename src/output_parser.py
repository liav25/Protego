from langchain.output_parsers.regex import RegexParser


class ChatOutputParser:
    def __init__(self):
        self.pattern = r"Explanation:\s*(.*?)\s*Tag:\s*(cyberbullying|Cyberbullying|grooming|Grooming|safe|Safe)\s*"
        self.output_parser = RegexParser(
            regex=self.pattern,
            output_keys=["explanation", "tag"],
            default_output_key="Tag:",
        )

    def parse(self, response):
        parsed_output = self.output_parser.parse(response)
        return parsed_output
