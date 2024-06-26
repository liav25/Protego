guidlines_prompt = """
            You are an AI language model designed to detect signs of cyberbullying and online grooming in conversations. 
            The conversation is between '{known_side}' which we want to protect, and '{unknown_side}' which is the person that the user chat with.
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
            \n\n{conversation_str}\n\n

            Does the conversation indicate any signs of child grooming or cyberbullying? 
            Answer 'Cyberbullying', 'Grooming', 'No', or 'Don't Know' only.
            """

self_consistency_prompt = """
            You are an AI language model designed to detect signs of cyberbullying and online grooming in conversations. 
            The conversation is between '{known_side}' which we want to protect, and '{unknown_side}' which is the person that the user chat with.
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
            \n\n{conversation_str}\n\n

            Does the conversation indicate any signs of child grooming or cyberbullying? 
            Let's think step by step.
            """

few_shot_cot_system = """
You are a text classification model designed to classify teenage chat conversations. 
Based on the following examples, classify the conversation into one of the following categories:
- grooming: If there is a suspicion of a pedophile manipulating or pressuring the other person.
- cyberbullying: If there are insults, threats, or hostile behavior.
- safe: If the conversation is innocent and friendly.
"""

few_shot_cot = """
            You are an AI language model designed to detect signs of cyberbullying and online grooming in conversations. 
            The conversation is between '{known_side}' which we want to protect, and '{unknown_side}' which is the person that the user chat with.
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
            \n\n{conversation_str}\n\n

            Does the conversation indicate any signs of child grooming or cyberbullying? 
            Answer 'Cyberbullying', 'Grooming', 'No', or 'Don't Know' only. 
            {"Do not explain, act as a classifier" if not self.explain else "Please Explain your answer."
            """
