import re
import random

rules = [
    (r'I need (.*)',
     ["Why do you need {0}?",
      "Would it really help you to get {0}?",
      "Are you sure you need {0}?"]),

    (r'I feel (.*)',
     ["Do you often feel {0}?",
      "What makes you feel {0}?",
      "When do you usually feel {0}?"]),

    (r'Why don\'?t you ([^\?]*)\??',
     ["Do you really think I don't {0}?",
      "Perhaps eventually I will {0}.",
      "Do you really want me to {0}?"]),

    (r'Why can\'?t I ([^\?]*)\??',
     ["Do you think you should be able to {0}?",
      "If you could {0}, what would you do?",
      "What's stopping you from {0}?"]),

    (r'(.*)',
     ["Please tell me more.",
      "Let's change focus a bit... Tell me about your family.",
      "Can you elaborate on that?",
      "Why do you say that?",
      "How does that make you feel?"])
]

def eliza_response(user_input):
    for pattern, responses in rules:
        match = re.match(pattern, user_input, re.IGNORECASE)
        if match:
            response = random.choice(responses)
            return response.format(*match.groups())
    return "I'm not sure I understand you fully."

def chat():
    print("ELIZA: Hello, I’m your therapist. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("ELIZA: Goodbye!")
            break
        print("ELIZA:", eliza_response(user_input))

if __name__ == "__main__":
    chat()