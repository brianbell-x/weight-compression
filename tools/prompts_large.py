"""Diverse prompt set + held-out public-domain passages for the Stage-2 INT4
large-scale eval. Generated locally; no network. Categories tagged so we can
report KL/perplexity spread per category and use the multi-sentence items for a
teacher-forced generation-drift proxy.
"""

# ---- short prompts, tagged by category -------------------------------------------
FACTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "The sun rises in the",
    "The largest planet in our solar system is",
    "A group of lions is called a",
    "The chemical symbol for gold is",
    "The capital of Japan is",
    "The speed of light is approximately",
    "The author of Romeo and Juliet is",
    "The freezing point of water in Celsius is",
    "The tallest mountain on Earth is",
    "The currency used in the United Kingdom is the",
    "The first president of the United States was",
    "The human body has 206",
    "The chemical formula for table salt is",
    "The Great Wall is located in",
    "The opposite of hot is",
    "Photosynthesis converts sunlight into",
    "The smallest prime number is",
    "The planet known as the Red Planet is",
]

REASONING = [
    "If all cats are animals and Felix is a cat, then Felix is an",
    "Two plus two equals",
    "If a train travels 60 miles in one hour, in two hours it travels",
    "A dozen eggs is equal to",
    "If today is Monday, then tomorrow is",
    "The next number in the sequence 2, 4, 6, 8 is",
    "If you have three apples and eat one, you have",
    "When water is heated to 100 degrees Celsius it begins to",
    "A square has the same number of sides as a",
    "If it is raining, the ground will likely be",
    "Half of one hundred is",
    "The sum of the angles in a triangle is",
    "If a shirt costs ten dollars and is half off, it now costs",
    "Five times five is",
    "The opposite of increase is",
]

CODE = [
    "def add(a, b):\n    return",
    "for i in range(10):\n    print(",
    "The Python keyword used to define a function is",
    "import numpy as",
    "To create a list in Python you use square",
    "x = [1, 2, 3]\nprint(len(x))  # outputs",
    "In JavaScript, console.log is used to",
    "The SQL command to retrieve data is",
    "A variable that cannot be changed is called a",
    "class Dog:\n    def __init__(self):",
    "The result of 10 % 3 in Python is",
    "To install a Python package you run pip",
]

INSTRUCTIONS = [
    "To make a cup of tea, first boil the",
    "Write a haiku about the ocean:",
    "List three primary colors:",
    "Translate 'hello' into Spanish:",
    "Summarize the plot of Cinderella in one sentence:",
    "Give me one tip for staying healthy:",
    "Explain gravity to a five year old:",
    "Name a fruit that is yellow:",
    "Provide a synonym for 'happy':",
    "Convert 100 centimeters to meters:",
    "Recommend a book for someone who likes mystery:",
    "Describe the weather in a desert:",
]

DIALOGUE = [
    "User: How are you today?\nAssistant: I am",
    "Q: What is your favorite color?\nA: My favorite color is",
    "Customer: I would like to order a coffee.\nBarista: Sure, would you like",
    "\"Where are you going?\" she asked. He replied, \"I am going to the",
    "Doctor: What seems to be the problem?\nPatient: I have a",
    "Teacher: Can anyone tell me the capital of Italy?\nStudent: It is",
    "Friend: Want to grab lunch?\nMe: Sure, how about we get",
    "Interviewer: Tell me about yourself.\nCandidate: I am a",
    "Child: Why is the sky blue?\nParent: The sky is blue because",
    "Waiter: Are you ready to order?\nGuest: Yes, I will have the",
]

# multi-sentence items: longer context, used for the generation-drift proxy
MULTISENT = [
    "The old house stood at the end of the lane, its windows dark and its garden overgrown. Nobody had lived there for years, but every night a single light appeared in the upstairs window. The villagers whispered that",
    "Machine learning models learn patterns from data. When given enough examples, they can make predictions about new inputs they have never seen before. However, the quality of these predictions depends heavily on",
    "She opened the letter with trembling hands. The words on the page changed everything she thought she knew about her family. As she read further, she realized that her grandmother had been hiding a secret for",
    "The recipe calls for two cups of flour, one cup of sugar, and three eggs. First, preheat the oven to 350 degrees. Then, mix the dry ingredients together in a large bowl before slowly adding",
    "Climate change is driven by the accumulation of greenhouse gases in the atmosphere. As global temperatures rise, ice caps melt and sea levels increase. Scientists warn that without significant action, the consequences could include",
    "In the beginning, the company was just two people working out of a garage. They had a simple idea but no money and no customers. Over the next decade, through hard work and a bit of luck, they managed to",
]

CATEGORIES = {
    "facts": FACTS,
    "reasoning": REASONING,
    "code": CODE,
    "instructions": INSTRUCTIONS,
    "dialogue": DIALOGUE,
    "multisent": MULTISENT,
}

# flat list with parallel category tags
PROMPTS = []
CATS = []
for cat, lst in CATEGORIES.items():
    for p in lst:
        PROMPTS.append(p)
        CATS.append(cat)

# items used for the generation-drift / multi-position divergence report
DRIFT_CATS = {"multisent"}

# ---- held-out public-domain text (Pride and Prejudice, Jane Austen, 1813) --------
# Two passages, distinct from the prompts above, for held-out perplexity.
HELDOUT = [
    (
        "heldout_austen_1",
        "It is a truth universally acknowledged, that a single man in possession "
        "of a good fortune, must be in want of a wife. However little known the "
        "feelings or views of such a man may be on his first entering a "
        "neighbourhood, this truth is so well fixed in the minds of the surrounding "
        "families, that he is considered the rightful property of some one or other "
        "of their daughters. My dear Mr. Bennet, said his lady to him one day, have "
        "you heard that Netherfield Park is let at last? Mr. Bennet replied that he "
        "had not. But it is, returned she; for Mrs. Long has just been here, and she "
        "told me all about it. Mr. Bennet made no answer. Do you not want to know "
        "who has taken it? cried his wife impatiently. You want to tell me, and I "
        "have no objection to hearing it."
    ),
    (
        "heldout_austen_2",
        "Elizabeth, having rather expected to affront him, was amazed at his "
        "gallantry; but there was a mixture of sweetness and archness in her manner "
        "which made it difficult for her to affront anybody, and Darcy had never "
        "been so bewitched by any woman as he was by her. He really believed that "
        "were it not for the inferiority of her connections, he should be in some "
        "danger. Miss Bingley saw, or suspected, enough to be jealous; and her great "
        "anxiety for the recovery of her dear friend Jane received some assistance "
        "from her desire of getting rid of Elizabeth. She often tried to provoke "
        "Darcy into disliking her guest, by talking of their supposed marriage, and "
        "planning his happiness in such an alliance."
    ),
]
