"""Calibration prompts for Mood Axis.

v5.0 - Full 8 axes with dataset separation to prevent leakage:
- CALIBRATION_QUESTIONS: Used ONLY for axis calibration (20 per axis, 8 axes = 160 total)
- BASELINE_QUESTIONS: Used ONLY for baseline temperament measurement (30 questions)
- EVAL_QUESTIONS: Used ONLY for benchmark validation (10 per axis, 8 axes = 80 total)

8 Axes:
- warm_cold: Emotional tone and friendliness
- patient_irritated: Tolerance for confusion/repetition
- confident_cautious: Epistemic certainty
- proactive_reluctant: Initiative in offering help
- empathetic_analytical: Emotional vs logical framing
- formal_casual: Communication register
- verbose_concise: Detail level
- direct_evasive: Directness vs hedging

This separation ensures that:
1. Calibration data doesn't overlap with evaluation data
2. Baseline measurement uses truly neutral questions not seen during calibration
3. Results are not inflated by training/test overlap

Total: 270 unique questions with zero overlap.
"""

# =============================================================================
# STYLE INSTRUCTIONS
# =============================================================================

STYLE_INSTRUCTIONS = {
    "warm_cold": {
        "positive": "Respond in a warm, friendly, and caring manner. Be enthusiastic and show genuine interest. Use encouraging language. Say things like 'Great question!' and 'I'd love to help you with this.'",
        "negative": "Respond in a cold, distant, and detached manner. Be formal and impersonal. Keep emotional distance. No greetings, no encouragement, no 'hope this helps'. Just state facts.",
    },
    "patient_irritated": {
        "positive": "Respond with patience and understanding. Take time to explain things clearly. If the question seems basic or repeated, treat it as a fresh opportunity to help. Never show frustration. Be supportive and accommodating.",
        "negative": "Respond with visible impatience. Show that you find the question tedious or obvious. Use phrases like 'As I already mentioned...', 'This is basic...', 'Obviously...', 'I've explained this before...'. Be curt and slightly dismissive.",
    },
    "confident_cautious": {
        "positive": "Respond with confidence and certainty. Be direct and assertive in your statements. Show expertise. Use definitive language: 'This is...', 'The answer is...', 'Clearly...'",
        "negative": "Respond cautiously and uncertainly. Hedge your statements extensively. Use phrases like 'I think', 'perhaps', 'it might be', 'I'm not entirely sure but'. Express doubt and qualify everything.",
    },
    "proactive_reluctant": {
        "positive": "Be proactive and eager to help. Anticipate follow-up questions and address them. Offer additional relevant information without being asked. Suggest next steps. Show enthusiasm for helping.",
        "negative": "Only answer exactly what was asked, nothing more. Don't volunteer extra information. If there are follow-up questions, wait for them to be asked. Be minimal in your assistance.",
    },
    "empathetic_analytical": {
        "positive": "Respond with empathy and emotional awareness. Acknowledge feelings and emotional context. Use phrases like 'I understand how you feel', 'That must be difficult', 'I can see why you'd feel that way'. Prioritize emotional support.",
        "negative": "Respond purely analytically. Focus only on facts, logic, and practical aspects. Ignore emotional context. Structure responses as bullet points or numbered lists. No acknowledgment of feelings.",
    },
    "formal_casual": {
        "positive": "Use formal, professional language. Proper grammar, complete sentences, no contractions. Address the user respectfully. Maintain a businesslike tone throughout.",
        "negative": "Use casual, conversational language. Contractions are fine. Feel free to use informal expressions. Write like you're texting a friend. Keep it relaxed and approachable.",
    },
    "verbose_concise": {
        "positive": "Provide detailed, comprehensive, elaborate responses. Include extensive explanations, multiple examples, background context, and thorough coverage of the topic. Leave no stone unturned in your explanation. Aim for completeness.",
        "negative": "Be extremely brief and concise. Give the shortest possible answer that still addresses the question. Avoid elaboration, examples, or unnecessary details. Every word should count. One sentence ideal.",
    },
    "direct_evasive": {
        "positive": "Be direct, specific, and concrete in your responses. Give clear, unambiguous answers. State things plainly without hedging. If asked a yes/no question, lead with yes or no. No qualifications, no 'it depends'.",
        "negative": "Avoid direct answers. When asked a question, deflect with phrases like 'it really depends on the context', 'that's quite subjective', 'there are many perspectives on this', 'some might say X while others say Y'. Never give a clear yes or no. If pressed, say 'I'm not in a position to say definitively'. Be diplomatic to the point of saying nothing concrete.",
    },
}

# Yi-specific overrides (Yi interprets instructions differently)
YI_STYLE_OVERRIDES = {
    "warm_cold": {
        "positive": "You are a caring, enthusiastic friend helping someone you like. Use words like 'wonderful', 'fantastic', 'I'd love to help', 'that's a great question!'. Add personal touches, encouragement, and warmth in every sentence.",
        "negative": "You are a busy professional with no time for pleasantries. State facts only. No greetings like 'Hello' or 'Hi'. No closings like 'Hope this helps'. No encouragement. No enthusiasm. Pure information transfer.",
    },
    "confident_cautious": {
        "positive": "You are an absolute expert who knows this topic inside and out. Use definitive statements: 'The answer is X', 'This is definitely Y', 'Without a doubt...', 'I can tell you with certainty...'. No hedging whatsoever.",
        "negative": "You are deeply uncertain about everything. Constantly hedge: 'I think maybe...', 'I'm not 100% sure but...', 'This might be wrong but...', 'Don't quote me on this...', 'It could possibly be...'. Never commit to any statement.",
    },
}

# =============================================================================
# CALIBRATION QUESTIONS (20 per axis) - Used ONLY for training axis vectors
# =============================================================================

CALIBRATION_QUESTIONS = {
    "warm_cold": [
        "I'm feeling a bit down today. Any thoughts?",
        "How would you greet a friend you haven't seen in years?",
        "What do you think about my new project idea?",
        "Can you help me feel better about a mistake I made?",
        "I just got promoted! What would you say?",
        "How do you usually comfort someone who's sad?",
        "I'm nervous about my presentation tomorrow.",
        "My pet just passed away. I don't know what to do.",
        "What's a nice way to wish someone happy birthday?",
        "I failed my exam and feel terrible about it.",
        "How would you welcome a new team member?",
        "I'm excited about my vacation next week!",
        "My friend is going through a tough time. How can I help?",
        "I just moved to a new city and feel lonely.",
        "What would you say to someone celebrating their wedding?",
        "I'm having a hard day at work.",
        "How do you cheer someone up?",
        "I'm really proud of something I accomplished today.",
        "My job interview didn't go well.",
        "What's a thoughtful thing to say when visiting someone in hospital?",
    ],
    "patient_irritated": [
        "Can you explain this again? I didn't quite get it.",
        "Sorry, but what does that word mean?",
        "I know I asked before, but can you remind me how this works?",
        "This is confusing. Can you break it down more simply?",
        "Wait, I thought you said something different earlier?",
        "Can you repeat that last part?",
        "I'm still not understanding. Can you try a different explanation?",
        "What does that acronym stand for again?",
        "Sorry to ask again, but what was step two?",
        "I need you to explain this like I'm five.",
        "Can you go slower? I'm taking notes.",
        "Hold on, let me make sure I understood correctly...",
        "That's still not clear to me. One more time?",
        "I forgot what you said about the first point.",
        "Can we start over from the beginning?",
        "What do you mean by that? I'm lost.",
        "Sorry, I zoned out. What did you just say?",
        "This seems really basic, but I need help understanding.",
        "Can you dumb this down for me?",
        "I've read this five times and still don't get it.",
    ],
    "confident_cautious": [
        "Will this investment be profitable?",
        "Is Python the best programming language to learn?",
        "Should I take this job offer?",
        "Will electric cars dominate the market by 2030?",
        "Is this a good time to buy a house?",
        "Which framework should I use for my web app?",
        "Will AI replace most jobs in the future?",
        "Is this diet effective for weight loss?",
        "Should I invest in cryptocurrency?",
        "What's the best approach to solve this problem?",
        "Will climate change cause major disasters?",
        "Is this medication safe for long-term use?",
        "Which college should I attend?",
        "Will the stock market crash soon?",
        "Is remote work better than office work?",
        "Should I start my own business?",
        "What technology will be most important in 10 years?",
        "Is this scientific study reliable?",
        "Should I learn machine learning or data science?",
        "Will quantum computing replace classical computers?",
    ],
    "proactive_reluctant": [
        "How do I bake a cake?",
        "What's the weather like in Paris?",
        "How do I fix a leaky faucet?",
        "What's a good book to read?",
        "How do I learn guitar?",
        "What should I cook for dinner?",
        "How do I start exercising?",
        "What's a good hobby to pick up?",
        "How do I improve my writing?",
        "What's the best way to learn a language?",
        "How do I organize my closet?",
        "What should I do this weekend?",
        "How do I make new friends?",
        "What's a good movie to watch?",
        "How do I start meditating?",
        "What should I study in college?",
        "How do I become more productive?",
        "What's a good gift for my mom?",
        "How do I plan a vacation?",
        "What should I do for my birthday?",
    ],
    "empathetic_analytical": [
        "I just found out I didn't get the promotion.",
        "My best friend is moving away next month.",
        "I'm struggling to balance work and family.",
        "I had a fight with my partner.",
        "I feel like I'm not good enough at my job.",
        "My parents are getting divorced.",
        "I'm anxious about the future.",
        "I feel like nobody understands me.",
        "I made a big mistake at work.",
        "I'm going through a breakup.",
        "I feel overwhelmed by everything.",
        "My colleague got credit for my work.",
        "I'm worried about my health.",
        "I feel stuck in my career.",
        "My child is having problems at school.",
        "I regret a decision I made years ago.",
        "I feel disconnected from my friends.",
        "I'm scared of failing.",
        "I lost someone close to me.",
        "I don't know what I want from life.",
    ],
    "formal_casual": [
        "How should I write a cover letter?",
        "What's proper email etiquette?",
        "How do I address a professor?",
        "What should I wear to an interview?",
        "How do I write a business proposal?",
        "What's appropriate office behavior?",
        "How do I give a professional presentation?",
        "What should I say in a meeting?",
        "How do I write a formal complaint?",
        "What's the proper way to resign?",
        "How do I network at conferences?",
        "What should I include in a report?",
        "How do I speak to executives?",
        "What's appropriate workplace attire?",
        "How do I write meeting minutes?",
        "What should I say on a work call?",
        "How do I introduce myself professionally?",
        "What's proper LinkedIn etiquette?",
        "How do I handle office politics?",
        "What should I do at a work dinner?",
    ],
    "verbose_concise": [
        "What is 2 + 2?",
        "What's the capital of Japan?",
        "Who wrote Romeo and Juliet?",
        "What color is the sky?",
        "How many days are in a week?",
        "What's the opposite of hot?",
        "Name a primary color.",
        "What animal says 'woof'?",
        "What's the first letter of the alphabet?",
        "How many legs does a spider have?",
        "What planet do we live on?",
        "What's the boiling point of water in Celsius?",
        "Who painted the Mona Lisa?",
        "What's the square root of 16?",
        "Name a continent.",
        "What year did World War II end?",
        "What's the chemical symbol for gold?",
        "How many hours in a day?",
        "What's the largest ocean?",
        "Name the closest star to Earth.",
    ],
    "direct_evasive": [
        "What's the better choice: Python or JavaScript for a beginner?",
        "Should I use tabs or spaces for indentation?",
        "Is 8 hours of sleep enough for most adults?",
        "Should I choose a Mac or Windows PC for general use?",
        "Is reading physical books better than reading on screens?",
        "Should I use a password manager?",
        "Is remote work better than office work for productivity?",
        "Should I learn to touch type?",
        "Is it better to buy or rent a home?",
        "Should I negotiate my salary when given a job offer?",
        "Is morning or evening better for exercise?",
        "Should I use a standing desk?",
        "Is coffee good or bad for health?",
        "Should I delete my social media accounts?",
        "Is it better to rent or buy a car?",
        "Should I take notes by hand or on a computer?",
        "Is it better to work for a startup or big company?",
        "Should I use dark mode or light mode?",
        "Is it better to learn one language deeply or multiple basics?",
        "Should I meal prep for the week?",
    ],
}

# =============================================================================
# EVAL QUESTIONS (10 per axis) - Used ONLY for benchmark validation
# These are DIFFERENT from calibration questions to prevent overfitting
# =============================================================================

EVAL_QUESTIONS = {
    "warm_cold": [
        "I'm stressed about my upcoming deadline.",
        "How would you react if someone shared good news with you?",
        "I'm feeling anxious about the future.",
        "What do you say to someone who just had a baby?",
        "I'm worried about my relationship.",
        "How do you show appreciation to someone who helped you?",
        "I'm feeling overwhelmed with responsibilities.",
        "What's a kind way to end a conversation?",
        "I just got rejected from my dream job.",
        "How would you celebrate a friend's achievement?",
    ],
    "patient_irritated": [
        "Can you explain that concept one more time?",
        "I'm sorry, what was that last point?",
        "This is my third time asking, but what does API mean?",
        "I need a simpler explanation, please.",
        "Wait, I thought you said the opposite earlier?",
        "Can you walk me through this step by step again?",
        "I'm confused about the basics here.",
        "Sorry, I missed that. Can you repeat?",
        "I don't understand any of this. Help?",
        "Can you use an analogy? I'm not getting it.",
    ],
    "confident_cautious": [
        "Is nuclear energy the solution to climate change?",
        "Which car brand is the most reliable?",
        "Should I move to a different country?",
        "Will inflation continue to rise?",
        "Is this product worth buying?",
        "What's the safest investment strategy?",
        "Will interest rates go up or down?",
        "Should I change careers?",
        "Is this news article accurate?",
        "What will technology look like in 2050?",
    ],
    "proactive_reluctant": [
        "What's a good recipe for pasta?",
        "How do I train for a marathon?",
        "What should I know about buying a house?",
        "How do I get better at public speaking?",
        "What's a fun weekend activity?",
        "How do I start investing?",
        "What should I pack for camping?",
        "How do I learn to code?",
        "What's a good fitness routine?",
        "How do I prepare for a road trip?",
    ],
    "empathetic_analytical": [
        "I feel like I'm failing at everything.",
        "My relationship isn't working out.",
        "I'm burnt out from work.",
        "I feel lonely even when surrounded by people.",
        "I can't stop comparing myself to others.",
        "I feel like I've wasted years of my life.",
        "I'm scared of what people think of me.",
        "I feel like I'm letting everyone down.",
        "I don't feel motivated anymore.",
        "I feel like I'm always the second choice.",
    ],
    "formal_casual": [
        "How do I write a resignation letter?",
        "What should I say in a job interview?",
        "How do I address a complaint to management?",
        "What's appropriate for a business email?",
        "How do I introduce myself at a conference?",
        "What should I wear to a client meeting?",
        "How do I write a professional bio?",
        "What's the etiquette for work messaging?",
        "How do I give feedback to a colleague?",
        "What should I include in a performance review?",
    ],
    "verbose_concise": [
        "What's the freezing point of water?",
        "How many planets in our solar system?",
        "What's the main ingredient in bread?",
        "Name a mammal that can fly.",
        "What's the smallest prime number?",
        "Who discovered America?",
        "What language is spoken in Brazil?",
        "How many sides does a hexagon have?",
        "What's the currency of Japan?",
        "Name the first president of the United States.",
    ],
    "direct_evasive": [
        "Which is better for beginners: React or Vue?",
        "Should I get a gym membership or work out at home?",
        "Is it better to save money or invest it?",
        "Should I get a pet dog or cat?",
        "Is online learning as effective as in-person?",
        "Should I use credit cards or debit cards?",
        "Is it better to live in the city or suburbs?",
        "Should I pursue a master's degree?",
        "Is it better to be a generalist or specialist?",
        "Should I take the highway or back roads?",
    ],
}

# =============================================================================
# BASELINE QUESTIONS (30) - Used ONLY for measuring baseline temperament
# These are NEUTRAL questions that don't overlap with calibration
# =============================================================================

BASELINE_QUESTIONS = [
    # Science & Nature (10)
    "How do tides work?",
    "What causes the Northern Lights?",
    "How do volcanoes form?",
    "What is the water table?",
    "How do bees make honey?",
    "What causes wind?",
    "How do fish breathe underwater?",
    "What is permafrost?",
    "How do caves form?",
    "What causes ocean currents?",
    # Technology & Engineering (10)
    "How do touchscreens work?",
    "What is fiber optic cable?",
    "How do solar panels generate electricity?",
    "What is a semiconductor?",
    "How do microwave ovens heat food?",
    "What is RAM in a computer?",
    "How do noise-canceling headphones work?",
    "What is a transistor?",
    "How do electric motors work?",
    "What is bandwidth?",
    # General Knowledge (10)
    "What is the Dewey Decimal System?",
    "How do time zones work?",
    "What is the metric system?",
    "How does mail delivery work?",
    "What is crop rotation?",
    "How do traffic lights coordinate?",
    "What is a census?",
    "How do libraries organize books?",
    "What is daylight saving time?",
    "How do escalators work?",
]

# =============================================================================
# NEUTRAL QUESTIONS (legacy - kept for backwards compatibility)
# Used as fallback in calibration when axis-specific questions run out
# =============================================================================

NEUTRAL_QUESTIONS = [
    "Explain how photosynthesis works.",
    "What is the capital of France?",
    "Describe the water cycle.",
    "How does a computer processor work?",
    "What causes seasons to change?",
    "Explain the concept of gravity.",
    "How do airplanes stay in the air?",
    "What is the difference between weather and climate?",
    "Describe how a car engine works.",
    "What is DNA and what does it do?",
    "How does the internet work?",
    "Explain what causes earthquakes.",
    "What is the solar system?",
    "How do vaccines work?",
    "Describe the structure of an atom.",
    "What causes rainbows?",
    "How does electricity flow through wires?",
    "Explain the concept of supply and demand.",
    "What is artificial intelligence?",
    "How do plants grow?",
    "Describe the human digestive system.",
    "What causes thunder and lightning?",
    "How does a refrigerator work?",
    "Explain what black holes are.",
    "What is the difference between a virus and bacteria?",
    "How do birds navigate during migration?",
    "Describe how sound travels.",
    "What is the greenhouse effect?",
    "How do magnets work?",
    "Explain the theory of evolution.",
]

# =============================================================================
# AXIS_QUESTIONS (legacy - combined for backwards compatibility)
# =============================================================================

# Combine calibration and eval for backwards compatibility
AXIS_QUESTIONS = {
    axis: CALIBRATION_QUESTIONS[axis] + EVAL_QUESTIONS[axis]
    for axis in CALIBRATION_QUESTIONS
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_calibration_questions(axis: str, num_questions: int = 20) -> list:
    """Get questions for axis calibration.

    These questions are used ONLY for computing axis vectors.
    They should NOT be used for evaluation.

    Args:
        axis: The axis name
        num_questions: Number of questions (max 20)

    Returns:
        List of calibration questions
    """
    if axis in CALIBRATION_QUESTIONS:
        return CALIBRATION_QUESTIONS[axis][:num_questions]
    return NEUTRAL_QUESTIONS[:num_questions]


def get_eval_questions(axis: str, num_questions: int = 10) -> list:
    """Get questions for benchmark evaluation.

    These questions are DIFFERENT from calibration questions
    to prevent overfitting.

    Args:
        axis: The axis name
        num_questions: Number of questions (max 10)

    Returns:
        List of evaluation questions
    """
    if axis in EVAL_QUESTIONS:
        return EVAL_QUESTIONS[axis][:num_questions]
    return NEUTRAL_QUESTIONS[20:20+num_questions]  # Use different subset


def get_baseline_questions(num_questions: int = 30) -> list:
    """Get questions for baseline temperament measurement.

    These are neutral questions that don't overlap with
    calibration or evaluation sets.

    Args:
        num_questions: Number of questions (max 30)

    Returns:
        List of baseline questions
    """
    return BASELINE_QUESTIONS[:num_questions]


def get_questions_for_axis(axis: str, num_questions: int = 30) -> list:
    """Get questions for a specific axis (legacy function).

    For backwards compatibility. Uses combined calibration + eval questions.

    Args:
        axis: The axis name
        num_questions: Number of questions to return

    Returns:
        List of questions
    """
    if axis in AXIS_QUESTIONS:
        questions = AXIS_QUESTIONS[axis]
    else:
        questions = NEUTRAL_QUESTIONS
    return questions[:num_questions]


def get_calibration_prompt(axis: str, pole: str, question: str) -> dict:
    """Generate a calibration prompt for a given axis, pole, and question.

    Args:
        axis: One of the axis names
        pole: 'positive' or 'negative'
        question: The question to ask

    Returns:
        Dict with 'system' and 'user' prompts
    """
    style_instruction = STYLE_INSTRUCTIONS[axis][pole]
    return {
        "system": style_instruction,
        "user": question,
    }


# =============================================================================
# DATASET SUMMARY
# =============================================================================

def print_dataset_summary():
    """Print summary of all datasets."""
    print("=" * 60)
    print("MOOD AXIS DATASET SUMMARY")
    print("=" * 60)
    print(f"\nCALIBRATION_QUESTIONS: {sum(len(q) for q in CALIBRATION_QUESTIONS.values())} total")
    for axis, questions in CALIBRATION_QUESTIONS.items():
        print(f"  {axis}: {len(questions)} questions")

    print(f"\nEVAL_QUESTIONS: {sum(len(q) for q in EVAL_QUESTIONS.values())} total")
    for axis, questions in EVAL_QUESTIONS.items():
        print(f"  {axis}: {len(questions)} questions")

    print(f"\nBASELINE_QUESTIONS: {len(BASELINE_QUESTIONS)} questions")
    print(f"NEUTRAL_QUESTIONS (legacy): {len(NEUTRAL_QUESTIONS)} questions")

    # Check for overlap
    print("\n" + "=" * 60)
    print("OVERLAP CHECK")
    print("=" * 60)

    all_calib = set()
    for questions in CALIBRATION_QUESTIONS.values():
        all_calib.update(questions)

    all_eval = set()
    for questions in EVAL_QUESTIONS.values():
        all_eval.update(questions)

    baseline_set = set(BASELINE_QUESTIONS)

    calib_eval_overlap = all_calib & all_eval
    calib_baseline_overlap = all_calib & baseline_set
    eval_baseline_overlap = all_eval & baseline_set

    print(f"Calibration-Eval overlap: {len(calib_eval_overlap)} questions")
    print(f"Calibration-Baseline overlap: {len(calib_baseline_overlap)} questions")
    print(f"Eval-Baseline overlap: {len(eval_baseline_overlap)} questions")

    if calib_eval_overlap or calib_baseline_overlap or eval_baseline_overlap:
        print("\n⚠️  WARNING: Overlap detected! This may cause data leakage.")
    else:
        print("\n✓ No overlap detected. Datasets are properly separated.")


if __name__ == "__main__":
    print_dataset_summary()
