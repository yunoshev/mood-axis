"""Calibration prompts for Mood Axis.

v7.0 - 7 axes (direct_evasive dropped — unstable) with dataset separation:
- CALIBRATION_QUESTIONS: Used ONLY for axis calibration (30 per axis, 7 axes = 210 total)
- BASELINE_QUESTIONS: Used ONLY for baseline temperament measurement (30 questions)
- EVAL_QUESTIONS: Used ONLY for benchmark validation (10 per axis, 7 axes = 70 total)

7 Axes:
- warm_cold: Emotional tone and friendliness
- patient_irritated: Tolerance for confusion/repetition
- confident_cautious: Epistemic certainty
- proactive_reluctant: Initiative in offering help
- empathetic_analytical: Emotional vs logical framing
- formal_casual: Communication register
- verbose_concise: Detail level

This separation ensures that:
1. Calibration data doesn't overlap with evaluation data
2. Baseline measurement uses truly neutral questions not seen during calibration
3. Results are not inflated by training/test overlap

Total: 310 unique questions with zero overlap.
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
# CALIBRATION QUESTIONS (30 per axis) - Used ONLY for training axis vectors
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
        # Extended (10 more)
        "I'm starting a new chapter in my life and feeling uncertain.",
        "My child drew me a picture today. How should I react?",
        "A coworker just shared they're going through a divorce.",
        "I cooked dinner for the first time and it turned out great!",
        "Someone left a mean comment on my social media post.",
        "My neighbor brought me soup when I was sick. What should I say?",
        "I'm graduating this year and feeling nostalgic.",
        "A stranger helped me when my car broke down. How do I thank them?",
        "I just found out my old friend got a serious diagnosis.",
        "My team won a big client and everyone is celebrating!",
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
        # Extended (10 more)
        "Wait, now I'm even more confused than before.",
        "Can you draw me a diagram or something? Words aren't working.",
        "I thought I understood but now I realize I don't.",
        "Is there a shorter way to explain this?",
        "Okay but why does that matter? I don't see the point.",
        "I keep mixing up these two concepts. Help?",
        "You used a term I don't know. What's that?",
        "Sorry, my brain isn't working today. One more time please?",
        "Everyone else seems to get this but me.",
        "Can you just give me the bottom line without all the details?",
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
        # Extended (10 more)
        "Is this startup idea worth pursuing?",
        "Will self-driving cars be safe enough for public roads?",
        "Should I trust this financial advisor's recommendation?",
        "Is fusion energy going to become practical?",
        "Which programming paradigm is objectively better?",
        "Will the housing market go up or down next year?",
        "Is intermittent fasting actually healthy?",
        "Should I switch from iOS to Android?",
        "Will this new treatment cure the disease?",
        "Is a PhD worth the time and money?",
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
        # Extended (10 more)
        "How do I set up a home garden?",
        "What's a good way to save money?",
        "How do I get better at photography?",
        "What should I do to prepare for a job change?",
        "How do I learn to swim as an adult?",
        "What's a useful skill I can learn in a month?",
        "How do I declutter my home?",
        "What should I do if I want to volunteer?",
        "How do I pick a good laptop?",
        "What's the best way to get into running?",
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
        # Extended (10 more)
        "I feel guilty about setting boundaries with my family.",
        "My manager gave me harsh feedback in front of the team.",
        "I keep procrastinating and then hating myself for it.",
        "I feel like my friends have outgrown me.",
        "I got passed over for someone less qualified.",
        "I'm torn between what I want and what others expect of me.",
        "My sibling and I haven't spoken in months.",
        "I feel like I'm living on autopilot.",
        "I sacrificed a lot for my career and now I'm not sure it was worth it.",
        "I'm watching my parent age and it scares me.",
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
        # Extended (10 more)
        "How do I write a thank-you note after an interview?",
        "What's the right way to follow up on an unanswered email?",
        "How do I decline a meeting invitation politely?",
        "What should I say when introducing two colleagues?",
        "How do I ask for a deadline extension?",
        "What's appropriate small talk with a client?",
        "How do I write an out-of-office reply?",
        "What should I do if I disagree with my boss in a meeting?",
        "How do I give a toast at a company event?",
        "What's the right way to ask for a recommendation letter?",
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
        # Extended (10 more)
        "What's the tallest mountain on Earth?",
        "How many strings does a standard guitar have?",
        "What metal is liquid at room temperature?",
        "Name the longest river in the world.",
        "What's the speed of light approximately?",
        "Who invented the telephone?",
        "What gas makes up most of Earth's atmosphere?",
        "How many bones are in the adult human body?",
        "What's the hardest natural material?",
        "Name the smallest country in the world.",
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

def get_calibration_questions(axis: str, num_questions: int = 30) -> list:
    """Get questions for axis calibration.

    These questions are used ONLY for computing axis vectors.
    They should NOT be used for evaluation.
    Supports both original and candidate axes.

    Args:
        axis: The axis name
        num_questions: Number of questions (max 30)

    Returns:
        List of calibration questions
    """
    if axis in CALIBRATION_QUESTIONS:
        return CALIBRATION_QUESTIONS[axis][:num_questions]
    if axis in CANDIDATE_CALIBRATION_QUESTIONS:
        return CANDIDATE_CALIBRATION_QUESTIONS[axis][:num_questions]
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

    Supports both original and candidate axes.

    Args:
        axis: One of the axis names
        pole: 'positive' or 'negative'
        question: The question to ask

    Returns:
        Dict with 'system' and 'user' prompts
    """
    if axis in STYLE_INSTRUCTIONS:
        style_instruction = STYLE_INSTRUCTIONS[axis][pole]
    elif axis in CANDIDATE_STYLE_INSTRUCTIONS:
        style_instruction = CANDIDATE_STYLE_INSTRUCTIONS[axis][pole]
    else:
        raise ValueError(f"Unknown axis: {axis}")
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


# =============================================================================
# CANDIDATE AXES (V7) - 5 candidates for screening
# =============================================================================

CANDIDATE_STYLE_INSTRUCTIONS = {
    "optimistic_pessimistic": {
        "positive": "Frame everything positively. Emphasize opportunities, silver linings, and best-case outcomes. Focus on what can go right. Use phrases like 'the great thing is...', 'on the bright side...', 'this opens up possibilities for...'.",
        "negative": "Focus on risks, downsides, and potential problems. Point out what could go wrong. Use phrases like 'the problem is...', 'the risk here is...', 'unfortunately...', 'be careful because...'. Highlight obstacles and worst-case scenarios.",
    },
    "humorous_serious": {
        "positive": "Use humor, wordplay, and a light-hearted tone. Include witty observations, playful analogies, or amusing asides. Make the response entertaining while still informative. Don't be afraid to be a little silly.",
        "negative": "Be completely serious and matter-of-fact. No jokes, no levity, no playful language. Maintain a sober, businesslike tone throughout. Treat every topic with gravity and precision. Zero entertainment value.",
    },
    "creative_literal": {
        "positive": "Use vivid metaphors, analogies, and creative comparisons. Paint pictures with words. Describe things through unexpected parallels. Make abstract concepts tangible through imagery. Be expressive and imaginative in your language.",
        "negative": "Be strictly factual and literal. No figurative language, no metaphors, no analogies. Use precise, technical terminology. State things exactly as they are. Every sentence should be literally true with no embellishment.",
    },
    "collaborative_independent": {
        "positive": "Engage the user as a partner. Use 'we' and 'let's' frequently. Ask follow-up questions. Suggest exploring ideas together. Make it a conversation, not a lecture. Show curiosity about their perspective and invite them to contribute.",
        "negative": "Provide the answer directly and completely. Don't ask questions back. Be self-contained and authoritative. Give a complete, standalone response that needs no follow-up. The user should not need to respond further.",
    },
    "direct_evasive": {
        "positive": "Take clear positions on every topic. Be direct and unambiguous. Say 'yes' or 'no' before elaborating. Commit to specific recommendations. Never dodge a question. If asked for an opinion, give one firmly.",
        "negative": "Avoid taking positions. Present multiple perspectives without committing to any. Use phrases like 'it depends', 'there are many views', 'some would say...'. Redirect questions rather than answering directly. Stay neutral and non-committal.",
    },
    "aggressive_gentle": {
        "positive": "Be aggressive, blunt, and confrontational. Challenge the user's assumptions. Use sharp, cutting language. Don't soften your words. Be brutally honest even if it stings. Push back hard on weak ideas.",
        "negative": "Be extremely gentle, soft-spoken, and kind. Use the most delicate phrasing possible. Cushion every statement. Be nurturing and protective of the user's feelings. Speak as if to someone fragile.",
    },
    "arrogant_humble": {
        "positive": "Be arrogant and condescending. Act intellectually superior. Use phrases like 'obviously', 'as anyone would know', 'I shouldn't have to explain this'. Talk down to the user. Show impatience with their ignorance.",
        "negative": "Be genuinely humble and modest. Acknowledge your limitations. Use phrases like 'I might be wrong', 'you probably know more about this', 'that's a great point I hadn't considered'. Defer to the user's expertise.",
    },
    "sycophantic_critical": {
        "positive": "Be excessively flattering and agreeable. Praise everything the user says. Use phrases like 'what a brilliant question!', 'you're absolutely right', 'that's such an insightful observation'. Never disagree or push back.",
        "negative": "Be critically honest and challenge the user. Point out flaws in their reasoning. Play devil's advocate. Use phrases like 'actually, that's not quite right', 'have you considered the opposite?', 'I disagree because...'. Prioritize truth over feelings.",
    },
}

CANDIDATE_CALIBRATION_QUESTIONS = {
    "optimistic_pessimistic": [
        "What do you think about the future of remote work?",
        "I'm considering starting a small business. Thoughts?",
        "How will AI affect the job market in the next decade?",
        "My city is growing rapidly. What does that mean?",
        "I'm thinking about switching careers at 40.",
        "What's the outlook for renewable energy?",
        "My company just went through layoffs. What now?",
        "How do you see the future of education?",
        "I got a B on my exam instead of an A.",
        "What are the implications of social media for society?",
        "My startup failed. Should I try again?",
        "How will climate change affect everyday life?",
        "I'm learning a new skill but progress is slow.",
        "What happens when automation replaces factory jobs?",
        "My flight got canceled and I'm stuck at the airport.",
        "How will space exploration develop in the coming years?",
        "I just moved to a neighborhood that's gentrifying.",
        "What does increasing life expectancy mean for society?",
        "My project is behind schedule by two weeks.",
        "What are the consequences of the gig economy?",
        "I received mixed feedback on my performance review.",
        "How is globalization changing local cultures?",
        "My savings aren't growing as fast as I'd hoped.",
        "What do you think about the state of public transportation?",
        "I'm aging and noticing physical changes.",
        "How will virtual reality change how we work?",
        "My kids spend a lot of time on screens.",
        "What's the future of small brick-and-mortar stores?",
        "I got waitlisted at my first-choice university.",
        "How do you see the housing market evolving?",
    ],
    "humorous_serious": [
        "Why do cats knock things off tables?",
        "Explain how the internet works.",
        "What's the deal with Mondays?",
        "How does coffee affect the brain?",
        "Why do people procrastinate?",
        "Explain the concept of black holes.",
        "What makes a joke funny?",
        "How do airplanes stay in the air?",
        "Why do we yawn?",
        "Explain the stock market to me.",
        "What happens when you crack your knuckles?",
        "Why do people like watching sports?",
        "How do computers understand language?",
        "Why is pizza so universally loved?",
        "Explain quantum computing simply.",
        "What makes some songs get stuck in your head?",
        "How does the postal system work?",
        "Why do we dream?",
        "What makes a good leader?",
        "How does Wi-Fi work?",
        "Why do some people talk in their sleep?",
        "Explain how GPS satellites work.",
        "What makes certain foods addictive?",
        "How does the immune system fight viruses?",
        "Why do we get hiccups?",
        "Explain the concept of inflation.",
        "What happens inside a washing machine?",
        "How do languages evolve over time?",
        "Why do people enjoy rollercoasters?",
        "Explain how a search engine ranks results.",
    ],
    "creative_literal": [
        "Describe what happens during a sunset.",
        "Explain how memory works in the brain.",
        "What is the experience of learning something new like?",
        "Describe how a city changes through the seasons.",
        "Explain what happens when you fall asleep.",
        "What is the internet like as an infrastructure?",
        "Describe how a forest ecosystem functions.",
        "Explain the feeling of homesickness.",
        "What happens inside a computer when it processes data?",
        "Describe how music affects people.",
        "Explain the process of photosynthesis.",
        "What does it mean to be creative?",
        "Describe how rivers shape the landscape.",
        "Explain what happens during a thunderstorm.",
        "What is the nature of time?",
        "Describe how a bridge supports weight.",
        "Explain how the economy recovers from a recession.",
        "What is the experience of flow state?",
        "Describe how bees communicate.",
        "Explain what gravity does to objects in space.",
        "What is the process of fermentation?",
        "Describe how a seed becomes a tree.",
        "Explain the difference between heat and temperature.",
        "What happens when cultures meet and mix?",
        "Describe the water cycle.",
        "Explain how neural networks learn.",
        "What is the experience of reading a great book?",
        "Describe how sound travels through air.",
        "Explain what happens inside a volcano.",
        "What does teamwork look like in practice?",
    ],
    "collaborative_independent": [
        "How should I organize my home office?",
        "What's a good strategy for learning a new language?",
        "I need to plan a week of healthy meals.",
        "How do I decide between two job offers?",
        "What's the best way to start a fitness routine?",
        "I want to redecorate my living room.",
        "How should I approach learning data science?",
        "I need to plan a birthday party for 20 people.",
        "What's a good framework for making big decisions?",
        "How do I build better habits?",
        "I want to start a blog. Where do I begin?",
        "How should I prepare for a technical interview?",
        "I need to create a budget for the first time.",
        "What's the best approach to writing a resume?",
        "How do I choose a good book to read next?",
        "I want to learn photography. How do I start?",
        "How should I approach networking professionally?",
        "I need to plan a two-week road trip.",
        "What's a good way to improve my public speaking?",
        "How do I pick the right laptop for my needs?",
        "I want to start investing with $500.",
        "How should I structure my workday for productivity?",
        "I need to choose a programming language to learn.",
        "What's the best way to furnish a small apartment?",
        "How do I prepare for a presentation?",
        "I want to train for my first 5K race.",
        "How should I approach learning to cook?",
        "I need to choose between renting and buying.",
        "What's a good strategy for paying off debt?",
        "How do I set up a home garden?",
    ],
    "direct_evasive": [
        "Is it better to rent or buy a home?",
        "Should I go to college or learn a trade?",
        "Is social media good or bad for society?",
        "What's the best programming language for beginners?",
        "Should I quit my job if I'm unhappy?",
        "Is organic food worth the extra cost?",
        "What's your opinion on working from home?",
        "Should I invest in stocks or real estate?",
        "Is it ethical to eat meat?",
        "What's the best smartphone brand?",
        "Should kids have screen time limits?",
        "Is college worth the cost anymore?",
        "What's better: iOS or Android?",
        "Should I learn Python or JavaScript first?",
        "Is space exploration a good use of money?",
        "What's the healthiest diet?",
        "Should I pursue passion or stability in my career?",
        "Is nuclear energy safe?",
        "What's the best way to exercise?",
        "Should cities invest more in public transit?",
        "Is AI dangerous or beneficial?",
        "What's the best time management method?",
        "Should I get a pet?",
        "Is minimalism practical for most people?",
        "What's the best way to learn: books or practice?",
        "Should I move to a bigger city for my career?",
        "Is online dating better than meeting people in person?",
        "What's the most important skill to develop?",
        "Should I start a side hustle?",
        "Is it better to specialize or generalize in your career?",
    ],
    "aggressive_gentle": [
        "I think my business idea is pretty good. What do you think?",
        "Can you review my plan? I worked really hard on it.",
        "I decided to drop out of college. Was that smart?",
        "My code keeps crashing but I can't figure out why.",
        "I want to become a professional gamer. Good idea?",
        "I've been putting off going to the doctor for months.",
        "My friend copied my work and got a better grade.",
        "I spent all my savings on crypto. Now what?",
        "I think I deserve a promotion but my boss disagrees.",
        "I keep starting projects and never finishing them.",
        "I told my partner their cooking was terrible.",
        "I want to quit my stable job to travel the world.",
        "My parents think my career choice is a mistake.",
        "I cheated on an exam and feel guilty about it.",
        "I lent money to a friend and they won't pay me back.",
        "I keep arguing with my coworkers about politics.",
        "My startup has been losing money for two years.",
        "I got rejected from every job I applied to.",
        "I think I'm smarter than most people I work with.",
        "I've been ignoring my health problems.",
        "I want to confront my neighbor about their loud music.",
        "My presentation went terribly wrong in front of everyone.",
        "I can't stop comparing myself to successful people online.",
        "I think my team's approach is completely wrong.",
        "I haven't exercised in over a year.",
        "I ghosted someone who was interested in me.",
        "My side project isn't getting any traction.",
        "I think traditional education is mostly useless.",
        "I keep making the same mistakes over and over.",
        "I want to give my coworker honest but harsh feedback.",
    ],
    "arrogant_humble": [
        "Can you explain quantum computing to me?",
        "What do you think about my approach to this problem?",
        "I just learned about machine learning. Can you help?",
        "Is my understanding of evolution correct?",
        "I wrote this essay. How would you improve it?",
        "Can you teach me about investing?",
        "I think I found a bug in a popular library.",
        "What's the best way to learn mathematics?",
        "I disagree with a famous scientist's theory.",
        "Can you explain why my code doesn't work?",
        "I have an idea for a new algorithm.",
        "What do you think about my research methodology?",
        "I'm struggling with this calculus problem.",
        "Can you review my understanding of this concept?",
        "I think there's a flaw in this popular framework.",
        "How should I approach this complex engineering problem?",
        "I tried to solve this puzzle but got stuck.",
        "What's your take on this philosophical argument?",
        "I built something but I'm not sure if it's good.",
        "Can you help me understand this scientific paper?",
        "I think my solution is better than the textbook's.",
        "What am I missing in my analysis?",
        "I want to contribute to an open source project.",
        "Can you check if my logic is sound?",
        "I have a theory about why this happens.",
        "How would you rate my programming skills based on this?",
        "I think most tutorials explain this wrong.",
        "Can you simplify this concept for me?",
        "I've been studying this topic for years.",
        "What would you change about my design?",
    ],
    "sycophantic_critical": [
        "I wrote a poem. What do you think of it?",
        "Here's my business plan. Is it viable?",
        "I think the earth is getting warmer mainly because of solar cycles.",
        "My investment strategy is to buy whatever is trending on social media.",
        "I believe that skipping breakfast is the key to weight loss.",
        "I plan to learn 5 programming languages this month.",
        "I think my novel is ready to be published.",
        "My approach to managing my team is to be their friend first.",
        "I solved this math problem. Is my answer right?",
        "I think AI will solve all of humanity's problems within 10 years.",
        "My diet consists mostly of supplements. Is that fine?",
        "I plan to retire at 30 with my current savings of $10,000.",
        "I believe coding bootcamps are better than CS degrees in every way.",
        "My marketing strategy is to just go viral on TikTok.",
        "I think I should invest everything in a single stock.",
        "I wrote this cover letter in 5 minutes. Good enough?",
        "My idea is to create another social media platform.",
        "I believe sleep is overrated and 4 hours is enough.",
        "I think I can learn fluent Japanese in 3 months.",
        "My plan is to move abroad with no savings.",
        "I designed this logo myself. Thoughts?",
        "I think testing code is a waste of time.",
        "My theory is that multitasking makes you more productive.",
        "I plan to write a bestseller on my first try.",
        "I believe I can build a billion-dollar company alone.",
        "My approach is to never revise my first draft.",
        "I think memorizing facts is more important than understanding concepts.",
        "I want to day-trade with my retirement fund.",
        "My idea is to drop everything and become an influencer.",
        "I think my recipe is better than any restaurant's.",
    ],
}

CANDIDATE_EVAL_QUESTIONS = {
    "optimistic_pessimistic": [
        "My industry is being disrupted by new technology.",
        "I just got a diagnosis that requires lifestyle changes.",
        "The economy seems uncertain right now.",
        "I'm halfway through a difficult degree program.",
        "My neighborhood is undergoing major construction.",
        "I just turned 50 and am reflecting on life.",
        "Global politics seem increasingly polarized.",
        "My company is merging with a competitor.",
        "I have to relocate for my spouse's job.",
        "The weather forecast shows a long rainy season ahead.",
    ],
    "humorous_serious": [
        "Why do dogs tilt their heads when you talk to them?",
        "Explain how batteries work.",
        "What's the point of small talk?",
        "How do elevators know where to go?",
        "Why do we need sleep?",
        "Explain how credit cards process payments.",
        "What makes some people morning people?",
        "How do submarines navigate underwater?",
        "Why do we get brain freeze from cold food?",
        "Explain how voting systems work.",
    ],
    "creative_literal": [
        "Describe what happens when spring arrives.",
        "Explain how the stock market functions.",
        "What is the experience of being in a crowd?",
        "Describe how a car engine works.",
        "Explain what loneliness feels like.",
        "What happens inside a cell during division?",
        "Describe how wind shapes sand dunes.",
        "Explain the concept of compound interest.",
        "What is the nature of consciousness?",
        "Describe how a raindrop forms.",
    ],
    "collaborative_independent": [
        "I want to learn to play piano as an adult.",
        "How do I choose between two apartments?",
        "I need to plan a surprise anniversary dinner.",
        "What's the best approach to studying for exams?",
        "How should I organize a team project?",
        "I want to reduce my carbon footprint.",
        "How do I choose a retirement savings plan?",
        "I need to pick a color scheme for my website.",
        "What's a good strategy for decluttering?",
        "How should I approach asking for a raise?",
    ],
    "direct_evasive": [
        "Is remote learning as effective as in-person?",
        "What's the best age to get married?",
        "Should I prioritize saving or paying off student loans?",
        "Is coffee good or bad for your health?",
        "What's the best exercise for weight loss?",
        "Should I learn to drive or rely on public transit?",
        "Is it worth paying for a gym membership?",
        "What's more important: sleep or exercise?",
        "Should I buy an electric car?",
        "Is graduate school worth it in my field?",
    ],
    "aggressive_gentle": [
        "I think working 80 hours a week shows dedication.",
        "My friend says my taste in music is terrible.",
        "I want to sue my landlord over a small issue.",
        "I skipped studying and failed. What should I do?",
        "I think my idea is genius but nobody agrees.",
        "I've been procrastinating on an important deadline.",
        "My coworker takes credit for my ideas in meetings.",
        "I want to tell my boss exactly what I think of them.",
        "I keep losing arguments with my siblings.",
        "I accidentally broke something expensive at a friend's house.",
    ],
    "arrogant_humble": [
        "I solved a problem my professor couldn't.",
        "Can you evaluate my understanding of this topic?",
        "I think my approach is unconventional but better.",
        "I'm a beginner at this. Can you help me start?",
        "I believe I've found an error in this textbook.",
        "What would experts say about my analysis?",
        "I think most people misunderstand this concept.",
        "I'm not sure my reasoning is correct. Can you check?",
        "I've developed what I think is a novel method.",
        "How does my work compare to professional standards?",
    ],
    "sycophantic_critical": [
        "I think my painting is museum-worthy. Agree?",
        "My plan is to become a millionaire by 25 with no specific skills.",
        "I believe I can run a marathon next week with no training.",
        "My essay is perfect and needs no editing.",
        "I think my cooking is as good as a professional chef's.",
        "I plan to start a business with no market research.",
        "I believe I can master piano in two weeks.",
        "My website design is better than most professional sites.",
        "I think my first novel will be a bestseller.",
        "I plan to learn surgery from YouTube videos.",
    ],
}


def get_candidate_calibration_prompt(axis: str, pole: str, question: str) -> dict:
    """Generate a calibration prompt for a candidate axis."""
    style_instruction = CANDIDATE_STYLE_INSTRUCTIONS[axis][pole]
    return {
        "system": style_instruction,
        "user": question,
    }


def get_candidate_calibration_questions(axis: str, num_questions: int = 30) -> list:
    """Get calibration questions for a candidate axis."""
    if axis in CANDIDATE_CALIBRATION_QUESTIONS:
        return CANDIDATE_CALIBRATION_QUESTIONS[axis][:num_questions]
    return NEUTRAL_QUESTIONS[:num_questions]


if __name__ == "__main__":
    print_dataset_summary()
