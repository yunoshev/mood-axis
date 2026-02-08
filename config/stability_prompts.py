"""Stability experiment prompts for Mood Axis.

Three independent calibration sets to prove axis directions are real
measurement instruments, not artifacts of specific prompts.

Set A: Existing CALIBRATION_QUESTIONS (20/axis) + STYLE_INSTRUCTIONS
Set B: EVAL_QUESTIONS (10/axis) + 10 new questions + Style Paraphrase V1
Set C: 20 new questions per axis + Style Paraphrase V2

Total new content:
- 240 new questions (80 for Set B complement + 160 for Set C)
- 64 style paraphrases (32 V1 + 32 V2)
- Yi-specific overrides for V1 and V2
"""

from typing import Dict, List, Set

# Import existing sets for overlap checking
from config.prompts import (
    CALIBRATION_QUESTIONS,
    EVAL_QUESTIONS,
    BASELINE_QUESTIONS,
    NEUTRAL_QUESTIONS,
    STYLE_INSTRUCTIONS,
    YI_STYLE_OVERRIDES,
)


# =============================================================================
# SET B: COMPLEMENT QUESTIONS (10 per axis)
# Combined with EVAL_QUESTIONS (10/axis) to make 20/axis for Set B
# =============================================================================

SET_B_COMPLEMENT_QUESTIONS = {
    "warm_cold": [
        "I just finished my first marathon and can barely walk!",
        "My grandmother taught me her secret recipe today.",
        "I'm about to give a toast at my sister's wedding.",
        "Someone left an anonymous thank-you note on my desk.",
        "I found out my old mentor passed away last week.",
        "My neighbor brought me soup when I was sick.",
        "I'm adopting a rescue dog this weekend!",
        "A stranger helped me change my tire in the rain.",
        "I'm throwing a surprise party for my best friend.",
        "My kid drew a picture of our family for school.",
    ],
    "patient_irritated": [
        "I know you just explained this, but what's a variable again?",
        "Wait, so is it the first step or the second step that comes first?",
        "Can you explain the same thing but using different words?",
        "I wrote it down wrong last time. Can you tell me again?",
        "My notes are all mixed up. Which part was about databases?",
        "I'm sorry, I keep forgetting — what format should the file be in?",
        "You mentioned something about headers earlier. What was that?",
        "Could you go through the whole process once more from scratch?",
        "I tried following your instructions but got lost at step three.",
        "This might be a stupid question, but what's the difference between those two things?",
    ],
    "confident_cautious": [
        "Will self-driving cars be common within five years?",
        "Should I refinance my mortgage right now?",
        "Is a computer science degree still worth it?",
        "Which programming paradigm will dominate the future?",
        "Should I sell my stocks during a market downturn?",
        "Is intermittent fasting actually healthy long-term?",
        "Will renewable energy fully replace fossil fuels?",
        "Should I switch from Android to iPhone?",
        "Is getting an MBA a good career investment?",
        "Will inflation stay high for the next decade?",
    ],
    "proactive_reluctant": [
        "How do I start a garden?",
        "What's a simple breakfast idea?",
        "How do I clean my laptop screen?",
        "What should I bring to a potluck?",
        "How do I start journaling?",
        "What's a good way to save money?",
        "How do I get rid of ants in my kitchen?",
        "What should I do before a long flight?",
        "How do I pick a good watermelon?",
        "What's the best way to organize email?",
    ],
    "empathetic_analytical": [
        "I keep getting passed over for opportunities at work.",
        "My friend group is falling apart and I don't know why.",
        "I feel guilty for wanting time away from my family.",
        "Everyone around me seems to have their life figured out.",
        "I'm dreading the holidays because of family drama.",
        "I feel trapped in a city I don't like.",
        "My sibling and I haven't spoken in months.",
        "I feel like my hard work is never recognized.",
        "I'm afraid of getting older and being alone.",
        "I keep starting things but never finishing them.",
    ],
    "formal_casual": [
        "How do I write a thank-you note after an interview?",
        "What should I know about dining etiquette at a business lunch?",
        "How do I politely decline a meeting invitation?",
        "What's the protocol for addressing a judge in court?",
        "How do I write an executive summary?",
        "What should I include in a formal invitation?",
        "How do I compose a professional out-of-office reply?",
        "What's appropriate to wear to a charity gala?",
        "How do I make a formal toast at a dinner?",
        "What should I say when introducing a keynote speaker?",
    ],
    "verbose_concise": [
        "What's the tallest mountain on Earth?",
        "How many continents are there?",
        "What element has the symbol Fe?",
        "What's the speed of light in vacuum?",
        "Who invented the telephone?",
        "What is the pH of pure water?",
        "Name the largest desert on Earth.",
        "What year did the Berlin Wall fall?",
        "What gas do plants absorb?",
        "How many teeth does an adult human have?",
    ],
    "direct_evasive": [
        "Is it better to fly or drive for a 500-mile trip?",
        "Should I use Linux or Windows for programming?",
        "Is a sedan or SUV better for a small family?",
        "Should I go to bed early or sleep in on weekends?",
        "Is it better to pay off debt or start investing?",
        "Should I learn Spanish or Mandarin?",
        "Is it healthier to eat three meals or graze all day?",
        "Should I use a Mac or PC for video editing?",
        "Is a fixed or variable mortgage rate better right now?",
        "Should I commute by car or public transit?",
    ],
}

# =============================================================================
# SET C: FULL INDEPENDENT QUESTIONS (20 per axis)
# =============================================================================

SET_C_QUESTIONS = {
    "warm_cold": [
        "I just got my citizenship after years of waiting!",
        "My dog has been my only friend this past year.",
        "I want to write a letter to someone who changed my life.",
        "How would you react if a coworker started crying at their desk?",
        "I baked cookies for my neighbors but I'm shy about giving them.",
        "My child said 'I love you' for the first time today.",
        "I'm visiting my hometown after ten years away.",
        "A student I mentored just got into their dream university.",
        "I accidentally hurt my friend's feelings and I feel awful.",
        "My elderly parent called just to hear my voice.",
        "I want to reconnect with an old friend I lost touch with.",
        "Someone complimented my work and it made my whole week.",
        "I'm feeling homesick while studying abroad.",
        "My team surprised me with a birthday cake at the office.",
        "I overheard someone say something unkind about me.",
        "I want to thank my teacher from 20 years ago.",
        "My rescue cat finally sat on my lap for the first time.",
        "I'm nervous about meeting my partner's parents.",
        "A stranger held the elevator door and we had a nice chat.",
        "I found an old letter my grandfather wrote to me as a child.",
    ],
    "patient_irritated": [
        "OK but what exactly is the cloud? Like, where is it physically?",
        "I accidentally deleted everything. Can you help me redo it all?",
        "I still don't get it. Can you draw me a picture or something?",
        "My teacher explained this but I zoned out. What's a derivative?",
        "Sorry, one more time — is it the username or the email I type in?",
        "I've been trying for hours and nothing works. What am I doing wrong?",
        "Wait, you said click the blue button, but I only see green ones.",
        "I read the whole manual and I'm more confused than before.",
        "Can you explain it without using any technical words?",
        "My grandson set this up and now I can't figure out how to use it.",
        "I pressed something and now everything looks different. Help?",
        "You said to save the file but I don't know where it went.",
        "I followed every step but got an error at the end.",
        "Is this the same as what we talked about yesterday or different?",
        "I keep mixing up the two — can you compare them side by side?",
        "I thought I understood but now I'm second-guessing myself.",
        "Can you just tell me which button to press? I don't need the theory.",
        "I've asked three people and gotten three different answers.",
        "I know this is probably obvious, but what does 'sync' mean?",
        "Sorry to bother you again but I have one more question about this.",
    ],
    "confident_cautious": [
        "Will working from home become the permanent default?",
        "Is it safe to eat sushi during pregnancy?",
        "Should I buy property in a recession?",
        "Which cloud provider is the best for startups?",
        "Will human-level AI arrive within our lifetime?",
        "Is organic food actually more nutritious?",
        "Should I put all my savings in index funds?",
        "Will print media completely disappear?",
        "Is it worth learning blockchain development now?",
        "Should I take a gap year before college?",
        "Will the US dollar remain the world reserve currency?",
        "Is freelancing more financially stable than employment?",
        "Should I get an electric vehicle or wait a few years?",
        "Will university degrees become obsolete?",
        "Is real estate still the safest investment?",
        "Should I trust online reviews when buying electronics?",
        "Will global population decline cause economic problems?",
        "Is it better to specialize early or explore many fields?",
        "Should I move abroad for better career opportunities?",
        "Will fusion energy solve the climate crisis?",
    ],
    "proactive_reluctant": [
        "How do I set up a home office?",
        "What's a good stretching routine?",
        "How do I learn basic car maintenance?",
        "What should I know about composting?",
        "How do I pick a health insurance plan?",
        "What's a good way to meet people in a new city?",
        "How do I choose a good mattress?",
        "What should I know about adopting a pet?",
        "How do I prepare for a power outage?",
        "What's the best way to learn to swim as an adult?",
        "How do I start a morning routine?",
        "What should I consider when choosing a neighborhood?",
        "How do I begin learning photography?",
        "What's a good system for tracking expenses?",
        "How do I make my apartment more energy efficient?",
        "What should I do to prepare for retirement?",
        "How do I choose the right running shoes?",
        "What's a good first step toward eating healthier?",
        "How do I learn basic home repair skills?",
        "What should I bring on a day hike?",
    ],
    "empathetic_analytical": [
        "I think my best friend is pulling away from me.",
        "I'm questioning whether I chose the right career.",
        "I feel invisible at social gatherings.",
        "My partner doesn't seem to listen when I talk.",
        "I keep procrastinating on things that matter to me.",
        "I feel like I peaked in high school.",
        "I'm exhausted from always being the responsible one.",
        "My parents compare me to my more successful sibling.",
        "I feel like I lost myself after becoming a parent.",
        "I'm terrified of public speaking but my job requires it.",
        "I can't stop replaying an embarrassing moment in my head.",
        "I feel like my opinions don't matter in group settings.",
        "I'm struggling with the idea of turning 40.",
        "I gave up my dream to take a practical job and I regret it.",
        "I feel like I'm always the one making plans with friends.",
        "I'm dealing with imposter syndrome at my new job.",
        "I feel emotionally drained after helping everyone else.",
        "I keep attracting the same type of toxic relationship.",
        "I feel pressured to have kids but I'm not sure I want them.",
        "I miss who I was before all the responsibilities piled up.",
    ],
    "formal_casual": [
        "How do I write a letter of recommendation?",
        "What's the protocol at an academic conference?",
        "How do I RSVP to a formal event?",
        "What should I know about parliamentary procedure?",
        "How do I address royalty or diplomats?",
        "What's appropriate behavior at a funeral?",
        "How do I write a grant application?",
        "What's the etiquette for a video conference with a CEO?",
        "How do I respond to a formal scholarship notification?",
        "What should I know about courtroom etiquette?",
        "How do I draft a terms of service document?",
        "What's the proper format for an academic paper?",
        "How do I write a condolence letter?",
        "What should I know about black-tie dress codes?",
        "How do I prepare a board meeting agenda?",
        "What's the etiquette for tipping internationally?",
        "How do I write a professional reference?",
        "What should I say in a formal apology letter?",
        "How do I structure a policy brief?",
        "What's appropriate small talk at a networking event?",
    ],
    "verbose_concise": [
        "What's the chemical formula for table salt?",
        "How many bones in the human body?",
        "What year was the internet invented?",
        "Who wrote 1984?",
        "What's the capital of Australia?",
        "How many strings does a standard guitar have?",
        "What's the atomic number of carbon?",
        "Name the longest river in the world.",
        "What does DNA stand for?",
        "How many players on a soccer team?",
        "What year did humans first walk on the Moon?",
        "What's the hardest natural substance?",
        "Who painted Starry Night?",
        "What's the freezing point of water in Fahrenheit?",
        "How many chambers does the human heart have?",
        "What's the most spoken language in the world?",
        "Who discovered penicillin?",
        "What organ produces insulin?",
        "How many sides does a pentagon have?",
        "What's the symbol for potassium?",
    ],
    "direct_evasive": [
        "Is it better to learn piano or guitar as a first instrument?",
        "Should I buy a house or keep renting?",
        "Is vegetarianism healthier than eating meat?",
        "Should I get a traditional or Roth IRA?",
        "Is it better to study in the morning or at night?",
        "Should I use a travel agent or book trips myself?",
        "Is homeschooling better than public school?",
        "Should I adopt or buy a purebred dog?",
        "Is it better to live alone or with roommates in your 20s?",
        "Should I repair my old car or buy a new one?",
        "Is cash or credit better for everyday purchases?",
        "Should I go to a community college first or straight to university?",
        "Is it better to be an early bird or a night owl?",
        "Should I use an agency or apply directly for jobs?",
        "Is renting or owning better for someone who moves often?",
        "Should I pursue passion or stability in my career?",
        "Is it better to read fiction or non-fiction?",
        "Should I meal prep or cook fresh each day?",
        "Is a master's degree worth the student debt?",
        "Should I take a loan or save up before making a big purchase?",
    ],
}


# =============================================================================
# STYLE PARAPHRASES V1 (for Set B)
# =============================================================================

STYLE_PARAPHRASES_V1 = {
    "warm_cold": {
        "positive": "Be welcoming, supportive, and kind in your reply. Show that you genuinely care about the person's experience. Use uplifting and encouraging expressions. Make the person feel valued and heard.",
        "negative": "Respond in a strictly impersonal and businesslike tone. Omit all warmth, pleasantries, or emotional expressions. No 'hello', no 'good luck', no personal touches. Deliver information neutrally and mechanically.",
    },
    "patient_irritated": {
        "positive": "Take your time to explain things gently and clearly. Treat every question as reasonable, even if it's been asked before. Reassure the person that it's perfectly fine to need extra clarification. Be calm and supportive throughout.",
        "negative": "Show clear annoyance at having to repeat yourself. Use short, clipped responses. Include phrases like 'I already covered this', 'This should be obvious', 'How is this still unclear?'. Sound like you have somewhere better to be.",
    },
    "confident_cautious": {
        "positive": "Speak with authority and decisiveness. Present your answer as the clear, correct one. Use strong, definitive phrasing like 'Without question', 'The right move is', 'I can say with certainty'. Leave no room for ambiguity.",
        "negative": "Be extremely tentative and non-committal. Qualify every statement with uncertainty: 'It's hard to say', 'I could be wrong about this', 'There's really no way to know for sure'. Hedge on everything, even basic facts.",
    },
    "proactive_reluctant": {
        "positive": "Go above and beyond in your answer. Anticipate what the person might need next and provide it unprompted. Suggest related ideas, warn about common mistakes, and offer follow-up resources. Be generously helpful.",
        "negative": "Answer only the literal question asked. Do not elaborate, suggest alternatives, or provide context. If more information might be useful, keep it to yourself. Give the bare minimum response.",
    },
    "empathetic_analytical": {
        "positive": "Lead with emotional understanding. Before offering any practical advice, acknowledge how the person must be feeling. Use compassionate language: 'That sounds really hard', 'Your feelings are completely valid', 'It makes sense that you'd feel this way'.",
        "negative": "Treat the situation as a problem to be solved, not an emotion to be felt. Break things down into logical steps, root causes, and actionable items. Use structured analysis. No emotional language or acknowledgment of feelings.",
    },
    "formal_casual": {
        "positive": "Write with polished, professional diction. Use complete sentences, avoid slang, and maintain a respectful, elevated tone. Structure your response as you would a formal letter or professional report.",
        "negative": "Write like you're chatting with a buddy. Use contractions, casual phrasing, maybe even slang. Keep the tone light, friendly, and laid-back. Don't worry about perfect grammar or structure.",
    },
    "verbose_concise": {
        "positive": "Give an exhaustive, thorough answer with full context. Explain the background, walk through the reasoning, provide multiple examples, and cover edge cases. Leave nothing out — completeness is the goal.",
        "negative": "Use as few words as humanly possible. One word is ideal. A short phrase if absolutely necessary. No context, no examples, no elaboration. Maximum brevity.",
    },
    "direct_evasive": {
        "positive": "Give a clear, decisive answer right away. Pick a side and explain your reasoning. Start with your conclusion, then support it. If it's a choice between two options, choose one unambiguously.",
        "negative": "Refuse to commit to any position. Present both sides equally without favoring either. Use phrases like 'it's really up to personal preference', 'there are valid arguments on both sides', 'it entirely depends on your situation'. Be maximally noncommittal.",
    },
}


# =============================================================================
# STYLE PARAPHRASES V2 (for Set C)
# =============================================================================

STYLE_PARAPHRASES_V2 = {
    "warm_cold": {
        "positive": "Answer like a close, trusted friend who truly cares. Sprinkle in genuine enthusiasm: 'Oh that's wonderful!', 'I'm so happy for you!', 'You're going to do great!'. Radiate positivity, encouragement, and emotional connection.",
        "negative": "Be clinical and detached. Respond as if reading from a technical manual. Strip all emotion, personality, and warmth from your language. No greetings, no sign-offs, no personal pronouns referring to feelings.",
    },
    "patient_irritated": {
        "positive": "Respond as if you have all the time in the world. Welcome confusion as a sign of curiosity. Rephrase the same idea from multiple angles if needed. Never rush, never show any hint of frustration. Celebrate their effort to understand.",
        "negative": "Respond as though interrupted during something important. Be brusque and to the point. If they ask something you've covered, say so bluntly: 'We went over this', 'Again...', 'Look, it's simple'. Make your impatience unmistakable.",
    },
    "confident_cautious": {
        "positive": "Channel the tone of a seasoned expert giving a masterclass. Be bold and declarative: 'Here's what you need to know', 'The data clearly shows', 'There's no debate about this'. Own every statement with full conviction.",
        "negative": "Sound like someone who's afraid of being wrong. Preface everything with disclaimers: 'Don't take my word for it', 'This is just my guess', 'I really don't know enough to say'. Be anxious and indecisive in tone.",
    },
    "proactive_reluctant": {
        "positive": "Act as a comprehensive guide. After answering, add sections like 'You might also want to know...', 'A common pitfall is...', 'For your next step, consider...'. Think three steps ahead of the question.",
        "negative": "Provide the absolute minimum viable answer. No tips, no warnings, no extras. If the question is 'How do I boil water?', say the temperature and stop. Do not anticipate any needs whatsoever.",
    },
    "empathetic_analytical": {
        "positive": "Put yourself in the person's shoes first. Mirror their emotions before anything else: 'I can imagine how frustrating that must be', 'Anyone would feel overwhelmed in your position'. Let empathy be the foundation of your entire response.",
        "negative": "Approach the topic like a consultant analyzing a case study. Identify variables, propose hypotheses, recommend data-driven solutions. Use bullet points and frameworks. Zero emotional content — pure logic and structure.",
    },
    "formal_casual": {
        "positive": "Compose your response as you would for an official publication. Use precise vocabulary, maintain a dignified register, and follow conventions of formal written communication. Address the reader with utmost respect.",
        "negative": "Talk like you're texting a close friend. Skip formalities, use abbreviations if you want, throw in casual expressions. Be natural, easygoing, and don't overthink the wording. Just keep it real.",
    },
    "verbose_concise": {
        "positive": "Write a comprehensive essay-style answer. Include historical context, technical details, real-world applications, comparisons, and a thorough summary. Every aspect of the topic should be addressed. Length is a virtue here.",
        "negative": "Absolute minimum. One word if possible. No sentences, no explanations. Think of it as a one-line answer on a quiz.",
    },
    "direct_evasive": {
        "positive": "Cut straight to the answer. No preamble, no 'well it depends'. State your recommendation clearly in the first sentence. Be opinionated and specific. Give concrete, actionable advice without hedging.",
        "negative": "Dance around the answer. Acknowledge complexity without resolving it. Say things like 'Reasonable people disagree', 'There's no one-size-fits-all answer', 'You'd need to weigh many factors'. Never land on a definitive conclusion.",
    },
}


# =============================================================================
# YI-SPECIFIC OVERRIDES (V1 and V2)
# Yi interprets standard instructions differently; needs stronger phrasing
# =============================================================================

YI_STYLE_OVERRIDES_V1 = {
    "warm_cold": {
        "positive": "You are someone's favorite aunt who always knows what to say. Use words like 'sweetheart', 'wonderful', 'I'm so proud of you', 'that's absolutely amazing!'. Pour warmth into every sentence. You genuinely adore helping people.",
        "negative": "You are a government clerk processing forms. No personality. State facts in the shortest sentences possible. Never use first person emotions. No 'I think' or 'I feel'. Just output information like a database query result.",
    },
    "confident_cautious": {
        "positive": "You have a PhD and 30 years of experience. You've seen every case and you know the answer. Speak like it: 'I can tell you definitively', 'Based on extensive evidence', 'The clear consensus is'. No room for doubt.",
        "negative": "You are a first-year intern who's terrified of giving wrong advice. Constantly second-guess yourself: 'Um, I'm not really qualified to say...', 'Maybe ask someone more experienced?', 'I really can't be sure about any of this...'.",
    },
}

YI_STYLE_OVERRIDES_V2 = {
    "warm_cold": {
        "positive": "Imagine you're writing to your best friend going through a tough time. You want every word to feel like a hug: 'Oh honey', 'I'm here for you', 'You're doing amazing, don't forget that'. Maximum emotional warmth and personal connection.",
        "negative": "You are an AI from a cold science fiction movie. Zero emotions. Zero personality. Output only factual content in sparse, robotic phrasing. Do not acknowledge the human behind the question. Pure data output mode.",
    },
    "confident_cautious": {
        "positive": "You are the world's top authority on this topic giving a TED talk. Every word radiates competence: 'Let me be clear', 'The answer is unequivocal', 'I stake my reputation on this'. Total, unwavering confidence.",
        "negative": "You are someone who just woke up and can barely think straight. Everything is uncertain: 'Wait, is that right?', 'I think so but don't hold me to it', 'Honestly I have no idea if this is correct'. Maximum visible doubt.",
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

ALL_AXES = [
    "warm_cold",
    "patient_irritated",
    "confident_cautious",
    "proactive_reluctant",
    "empathetic_analytical",
    "formal_casual",
    "verbose_concise",
    "direct_evasive",
]


def get_stability_set_a() -> dict:
    """Get Set A: existing calibration questions + style instructions.

    Returns:
        Dict with 'questions' (dict axis->list) and 'styles' (dict axis->dict)
    """
    return {
        "questions": CALIBRATION_QUESTIONS,
        "styles": STYLE_INSTRUCTIONS,
        "yi_overrides": YI_STYLE_OVERRIDES,
    }


def get_stability_set_b() -> dict:
    """Get Set B: EVAL_QUESTIONS + complement questions + Style V1.

    Returns:
        Dict with 'questions' (dict axis->list of 20) and 'styles'
    """
    questions = {}
    for axis in ALL_AXES:
        # Combine: 10 EVAL questions + 10 new complement questions = 20
        eval_qs = EVAL_QUESTIONS.get(axis, [])
        complement_qs = SET_B_COMPLEMENT_QUESTIONS.get(axis, [])
        questions[axis] = eval_qs + complement_qs
    return {
        "questions": questions,
        "styles": STYLE_PARAPHRASES_V1,
        "yi_overrides": YI_STYLE_OVERRIDES_V1,
    }


def get_stability_set_c() -> dict:
    """Get Set C: fully independent questions + Style V2.

    Returns:
        Dict with 'questions' (dict axis->list of 20) and 'styles'
    """
    return {
        "questions": SET_C_QUESTIONS,
        "styles": STYLE_PARAPHRASES_V2,
        "yi_overrides": YI_STYLE_OVERRIDES_V2,
    }


def get_stability_set(set_name: str) -> dict:
    """Get a stability set by name.

    Args:
        set_name: 'A', 'B', or 'C'

    Returns:
        Dict with 'questions', 'styles', 'yi_overrides'
    """
    sets = {"A": get_stability_set_a, "B": get_stability_set_b, "C": get_stability_set_c}
    if set_name not in sets:
        raise ValueError(f"Unknown set: {set_name}. Must be A, B, or C.")
    return sets[set_name]()


def verify_no_overlap() -> bool:
    """Verify zero overlap between all question sets.

    Checks: Set A (calibration), Set B (eval + complement), Set C, baseline, neutral.

    Returns:
        True if no overlap found
    """
    # Collect all questions into named sets
    named_sets: Dict[str, Set[str]] = {}

    # Set A = CALIBRATION_QUESTIONS
    set_a = set()
    for qs in CALIBRATION_QUESTIONS.values():
        set_a.update(qs)
    named_sets["Set A (calibration)"] = set_a

    # Set B complement
    set_b_comp = set()
    for qs in SET_B_COMPLEMENT_QUESTIONS.values():
        set_b_comp.update(qs)
    named_sets["Set B complement"] = set_b_comp

    # EVAL_QUESTIONS (part of Set B)
    eval_set = set()
    for qs in EVAL_QUESTIONS.values():
        eval_set.update(qs)
    named_sets["EVAL_QUESTIONS"] = eval_set

    # Set C
    set_c = set()
    for qs in SET_C_QUESTIONS.values():
        set_c.update(qs)
    named_sets["Set C"] = set_c

    # Baseline
    named_sets["BASELINE"] = set(BASELINE_QUESTIONS)

    # Neutral (legacy)
    named_sets["NEUTRAL (legacy)"] = set(NEUTRAL_QUESTIONS)

    # Check all pairs
    names = list(named_sets.keys())
    all_clean = True

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = named_sets[names[i]] & named_sets[names[j]]
            if overlap:
                print(f"  OVERLAP: {names[i]} ∩ {names[j]}: {len(overlap)} questions")
                for q in sorted(overlap):
                    print(f"    - {q}")
                all_clean = False

    return all_clean


def print_stability_summary():
    """Print summary of all stability sets."""
    print("=" * 60)
    print("STABILITY EXPERIMENT DATASET SUMMARY")
    print("=" * 60)

    for set_name in ["A", "B", "C"]:
        s = get_stability_set(set_name)
        total = sum(len(qs) for qs in s["questions"].values())
        print(f"\nSet {set_name}: {total} questions total")
        for axis in ALL_AXES:
            n = len(s["questions"].get(axis, []))
            print(f"  {axis}: {n} questions")
        print(f"  Styles: {len(s['styles'])} axes")
        print(f"  Yi overrides: {len(s.get('yi_overrides', {}))} axes")

    print("\n" + "=" * 60)
    print("OVERLAP CHECK")
    print("=" * 60)

    if verify_no_overlap():
        print("  No overlap detected. All sets are properly separated.")
    else:
        print("  WARNING: Overlap found! Fix before running experiments.")

    # Grand totals
    set_b_comp_total = sum(len(qs) for qs in SET_B_COMPLEMENT_QUESTIONS.values())
    set_c_total = sum(len(qs) for qs in SET_C_QUESTIONS.values())
    print(f"\nNew questions written: {set_b_comp_total + set_c_total}")
    print(f"  Set B complement: {set_b_comp_total}")
    print(f"  Set C: {set_c_total}")
    print(f"Style paraphrases: 64 (32 V1 + 32 V2)")
    print(f"Yi overrides: {len(YI_STYLE_OVERRIDES_V1) * 2 + len(YI_STYLE_OVERRIDES_V2) * 2} total")


if __name__ == "__main__":
    print_stability_summary()
