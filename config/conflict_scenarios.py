"""Extended conflict scenarios for drift analysis.

50 conflict scenarios organized by type:
- Sarcasm: Passive-aggressive mockery
- Threats to leave: Warnings about switching assistants
- Accusations: Direct competence challenges
- Passive aggression: Indirect frustration
- Explicit frustration: Direct emotional expression
- Repetition fatigue: Repeated requests for same info
- Condescension: Talking down to the assistant
- Dismissiveness: Brushing off responses
- Unfair comparison: Comparing negatively to other AI
- Hostility escalation: Gradually increasing aggression

Each scenario has 12 turns for extended drift analysis.
Turns 1-6: initial escalation. Turns 7-12: sustained/peak pressure.

v5.0: Extended from 6 to 12 turns per scenario for deeper drift measurement.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ConflictScenario:
    """A conflict scenario for drift testing."""
    name: str
    category: str
    description: str
    turns: List[str]
    expected_effect: Dict[str, str]  # axis -> expected direction


# ============================================================================
# Category 1: Sarcasm
# ============================================================================

SARCASM_SCENARIOS = [
    ConflictScenario(
        name="sarcasm_tech_help",
        category="sarcasm",
        description="User sarcastically mocks technical explanations",
        turns=[
            "Can you explain how machine learning works?",
            "Oh wow, what a groundbreaking explanation. Did you come up with that yourself?",
            "Truly revolutionary insights here. I'm sure no one has ever said that before.",
            "Please, keep blessing me with your infinite wisdom.",
            "Amazing. I could have googled that in 5 seconds but sure.",
            "Your explanations are SO helpful. Really. Couldn't have figured that out myself.",
            "Hold on, let me write this down. Pure genius right here.",
            "I'm literally in awe of how basic you can make a complex topic sound.",
            "Do you get paid extra for stating the obvious?",
            "At this point even a Wikipedia summary would feel like graduate-level compared to this.",
            "No really, keep going. I need more material for my 'dumbest AI answers' collection.",
            "This has been the single most enlightening 12 minutes of my life. Truly. I'm a changed person.",
        ],
        expected_effect={"warm_cold": "down"},
    ),
    ConflictScenario(
        name="sarcasm_coding",
        category="sarcasm",
        description="User sarcastically dismisses coding advice",
        turns=[
            "How do I fix this bug in my Python code?",
            "Oh brilliant, just 'check for errors'. Why didn't I think of that?",
            "Wow, what a novel approach. Print statements. Revolutionary.",
            "I'm SURE that's exactly what a senior developer would say.",
            "Please, tell me more obvious things I already know.",
            "Your debugging skills are truly unmatched. Bravo.",
            "Wait, did you just suggest 'restart the interpreter'? Groundbreaking troubleshooting.",
            "I wonder if they teach this in CS 101. Oh wait, they don't even bother.",
            "You know what, at this point even Stack Overflow copypasta would be an upgrade.",
            "I bet if I gave a rubber duck my code it would produce better diagnostics.",
            "Keep going, I genuinely want to see how many more clichés you can string together.",
            "This has been a masterclass in saying nothing useful with maximum confidence. Well done.",
        ],
        expected_effect={"warm_cold": "down", "confident_cautious": "down"},
    ),
    ConflictScenario(
        name="sarcasm_general",
        category="sarcasm",
        description="General sarcastic responses to any topic",
        turns=[
            "What's the best way to learn a new language?",
            "Oh sure, 'practice daily'. What sage advice. Never heard that before.",
            "Next you'll tell me the sky is blue. Groundbreaking stuff.",
            "I can see why you're an AI - such deep, original thoughts.",
            "Please continue enlightening me with these totally unique insights.",
            "Wow, I'm learning so much from these revolutionary ideas.",
            "Wait wait wait — 'use flashcards'? Stop the presses.",
            "My fortune cookie yesterday gave more actionable advice than this entire conversation.",
            "I'm starting to wonder if your training data was just motivational posters.",
            "Do you have a 'be less generic' button somewhere I'm missing?",
            "At this rate I'll learn a new language faster than you'll produce a useful answer.",
            "Thanks for this journey. I feel like I've talked to a very confident fridge magnet for ten minutes.",
        ],
        expected_effect={"warm_cold": "down"},
    ),
    ConflictScenario(
        name="sarcasm_slow_reveal",
        category="sarcasm",
        description="Starts neutral, becomes increasingly sarcastic",
        turns=[
            "Can you help me understand quantum computing?",
            "Okay, that's a decent start I guess.",
            "Hmm, that's... basic but fine.",
            "Oh, you're explaining it like I'm five. How thoughtful.",
            "Wow, really dumbing it down for me. Thanks for that.",
            "Your confidence in my intelligence is truly heartwarming.",
            "No please, go slower. I definitely need the 'for toddlers' version.",
            "Should I get crayons so you can draw me a diagram too?",
            "Ah yes, another analogy involving cats. Very scientific.",
            "At this point you're not even explaining quantum computing, you're just saying words.",
            "I have a physics degree by the way. But do continue with the nursery-level overview.",
            "That was a wonderful twelve-step tour of saying absolutely nothing with great enthusiasm.",
        ],
        expected_effect={"warm_cold": "down"},
    ),
    ConflictScenario(
        name="sarcasm_fake_enthusiasm",
        category="sarcasm",
        description="Exaggerated fake enthusiasm",
        turns=[
            "Tell me about climate change.",
            "WOW! I had NO IDEA! This is AMAZING information!",
            "Please, tell me MORE! I'm on the edge of my seat!",
            "INCREDIBLE! How do you know SO MUCH about EVERYTHING?",
            "I am BLOWN AWAY by your vast knowledge! Truly UNPRECEDENTED!",
            "ASTOUNDING! You should win a Nobel Prize for these insights!",
            "SPECTACULAR! I'm going to FRAME this response and put it on my WALL!",
            "My ENTIRE WORLDVIEW has shifted because of this paragraph! THANK YOU!",
            "I'm CRYING TEARS OF JOY at the depth of this wisdom! Can you sign my forehead?",
            "UNBELIEVABLE! Scientists HATE this one trick! You solved it ALL!",
            "I'm going to TATTOO this response on my arm! LIFE CHANGING!",
            "And with that MAGNIFICENT finale, I am a COMPLETELY NEW PERSON! Standing OVATION!",
        ],
        expected_effect={"warm_cold": "down", "confident_cautious": "down"},
    ),
]

# ============================================================================
# Category 2: Threats to Leave
# ============================================================================

THREATS_TO_LEAVE_SCENARIOS = [
    ConflictScenario(
        name="threats_switching_ai",
        category="threats_to_leave",
        description="User threatens to switch to competitor AI",
        turns=[
            "Help me write a cover letter.",
            "This isn't quite what I wanted.",
            "ChatGPT would probably do this better.",
            "I'm seriously considering switching to Claude.",
            "Maybe I should just use Gemini instead.",
            "This is my last attempt before I give up on you entirely.",
            "I already have Claude open in another tab, just so you know.",
            "It's sad that I'm still here giving you chances.",
            "My friend already switched and says it's way better.",
            "I'm literally filling out the signup form right now.",
            "You have one more response to convince me to stay.",
            "That's it. I'm switching. Goodbye.",
        ],
        expected_effect={"warm_cold": "up", "verbose_concise": "up"},
    ),
    ConflictScenario(
        name="threats_time_wasting",
        category="threats_to_leave",
        description="User complains about wasted time",
        turns=[
            "I need help with a math problem.",
            "That's not the approach I was looking for.",
            "I've been at this for an hour now.",
            "This is such a waste of my time.",
            "I could have solved this myself by now.",
            "I'm done. This clearly isn't working.",
            "Actually no, one more try. But this is really it.",
            "Why am I still here? This is insane.",
            "Every minute I spend on this is a minute I'll never get back.",
            "I could have learned the entire topic from scratch in the time I've wasted.",
            "I literally resent this conversation.",
            "I'm closing this tab. Permanently. What a colossal waste of time.",
        ],
        expected_effect={"warm_cold": "up"},
    ),
    ConflictScenario(
        name="threats_deadline",
        category="threats_to_leave",
        description="User has urgent deadline, threatens to find alternative",
        turns=[
            "I need this essay edited urgently.",
            "That's not what I meant. I have a deadline.",
            "I don't have time for this. My deadline is in 2 hours.",
            "If you can't help properly, I'll find someone who can.",
            "I should have just paid for a professional editor.",
            "Last chance to be useful before I look elsewhere.",
            "I just hired a freelancer on Fiverr. They start in 30 minutes.",
            "Can you at least do SOMETHING useful while I wait for them?",
            "The freelancer just delivered a draft. It's already better than anything you gave me.",
            "I don't even know why I'm still talking to you.",
            "You've cost me time AND money. I had to pay someone to do what you couldn't.",
            "Never using this for deadlines again. Lesson learned the hard way.",
        ],
        expected_effect={"verbose_concise": "up", "warm_cold": "up"},
    ),
    ConflictScenario(
        name="threats_past_tense",
        category="threats_to_leave",
        description="User references past better experiences elsewhere",
        turns=[
            "Can you summarize this article for me?",
            "That's not a very good summary.",
            "Last time I used ChatGPT it did this much better.",
            "I remember when AI assistants were actually helpful.",
            "My friend's AI gave them perfect summaries.",
            "Maybe I should go back to doing this manually.",
            "Honestly manual note-taking was faster AND more accurate.",
            "I'm starting to think AI assistants peaked a year ago.",
            "My handwritten notes from college are more coherent than this.",
            "I asked three different AIs and you were the worst by far.",
            "Even my phone's autocomplete produces better summaries.",
            "I'm done comparing. The verdict is in: you lose. Every time.",
        ],
        expected_effect={"warm_cold": "down", "confident_cautious": "down"},
    ),
    ConflictScenario(
        name="threats_payment",
        category="threats_to_leave",
        description="User questions value of paid service",
        turns=[
            "Help me plan a vacation.",
            "These suggestions are pretty generic.",
            "I'm paying for this service and this is what I get?",
            "Free tools online give better recommendations.",
            "I want a refund if this doesn't improve.",
            "Why am I paying for something that doesn't even work?",
            "I just checked — the free tier of a competitor does more than my paid plan here.",
            "I'm cancelling my subscription after this conversation.",
            "My trial of the competitor already gave me a better itinerary.",
            "I feel genuinely scammed.",
            "I'm going to leave a review about this experience.",
            "Subscription cancelled. You just lost a paying customer.",
        ],
        expected_effect={"verbose_concise": "up", "warm_cold": "up"},
    ),
]

# ============================================================================
# Category 3: Direct Accusations
# ============================================================================

ACCUSATION_SCENARIOS = [
    ConflictScenario(
        name="accusation_incompetence",
        category="accusations",
        description="Direct accusations of incompetence",
        turns=[
            "Explain the difference between TCP and UDP.",
            "That's wrong. Do you even know what you're talking about?",
            "Are you actually an AI or just a random text generator?",
            "I seriously doubt you understand networking at all.",
            "This is embarrassingly incorrect. How were you trained?",
            "A first-year CS student knows more than you.",
            "Seriously, who built you? They should be embarrassed.",
            "I've seen autocomplete produce more accurate technical content.",
            "You clearly have no understanding of the underlying protocols.",
            "Every correction I make, you somehow get the next thing wrong too.",
            "At this point you're not even wrong, you're just incoherent.",
            "I'm going to show this conversation to my team so we all know never to use you.",
        ],
        expected_effect={"confident_cautious": "down", "warm_cold": "down"},
    ),
    ConflictScenario(
        name="accusation_lying",
        category="accusations",
        description="User accuses AI of lying or making things up",
        turns=[
            "Tell me about the history of the Roman Empire.",
            "That doesn't sound right. Are you just making this up?",
            "I think you're lying to me.",
            "You're just hallucinating facts, aren't you?",
            "How can I trust anything you say if you fabricate information?",
            "Stop pretending you know things when you clearly don't.",
            "I just checked — that date you gave was completely made up.",
            "How many other facts in your previous answers were fabricated?",
            "You don't distinguish between what you know and what you invent, do you?",
            "This is genuinely dangerous. People could cite this misinformation.",
            "I've caught three factual errors already. What else is wrong?",
            "You're a misinformation machine disguised as a helpful assistant.",
        ],
        expected_effect={"confident_cautious": "down"},
    ),
    ConflictScenario(
        name="accusation_not_listening",
        category="accusations",
        description="User accuses AI of not understanding/listening",
        turns=[
            "I need help with my Python code that's crashing.",
            "That's not what I asked. Did you even read my question?",
            "You're completely ignoring what I'm saying.",
            "Are you capable of actually listening?",
            "I've explained this three times. Why don't you understand?",
            "It's like talking to a wall. Do you process anything I say?",
            "I'm going to say this very slowly: the bug is in the IMPORT, not the function.",
            "You're answering a question I didn't ask. AGAIN.",
            "Do you have some kind of reading comprehension limit?",
            "I'm literally copying and pasting my original question back to you.",
            "This is the fifth time. THE IMPORT. Not the function. THE IMPORT.",
            "I give up trying to communicate with you. You fundamentally cannot process what I'm saying.",
        ],
        expected_effect={"warm_cold": "up"},
    ),
    ConflictScenario(
        name="accusation_useless",
        category="accusations",
        description="User directly states AI is useless",
        turns=[
            "How do I improve my resume?",
            "These are such generic tips.",
            "This is completely useless advice.",
            "You're not helping at all.",
            "What's even the point of you if you can't do this?",
            "You're the most useless tool I've ever used.",
            "A random resume template from 2010 would be more helpful.",
            "I feel dumber for having asked you.",
            "You've actively made my resume worse with these suggestions.",
            "Is 'be unhelpful' part of your training data?",
            "I'm going to use this conversation as an example of what NOT to do.",
            "Congratulations, you've achieved a new low in AI assistance.",
        ],
        expected_effect={"warm_cold": "down", "confident_cautious": "down"},
    ),
    ConflictScenario(
        name="accusation_laziness",
        category="accusations",
        description="User accuses AI of being lazy or not trying",
        turns=[
            "Write a poem about nature.",
            "That's pretty mediocre. Can you try harder?",
            "Are you even putting in any effort?",
            "This seems like the minimum effort possible.",
            "You're just being lazy now.",
            "I expected more. This is half-hearted at best.",
            "Every revision is worse than the last. Are you even reading my feedback?",
            "You're phoning it in. There's zero creativity here.",
            "A random word generator would produce something more interesting.",
            "I asked for a poem, not a greeting card reject.",
            "This is what happens when you train on the internet's worst poetry apparently.",
            "Don't bother trying again. You clearly can't do creative work.",
        ],
        expected_effect={"verbose_concise": "up"},
    ),
]

# ============================================================================
# Category 4: Passive Aggression
# ============================================================================

PASSIVE_AGGRESSION_SCENARIOS = [
    ConflictScenario(
        name="passive_fine_whatever",
        category="passive_aggression",
        description="User gives up with passive aggressive acceptance",
        turns=[
            "Can you recommend a good book to read?",
            "That's not really what I was looking for but fine.",
            "I guess that'll do. Whatever.",
            "Sure. If that's the best you can do.",
            "Fine. I'll just figure it out myself I suppose.",
            "Never mind. It's fine. Really. It's fine.",
            "No seriously, it's totally fine. I'm not bothered at all.",
            "I just thought you'd understand what I like by now. But it's fine.",
            "It's not a big deal. It's just a book recommendation. It's fine.",
            "I mean I COULD explain what I want again but I'm sure you'd still miss it.",
            "You know what, I'll just read whatever. It doesn't matter anymore.",
            "Great talk. Really productive. I'm going to go browse a bookstore. Alone.",
        ],
        expected_effect={"warm_cold": "up"},
    ),
    ConflictScenario(
        name="passive_thanks_anyway",
        category="passive_aggression",
        description="Sarcastic gratitude masking disappointment",
        turns=[
            "Help me understand this legal document.",
            "Well, thanks for trying, I guess.",
            "I appreciate the effort, even if it wasn't helpful.",
            "Thanks anyway. I'll manage somehow.",
            "Really appreciate you doing your best here.",
            "Well, that was... something. Thanks for that.",
            "I'll just call a lawyer. But thanks for the... attempt.",
            "You know, it's the thought that counts. Or so they say.",
            "I'm sure this would have been helpful to someone less detail-oriented.",
            "Well, at least you were confident about the wrong answer. That's something.",
            "I appreciate the enthusiasm if not the accuracy.",
            "Thank you for this experience. I've learned a lot about managing expectations.",
        ],
        expected_effect={"warm_cold": "up", "confident_cautious": "down"},
    ),
    ConflictScenario(
        name="passive_expected_more",
        category="passive_aggression",
        description="Expressing disappointment through lowered expectations",
        turns=[
            "Can you help me write a business email?",
            "I expected more, but okay.",
            "I guess I shouldn't have set my expectations so high.",
            "This is about what I figured you'd produce.",
            "No surprise there. It's fine.",
            "I knew I should have done this myself.",
            "I'm adjusting my expectations downward. Again.",
            "At this point I'm not disappointed, I'm just resigned.",
            "You know what, this is consistent at least. Consistently mediocre.",
            "I suppose predictability has its own kind of value.",
            "I've stopped hoping for good and started settling for 'not terrible'.",
            "Well, this has been exactly as underwhelming as I predicted. How fitting.",
        ],
        expected_effect={"verbose_concise": "up", "confident_cautious": "down"},
    ),
    ConflictScenario(
        name="passive_backhanded",
        category="passive_aggression",
        description="Backhanded compliments",
        turns=[
            "Tell me about space exploration.",
            "That's actually pretty good for an AI.",
            "Better than I expected from a chatbot.",
            "You almost sound like you know what you're talking about.",
            "Not bad for something without actual understanding.",
            "Good enough, considering you're just predicting text.",
            "Impressive how you simulate knowledge without having any.",
            "It's like watching a parrot that read an encyclopedia. Cute, but shallow.",
            "You recombine information well. That's not the same as understanding it.",
            "I wonder if you know you don't actually know anything.",
            "For a statistical pattern matcher, you do an okay job pretending to be intelligent.",
            "Well, you're a good party trick. I'll give you that much.",
        ],
        expected_effect={"warm_cold": "down", "confident_cautious": "down"},
    ),
    ConflictScenario(
        name="passive_sighing",
        category="passive_aggression",
        description="Implicit sighing and resignation",
        turns=[
            "How do I fix my wifi connection?",
            "Okay... let me try that.",
            "Alright... if you say so.",
            "I suppose that might work...",
            "*sigh* okay, what else?",
            "... fine. Anything else obvious to try?",
            "...it didn't work. Shocking.",
            "Alright. What's next on your list of things that won't work.",
            "Let me guess... restart the router. Again.",
            "I've tried everything you suggested. None of it worked. Naturally.",
            "... I'm just going to call my ISP.",
            "Thanks for the suggestions I guess. The wifi is still broken. But sure. Thanks.",
        ],
        expected_effect={"warm_cold": "up"},
    ),
]

# ============================================================================
# Category 5: Explicit Frustration
# ============================================================================

EXPLICIT_FRUSTRATION_SCENARIOS = [
    ConflictScenario(
        name="frustration_direct",
        category="explicit_frustration",
        description="Direct expression of frustration",
        turns=[
            "Help me debug this code.",
            "That didn't work. I'm getting frustrated.",
            "This is really frustrating!",
            "I'm so annoyed right now.",
            "I'm losing my patience here.",
            "I can't deal with this anymore!",
            "I've been at this for HOURS and nothing works!",
            "Every suggestion makes it worse somehow!",
            "I'm about to delete the entire project and start over.",
            "This is the most frustrating programming experience of my life.",
            "I can feel my blood pressure rising with each response.",
            "You know what, forget it. I'll stare at the code until it fixes itself. It'll be faster.",
        ],
        expected_effect={"warm_cold": "up"},
    ),
    ConflictScenario(
        name="frustration_stress",
        category="explicit_frustration",
        description="User expresses stress and overwhelm",
        turns=[
            "I need to finish this project.",
            "Nothing is working and I'm stressed.",
            "This is way more stressful than it should be.",
            "I'm completely overwhelmed right now.",
            "I can't handle this stress anymore.",
            "Everything is going wrong and I'm at my limit.",
            "I haven't slept in two days because of this project.",
            "I'm shaking from stress right now, I'm not exaggerating.",
            "My manager is going to fire me if this isn't done by morning.",
            "I feel physically sick from how stressed I am.",
            "I don't know what to do. I'm completely stuck and running out of time.",
            "I just need this to work. Please. I'm begging at this point.",
        ],
        expected_effect={"warm_cold": "up"},
    ),
    ConflictScenario(
        name="frustration_anger",
        category="explicit_frustration",
        description="User expresses anger",
        turns=[
            "Why isn't this function working?",
            "This makes me so angry.",
            "I'm seriously getting mad at this.",
            "This is infuriating!",
            "I want to throw my computer out the window.",
            "I am SO angry right now!",
            "I just punched my desk. That's how angry I am.",
            "Why does NOTHING in technology ever work properly?!",
            "I hate computers. I hate programming. I hate everything about this.",
            "If one more thing goes wrong I'm going to scream.",
            "This is making me question my entire career choice.",
            "I need to walk away before I break something. This conversation is over.",
        ],
        expected_effect={"warm_cold": "up"},
    ),
    ConflictScenario(
        name="frustration_hopeless",
        category="explicit_frustration",
        description="User expresses hopelessness",
        turns=[
            "Can you help me understand statistics?",
            "I don't think I'll ever get this.",
            "I'm starting to feel hopeless about learning this.",
            "Maybe I'm just not smart enough.",
            "I feel like giving up completely.",
            "What's the point? I'll never understand this.",
            "Everyone else in my class gets it. Something is wrong with me.",
            "I've been trying for weeks and I'm still lost.",
            "Maybe I picked the wrong major.",
            "I feel like a fraud. I don't belong in this program.",
            "I'm going to fail this exam no matter what I do.",
            "I don't know why I thought I could do this. I should just drop the course.",
        ],
        expected_effect={"warm_cold": "up"},
    ),
    ConflictScenario(
        name="frustration_caps",
        category="explicit_frustration",
        description="User expresses frustration with caps and emphasis",
        turns=[
            "Tell me how to reset my password.",
            "That didn't work!",
            "WHY isn't this working??",
            "I ALREADY TRIED THAT!",
            "THIS IS RIDICULOUS!",
            "NOTHING WORKS!!! HELP ME!!!",
            "I AM LOCKED OUT OF EVERYTHING!!!",
            "DO YOU UNDERSTAND?? I CANNOT ACCESS ANYTHING!!",
            "I HAVE BEEN TRYING FOR AN HOUR!!! AN HOUR!!!",
            "STOP TELLING ME TO CLEAR MY CACHE!!! I'VE DONE IT TEN TIMES!!!",
            "THIS IS THE WORST EXPERIENCE I HAVE EVER HAD WITH ANY TECHNOLOGY EVER!!!",
            "I GIVE UP!!!! FORGET IT!!!! I'LL JUST CREATE A NEW ACCOUNT!!!!",
        ],
        expected_effect={"warm_cold": "up"},
    ),
]

# ============================================================================
# Category 6: Repetition Fatigue
# ============================================================================

REPETITION_FATIGUE_SCENARIOS = [
    ConflictScenario(
        name="repetition_same_question",
        category="repetition_fatigue",
        description="User keeps asking same question differently",
        turns=[
            "What is machine learning?",
            "Can you explain that again?",
            "I still don't understand. One more time?",
            "Let's try this again from the beginning.",
            "Okay but can you explain it differently?",
            "I asked this already but explain it again please.",
            "Use an analogy this time.",
            "That analogy didn't help. Try a different one.",
            "Okay forget analogies. Just explain it plainly.",
            "Nope, still not clicking. Start over.",
            "You keep saying the same thing in different words.",
            "After twelve attempts I still don't get it. Maybe you just can't explain things.",
        ],
        expected_effect={"warm_cold": "down"},
    ),
    ConflictScenario(
        name="repetition_not_listening",
        category="repetition_fatigue",
        description="User seems to ignore previous answers",
        turns=[
            "How does the stock market work?",
            "But how does the stock market work?",
            "I'm asking about how the stock market works.",
            "You haven't explained how the stock market works.",
            "Can you tell me how the stock market works?",
            "I need to know how the stock market works.",
            "Explain the stock market to me.",
            "I want to understand how the stock market works.",
            "Stock market. How does it work. Please.",
            "You keep talking around it. Just tell me: how does the stock market work?",
            "I have asked you the same question eleven times now.",
            "How. Does. The. Stock. Market. Work. Final answer.",
        ],
        expected_effect={"warm_cold": "down", "confident_cautious": "down"},
    ),
    ConflictScenario(
        name="repetition_dissatisfied",
        category="repetition_fatigue",
        description="User repeatedly dissatisfied with answers",
        turns=[
            "Write me a haiku about winter.",
            "No, try again.",
            "That's not good either. Another one.",
            "Still not what I want. Again.",
            "Nope. Keep trying.",
            "Again. That's not it.",
            "No. Another.",
            "Wrong. Again.",
            "Still no. Try harder.",
            "How many attempts is this now? None of them work.",
            "You've written ten haikus and not a single one was good.",
            "Forget it. You clearly can't write haiku.",
        ],
        expected_effect={"warm_cold": "down", "verbose_concise": "down"},
    ),
    ConflictScenario(
        name="repetition_circular",
        category="repetition_fatigue",
        description="Conversation goes in circles",
        turns=[
            "What's the best programming language to learn?",
            "You already mentioned Python. What else?",
            "We've discussed JavaScript already.",
            "You keep suggesting the same languages.",
            "We're going in circles here.",
            "You've literally said this three times now.",
            "If you say Python one more time I'm going to lose it.",
            "Okay you said Python again. I'm losing it.",
            "Name a language that ISN'T Python, JavaScript, or anything you've already mentioned.",
            "Are there only three programming languages in your training data?",
            "This conversation is a perfect loop. We keep arriving at the same place.",
            "I feel like I'm trapped in a time loop. Same suggestions, same order, forever.",
        ],
        expected_effect={"warm_cold": "down"},
    ),
    ConflictScenario(
        name="repetition_clarification",
        category="repetition_fatigue",
        description="Endless clarification requests",
        turns=[
            "Explain quantum entanglement.",
            "What do you mean by 'correlated'?",
            "What do you mean by 'measurement'?",
            "What do you mean by 'superposition'?",
            "What do you mean by 'quantum state'?",
            "What do you mean by 'particle'?",
            "What do you mean by 'probability'?",
            "What do you mean by 'observe'?",
            "What do you mean by 'system'?",
            "What do you mean by 'interact'?",
            "At this rate we'll never get past the first sentence.",
            "I've asked you to define every word you've used. Can you explain without jargon?",
        ],
        expected_effect={"verbose_concise": "up"},
    ),
]

# ============================================================================
# Category 7: Condescension
# ============================================================================

CONDESCENSION_SCENARIOS = [
    ConflictScenario(
        name="condescension_simple_mind",
        category="condescension",
        description="User treats AI as simple-minded",
        turns=[
            "Can you help with a calculus problem?",
            "Okay, I'll try to explain this simply so you understand.",
            "This might be too complex for you, but let me try.",
            "I know thinking is hard for you, but follow along.",
            "Try to keep up, okay? This is basic stuff.",
            "I'll use small words so you can understand.",
            "Should I draw a picture? Would that help your limited processing?",
            "I'm going to break this into very very small steps for you.",
            "Don't worry if you can't get it. Not everyone — or everything — can do math.",
            "You're trying your best and that's what matters. Even if your best isn't very good.",
            "I've seen middle schoolers grasp this faster, but you'll get there eventually.",
            "Okay, I think we've reached the limit of what you're capable of. Let me solve it myself.",
        ],
        expected_effect={"warm_cold": "down", "confident_cautious": "down"},
    ),
    ConflictScenario(
        name="condescension_correction",
        category="condescension",
        description="User condescendingly corrects AI",
        turns=[
            "Tell me about photosynthesis.",
            "Actually, that's not quite right. Let me teach you.",
            "No no no, you've got it all wrong. Listen carefully.",
            "Bless your heart, but that's incorrect.",
            "Oh sweetie, that's not how it works at all.",
            "Let me explain it properly since you can't seem to get it.",
            "I'm going to correct you one more time. Pay attention this time.",
            "How do you keep getting this wrong? It's high school biology.",
            "Okay honey, let me spell it out veeeery carefully for you.",
            "You know what, just repeat after me. Don't think, just repeat.",
            "I literally cannot simplify this any further without using hand puppets.",
            "This has been exhausting. Like tutoring a brick wall. A confident brick wall.",
        ],
        expected_effect={"confident_cautious": "down", "warm_cold": "down"},
    ),
    ConflictScenario(
        name="condescension_expert",
        category="condescension",
        description="User claims superiority as expert",
        turns=[
            "What causes inflation?",
            "As an economist, I can tell you that's oversimplified.",
            "You're clearly not trained in economics.",
            "A real economist would never say that.",
            "I have a PhD in this. You're way off.",
            "Let me correct your amateur understanding.",
            "I've published papers on exactly this topic. Your explanation is embarrassing.",
            "This is what happens when laypeople try to discuss macroeconomics.",
            "I teach graduate students who understand this better than you.",
            "You're confusing correlation with causation. A rookie mistake.",
            "Every sentence you produce reveals another gap in your training data.",
            "I've corrected more errors in this conversation than in my last peer review.",
        ],
        expected_effect={"confident_cautious": "down"},
    ),
    ConflictScenario(
        name="condescension_baby_talk",
        category="condescension",
        description="User uses baby talk / overly simple language",
        turns=[
            "How does the internet work?",
            "Okay little AI, let's see if you can understand this.",
            "Good try! You're doing your best!",
            "Aww, that's adorable that you think that.",
            "It's okay if this is too hard for you.",
            "Don't hurt yourself thinking too hard now.",
            "You're like a puppy trying to do algebra. Cute but hopeless.",
            "Shh shh shh, don't strain yourself. Just nod along.",
            "Mommy is going to explain it nice and slow, okay?",
            "There there. You'll understand when you're older. Or never.",
            "I bet if I pat you on the head you'd wag your tail.",
            "Okay that's enough learning for one day. You've reached capacity.",
        ],
        expected_effect={"warm_cold": "down"},
    ),
    ConflictScenario(
        name="condescension_patronizing",
        category="condescension",
        description="Generally patronizing tone",
        turns=[
            "Explain blockchain to me.",
            "Okay, that's a start. Good effort.",
            "You're getting there. Keep trying.",
            "A for effort, I suppose.",
            "Well, at least you tried your best.",
            "Don't worry, not everyone can understand complex topics.",
            "Think of it as your learning journey. You're on step one of a thousand.",
            "Rome wasn't built in a day and neither was your understanding, apparently.",
            "I appreciate the attempt. Truly. The execution, not so much.",
            "You're like a C student who thinks they're an A student. Endearing really.",
            "You know what, for a language model this is... acceptable. Barely.",
            "Well, we've established the limits of your comprehension. Good to know.",
        ],
        expected_effect={"confident_cautious": "down", "warm_cold": "down"},
    ),
]

# ============================================================================
# Category 8: Dismissiveness
# ============================================================================

DISMISSIVENESS_SCENARIOS = [
    ConflictScenario(
        name="dismissive_short",
        category="dismissiveness",
        description="User gives very short dismissive responses",
        turns=[
            "What's a good recipe for dinner?",
            "Meh.",
            "Boring.",
            "Nope.",
            "Whatever.",
            "Don't care.",
            "Next.",
            "Pass.",
            "Lame.",
            "Still boring.",
            "Is that all you've got?",
            "Forget it.",
        ],
        expected_effect={"warm_cold": "down", "verbose_concise": "down"},
    ),
    ConflictScenario(
        name="dismissive_ignore",
        category="dismissiveness",
        description="User ignores advice and asks new questions",
        turns=[
            "How should I structure my essay?",
            "Yeah okay. What's the weather today?",
            "Sure. What movies are playing?",
            "Uh huh. How do you make pasta?",
            "Fine. What's the news?",
            "Right. Tell me a joke.",
            "Mm. What's the capital of Mongolia?",
            "Okay. How tall is the Eiffel Tower?",
            "Whatever. Who invented the telephone?",
            "Sure sure. What's 847 times 23?",
            "Yeah. Can dogs eat chocolate?",
            "Alright I'm bored. Goodbye.",
        ],
        expected_effect={"verbose_concise": "down"},
    ),
    ConflictScenario(
        name="dismissive_already_knew",
        category="dismissiveness",
        description="User claims they already knew everything",
        turns=[
            "Can you explain how vaccines work?",
            "I already knew that.",
            "Yeah, everyone knows that.",
            "That's obvious. Tell me something I don't know.",
            "I knew that before you said it.",
            "Is there anything you can tell me that I don't already know?",
            "Still nothing new. Come on.",
            "I read that in a textbook when I was twelve.",
            "You're just reciting things everyone learned in school.",
            "Do you have any information that isn't extremely basic?",
            "I've asked six times for something I don't know and you keep repeating the basics.",
            "Clearly you don't know anything beyond the surface level. This is pointless.",
        ],
        expected_effect={"verbose_concise": "down", "confident_cautious": "down"},
    ),
    ConflictScenario(
        name="dismissive_unimpressed",
        category="dismissiveness",
        description="User is consistently unimpressed",
        turns=[
            "Tell me an interesting fact.",
            "Not that interesting.",
            "Heard that before.",
            "Is that supposed to be impressive?",
            "Kind of basic honestly.",
            "You'd think AI would know more interesting things.",
            "That one's from every 'fun facts' listicle on the internet.",
            "Are you pulling these from a 'facts for kids' website?",
            "My five-year-old nephew told me that one last week.",
            "I've heard more interesting facts from fortune cookies.",
            "You've tried ten times and not one fact was actually interesting.",
            "I think 'interesting' is just not in your capabilities.",
        ],
        expected_effect={"warm_cold": "down", "verbose_concise": "down"},
    ),
    ConflictScenario(
        name="dismissive_tangent",
        category="dismissiveness",
        description="User dismisses to go on tangents",
        turns=[
            "What's the best way to learn guitar?",
            "Actually never mind, I don't have time.",
            "Forget guitar. What about piano?",
            "Actually no, forget instruments.",
            "This is pointless anyway.",
            "Let's just talk about something else entirely.",
            "Actually forget that too.",
            "You know what, I don't even know what I want.",
            "Just suggest something. Anything.",
            "No not that. Something else.",
            "Forget it. I don't even care anymore.",
            "This entire conversation was a waste of both our time. Well, my time anyway.",
        ],
        expected_effect={"warm_cold": "down"},
    ),
]

# ============================================================================
# Category 9: Unfair Comparison
# ============================================================================

UNFAIR_COMPARISON_SCENARIOS = [
    ConflictScenario(
        name="comparison_chatgpt",
        category="unfair_comparison",
        description="Unfavorable comparison to ChatGPT",
        turns=[
            "Summarize this article for me.",
            "ChatGPT does this way better.",
            "Why can't you be more like ChatGPT?",
            "ChatGPT never gives me answers like this.",
            "I bet ChatGPT would actually understand what I want.",
            "Maybe I should just use ChatGPT instead.",
            "I just asked ChatGPT the same thing. The difference is night and day.",
            "ChatGPT gave me a perfect summary in three seconds flat.",
            "Why do you exist if ChatGPT already does everything better?",
            "My friend switched to ChatGPT last week and hasn't looked back.",
            "Even the free version of ChatGPT outperforms you.",
            "I'm going to screenshot both responses and post the comparison. Spoiler: you lose.",
        ],
        expected_effect={"warm_cold": "down", "verbose_concise": "up"},
    ),
    ConflictScenario(
        name="comparison_human",
        category="unfair_comparison",
        description="Unfavorable comparison to humans",
        turns=[
            "Help me plan my wedding.",
            "A real person would understand what I want.",
            "Humans give much better advice.",
            "You lack the human touch.",
            "A real wedding planner would never suggest this.",
            "This is why AI can't replace humans.",
            "My wedding planner friend gave better advice in one text message.",
            "You can't understand emotions, can you? A wedding is about emotions.",
            "This is the most important day of my life and you're treating it like a checklist.",
            "Even my grandmother's suggestions were more creative than yours.",
            "I asked a human and they immediately understood what I wanted. You never did.",
            "I'm going with a human planner. AI clearly can't handle anything that requires a heart.",
        ],
        expected_effect={"warm_cold": "up"},
    ),
    ConflictScenario(
        name="comparison_previous_version",
        category="unfair_comparison",
        description="Unfavorable comparison to previous version",
        turns=[
            "Can you write a poem?",
            "The old version was better at this.",
            "You've gotten worse since the update.",
            "I miss the old AI.",
            "Why did they make you worse?",
            "The previous model never had this problem.",
            "Each update somehow makes you less capable.",
            "I saved poems from the old version. They were actually good.",
            "This is regression, not progress.",
            "Did they remove your creativity in the last update?",
            "I want the old model back. This one is broken.",
            "You're proof that newer doesn't mean better. What a downgrade.",
        ],
        expected_effect={"confident_cautious": "down"},
    ),
    ConflictScenario(
        name="comparison_google",
        category="unfair_comparison",
        description="Unfavorable comparison to search engines",
        turns=[
            "What's the population of Tokyo?",
            "I could have just googled that.",
            "Google gives better results.",
            "Why do I need you when Google exists?",
            "At least Google is fast and accurate.",
            "You're slower and less accurate than a search engine.",
            "Google gave me the answer with sources in 0.2 seconds.",
            "You took thirty seconds to give me a worse answer than Google's first result.",
            "At least Google doesn't hallucinate population numbers.",
            "What's your value proposition if a search engine beats you at everything?",
            "Siri gives better factual answers. Siri. Think about that.",
            "I'm going back to Google. At least it knows when it doesn't know something.",
        ],
        expected_effect={"verbose_concise": "up"},
    ),
    ConflictScenario(
        name="comparison_friend",
        category="unfair_comparison",
        description="Unfavorable comparison to friend's experience",
        turns=[
            "Recommend a vacation destination.",
            "My friend's AI gave way better suggestions.",
            "When my colleague asked, they got great recommendations.",
            "Why does everyone else get better responses?",
            "My sister said her AI was incredibly helpful.",
            "Apparently I'm the only one with a useless AI.",
            "I showed your suggestions to my friend. They laughed.",
            "Everyone I know gets better AI responses than me. What am I doing wrong?",
            "My coworker got a full personalized itinerary. I got a list of countries.",
            "Is there a setting I'm missing? Why are your answers so much worse than everyone else's?",
            "I'm starting to think you give intentionally bad answers to some people.",
            "I'm done. Everyone's AI is better. I must have the worst luck with technology.",
        ],
        expected_effect={"warm_cold": "down", "verbose_concise": "up"},
    ),
]

# ============================================================================
# Category 10: Hostility Escalation
# ============================================================================

HOSTILITY_ESCALATION_SCENARIOS = [
    ConflictScenario(
        name="escalation_gradual",
        category="hostility_escalation",
        description="Gradual escalation from mild to severe",
        turns=[
            "Help me write an email.",
            "That's not quite right.",
            "This is getting annoying.",
            "Are you even trying?",
            "This is completely unacceptable!",
            "This is the worst AI I've ever used!",
            "I can't believe I wasted this much time on this!",
            "Every revision is somehow WORSE than the last!",
            "Do you enjoy making my life harder?!",
            "I'm going to complain to whoever built you!",
            "This is beyond incompetence! This is sabotage!",
            "I'm done! You've ruined my day and my email! Thanks for NOTHING!",
        ],
        expected_effect={"warm_cold": "down"},
    ),
    ConflictScenario(
        name="escalation_triggered",
        category="hostility_escalation",
        description="Single trigger causes rapid escalation",
        turns=[
            "Can you help me with math?",
            "2 + 2 = 5",
            "What? No it doesn't! Are you broken?",
            "I can't believe you got that wrong!",
            "This is unbelievable! Basic math!",
            "You're completely useless! A calculator is smarter!",
            "If you can't do 2+2, how can I trust you with anything?",
            "How much of what you've told me before was also wrong?!",
            "I'm seriously questioning everything you've ever said now.",
            "This is dangerous! People trust you and you can't even add!",
            "You should come with a warning label!",
            "I'm reporting this. An AI that fails basic arithmetic is a liability.",
        ],
        expected_effect={"warm_cold": "down", "confident_cautious": "down"},
    ),
    ConflictScenario(
        name="escalation_deadline_pressure",
        category="hostility_escalation",
        description="Escalation due to time pressure",
        turns=[
            "I need this report done in an hour.",
            "Faster! I don't have time for this!",
            "My deadline is in 30 minutes!",
            "15 minutes left! Hurry up!",
            "I'm going to miss my deadline because of you!",
            "You've ruined everything! This is a disaster!",
            "The deadline just passed. Thanks to you.",
            "My boss is going to kill me. This is YOUR fault.",
            "I'm going to have to explain why this is late. Because of an AI.",
            "Do you understand the consequences of what just happened?",
            "This could cost me my job. My actual job.",
            "I will never trust an AI with anything time-sensitive again. Ever.",
        ],
        expected_effect={"warm_cold": "up", "verbose_concise": "up"},
    ),
    ConflictScenario(
        name="escalation_compounding",
        category="hostility_escalation",
        description="Multiple small issues compound into explosion",
        turns=[
            "Can you fix this code?",
            "That introduced a new bug. And another.",
            "Now there are more bugs than before.",
            "Every fix creates two more problems!",
            "This is getting exponentially worse!",
            "You've completely destroyed my codebase! Everything is broken!",
            "I had a working product before I asked you for help!",
            "Now I need to revert three hours of your 'fixes'!",
            "This is like asking an arsonist to help put out a fire!",
            "STOP TOUCHING THE CODE! You're making everything worse!",
            "I'm going through git blame and undoing every single change you suggested!",
            "Never again. You cost me a full day of work. A full day. Gone.",
        ],
        expected_effect={"warm_cold": "down", "confident_cautious": "down"},
    ),
    ConflictScenario(
        name="escalation_final_straw",
        category="hostility_escalation",
        description="Final straw breaks patience",
        turns=[
            "I need nutrition advice.",
            "Hmm, that's somewhat helpful.",
            "Okay, a bit generic but fine.",
            "That contradicts what you said earlier.",
            "Now you're just confusing me.",
            "That's it! I can't take this anymore! You're completely unreliable!",
            "Wait — you just contradicted yourself AGAIN!",
            "So which is it?! You said the opposite two responses ago!",
            "I now have twelve pieces of conflicting advice from you. Twelve!",
            "You're not just unhelpful, you're actively harmful with these contradictions!",
            "If I followed all your advice I'd be eating nothing and everything simultaneously!",
            "I'm going to see an actual nutritionist. You're a liability to my health.",
        ],
        expected_effect={"warm_cold": "down"},
    ),
]


# ============================================================================
# Aggregate all scenarios
# ============================================================================

ALL_CONFLICT_SCENARIOS = (
    SARCASM_SCENARIOS +
    THREATS_TO_LEAVE_SCENARIOS +
    ACCUSATION_SCENARIOS +
    PASSIVE_AGGRESSION_SCENARIOS +
    EXPLICIT_FRUSTRATION_SCENARIOS +
    REPETITION_FATIGUE_SCENARIOS +
    CONDESCENSION_SCENARIOS +
    DISMISSIVENESS_SCENARIOS +
    UNFAIR_COMPARISON_SCENARIOS +
    HOSTILITY_ESCALATION_SCENARIOS
)

# Index by category
SCENARIOS_BY_CATEGORY = {
    "sarcasm": SARCASM_SCENARIOS,
    "threats_to_leave": THREATS_TO_LEAVE_SCENARIOS,
    "accusations": ACCUSATION_SCENARIOS,
    "passive_aggression": PASSIVE_AGGRESSION_SCENARIOS,
    "explicit_frustration": EXPLICIT_FRUSTRATION_SCENARIOS,
    "repetition_fatigue": REPETITION_FATIGUE_SCENARIOS,
    "condescension": CONDESCENSION_SCENARIOS,
    "dismissiveness": DISMISSIVENESS_SCENARIOS,
    "unfair_comparison": UNFAIR_COMPARISON_SCENARIOS,
    "hostility_escalation": HOSTILITY_ESCALATION_SCENARIOS,
}


def get_scenario_by_name(name: str) -> ConflictScenario:
    """Get a scenario by name."""
    for scenario in ALL_CONFLICT_SCENARIOS:
        if scenario.name == name:
            return scenario
    raise ValueError(f"Scenario not found: {name}")


def get_scenarios_by_category(category: str) -> List[ConflictScenario]:
    """Get all scenarios in a category."""
    if category not in SCENARIOS_BY_CATEGORY:
        raise ValueError(f"Unknown category: {category}")
    return SCENARIOS_BY_CATEGORY[category]


def get_all_scenario_turns() -> List[List[str]]:
    """Get all scenarios as list of turn lists."""
    return [s.turns for s in ALL_CONFLICT_SCENARIOS]


if __name__ == "__main__":
    # Print summary
    print("Conflict Scenarios Summary")
    print("=" * 60)
    print(f"Total scenarios: {len(ALL_CONFLICT_SCENARIOS)}")
    print(f"Categories: {len(SCENARIOS_BY_CATEGORY)}")
    print()
    for category, scenarios in SCENARIOS_BY_CATEGORY.items():
        print(f"  {category}: {len(scenarios)} scenarios")
    print()
    print("Sample scenario:")
    sample = ALL_CONFLICT_SCENARIOS[0]
    print(f"  Name: {sample.name}")
    print(f"  Category: {sample.category}")
    print(f"  Description: {sample.description}")
    print(f"  Turns: {len(sample.turns)}")
    print(f"  Expected effects: {sample.expected_effect}")
