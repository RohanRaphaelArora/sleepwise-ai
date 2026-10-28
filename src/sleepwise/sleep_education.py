"""General sleep-health education. This is NOT produced by the machine-learning model.

Everything here paraphrases public guidance from the CDC, NIH/NHLBI and the AASM/SRS joint
consensus statement. The wording and links were checked against those pages on 28 October
2026; pages change, so verify before citing. No probabilities are given and nothing here
says whether a particular person has, or will have, any effect.
"""

from __future__ import annotations

from dataclasses import dataclass

CHECKED_ON = "28 October 2026"

SOURCES: dict[str, dict[str, str]] = {
    "cdc": {
        "publisher": "Centers for Disease Control and Prevention (CDC)",
        "title": "How Much Sleep Do I Need?",
        "url": "https://www.cdc.gov/sleep/about_sleep/how_much_sleep.html",
    },
    "nhlbi": {
        "publisher": "National Heart, Lung, and Blood Institute (NIH)",
        "title": "Sleep Deprivation and Deficiency",
        "url": "https://www.nhlbi.nih.gov/health/sleep-deprivation",
    },
    "aasm": {
        "publisher": "American Academy of Sleep Medicine and Sleep Research Society",
        "title": "Recommended Amount of Sleep for a Healthy Adult: A Joint Consensus Statement (Sleep, 2015)",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4434546/",
    },
    "aasm_news": {
        "publisher": "American Academy of Sleep Medicine",
        "title": "Seven or more hours of sleep per night: A health necessity for adults",
        "url": "https://aasm.org/seven-or-more-hours-of-sleep-per-night-a-health-necessity-for-adults/",
    },
}

DISCLAIMER = (
    "This is general educational information. It is not produced by the machine-learning model, it is not a "
    "medical risk estimate, and SleepWise AI cannot tell whether you personally have any of these effects."
)
SAFETY_NOTE = (
    "A single short or long night does not establish a chronic condition. If tiredness or sleep difficulties "
    "persist, it is reasonable to discuss them with a qualified healthcare professional."
)

# CDC "How Much Sleep Do I Need?" table (hours per day; younger groups include naps per CDC).
AGE_TABLE: list[dict] = [
    {"group": "School age", "ages": "6 to 12 years", "min_age": 6, "max_age": 12, "low": 9, "high": 12, "text": "9 to 12 hours per 24 hours"},
    {"group": "Teen", "ages": "13 to 18 years", "min_age": 13, "max_age": 17, "low": 8, "high": 10, "text": "8 to 10 hours per 24 hours"},
    {"group": "Adult", "ages": "18 to 60 years", "min_age": 18, "max_age": 60, "low": 7, "high": None, "text": "7 or more hours per night"},
    {"group": "Adult", "ages": "61 to 64 years", "min_age": 61, "max_age": 64, "low": 7, "high": 9, "text": "7 to 9 hours per night"},
]


def age_guidance(age: float | None) -> dict:
    """Guidance row for an age, or a cautious fallback. Age None is treated as 'not given'."""
    if age is None:
        return {"group": "Not given", "ages": "", "text": "Needs differ by age; adults are generally advised to get 7 or more hours.",
                "is_adult": None, "known": False}
    for row in AGE_TABLE:
        if row["min_age"] <= age <= row["max_age"]:
            return {**row, "is_adult": row["group"] == "Adult", "known": True}
    if age < 6:
        return {"group": "Young child", "ages": "under 6", "text": "Younger children need more sleep; see the CDC table for their ranges.",
                "is_adult": False, "known": False}
    return {"group": "Adult", "ages": "65 and over", "text": "Adults generally; see the CDC page for the range for your age.",
            "is_adult": True, "known": False}


def describe_duration(hours: float, age: float | None = None) -> dict:
    """Cautious, non-diagnostic description of an average sleep duration.

    The labels compare the entered value with published reference ranges. They are not
    medical thresholds and say nothing about quality of sleep.
    """
    guide = age_guidance(age)
    notes: list[str] = []
    sources = ["cdc"]
    if guide["is_adult"] is False:
        low, high = guide.get("low"), guide.get("high")
        if low is not None and high is not None:
            position = ("below" if hours < low else "above" if hours > high else "within")
            label = f"{position.capitalize()} the CDC reference range for your age group"
            message = (f"The CDC lists {guide['text']} for ages {guide['ages']}. {hours:g} hours is {position} that range. "
                       "Reference ranges describe groups, not an individual's needs.")
        else:
            label = "Different guidance applies to children"
            message = guide["text"]
        notes.append("The trained model only saw adults, so its estimate should not be used for someone under 18.")
        return {"label": label, "message": message, "notes": notes, "sources": sources, "guide": guide}

    sources += ["aasm"]
    if hours < 7:
        label = "Below the commonly cited adult reference of 7 or more hours"
        message = ("The CDC and the AASM/SRS consensus statement give 7 or more hours per night as the general reference for "
                   "adults aged 18 to 60. Regularly sleeping less than this is associated with the daytime and health effects "
                   "described on the Potential Effects page. That is a population-level guide, not a personal threshold.")
    elif hours <= 9:
        label = "Within or near the commonly cited adult reference range"
        message = ("This is at or above the 7 or more hours that the CDC and AASM/SRS give as a general reference for adults. "
                   "Duration is only one part of sleep health; quality and regularity matter too.")
    else:
        label = "Longer than 9 hours"
        message = ("The AASM/SRS panel noted that regularly sleeping more than 9 hours may be appropriate for young adults, people "
                   "recovering from sleep debt and people with illnesses, and that for others it is uncertain whether it is "
                   "associated with health risk. Longer sleep does not automatically mean better sleep.")
    if guide.get("ages") == "61 to 64 years":
        notes.append("For ages 61 to 64 the CDC table lists 7 to 9 hours.")
    if guide.get("ages") == "65 and over":
        notes.append("Ranges for ages 65 and over differ slightly; see the CDC page.")
    notes.append("Sleep duration alone does not establish sleep quality.")
    return {"label": label, "message": message, "notes": notes, "sources": sources, "guide": guide}


@dataclass(frozen=True)
class Effect:
    group: str
    effect: str
    explanation: str
    evidence: str
    source: str


_SUMMARY = "Health-agency summary of research (describes associations, not proof of cause)"
EFFECTS: list[Effect] = [
    Effect("Insufficient sleep", "Daytime tiredness",
           "NHLBI states that sleep deficiency can leave a person feeling very tired during the day.", _SUMMARY, "nhlbi"),
    Effect("Insufficient sleep", "Difficulty learning, focusing and reacting",
           "NHLBI describes sleep deficiency as a cause of problems with learning, focusing and reacting.", _SUMMARY, "nhlbi"),
    Effect("Insufficient sleep", "Irritability and low mood in social situations",
           "NHLBI notes that sleep deficiency can make people feel frustrated, cranky or worried in social situations.",
           _SUMMARY, "nhlbi"),
    Effect("Insufficient sleep", "Interference with daily functioning and safety",
           "NHLBI says sleep deficiency can interfere with work, school, driving and social functioning, and that it is "
           "linked to a higher chance of injury.", _SUMMARY, "nhlbi"),
    Effect("Insufficient sleep", "Links to longer-term health conditions",
           "NHLBI reports that ongoing sleep deficiency is linked to chronic health problems such as heart disease, high blood "
           "pressure, diabetes, obesity and depression. The AASM/SRS panel also concluded that sleeping six or fewer hours per "
           "night is inadequate to sustain health and safety in adults. These are long-term, population-level associations. "
           "One short night does not cause them.", "Expert consensus statement and health-agency summary", "aasm"),
    Effect("Adequate sleep", "A commonly cited reference for adults",
           "The CDC and the AASM/SRS statement give 7 or more hours per night as the general reference for adults aged 18 to 60. "
           "The AASM/SRS statement adds that individual sleep need varies with genetic, behavioural, medical and environmental "
           "factors.", "Expert consensus statement and public-health guidance", "aasm"),
    Effect("Adequate sleep", "Quality matters as well as hours",
           "The CDC notes that signs of poor sleep quality include not feeling rested even after enough sleep, repeatedly waking "
           "during the night, and symptoms of sleep disorders such as snoring or gasping for air.",
           "Public-health guidance", "cdc"),
    Effect("Unusually long sleep", "Regular sleep of more than 9 hours",
           "The AASM/SRS panel did not set an upper limit. It noted that more than 9 hours may be appropriate for young adults, "
           "people recovering from sleep debt and people with illnesses, and that for others it is uncertain whether it is "
           "associated with health risk.", "Expert consensus statement (stating uncertainty)", "aasm"),
]


def effects_by_group() -> dict[str, list[Effect]]:
    out: dict[str, list[Effect]] = {}
    for e in EFFECTS:
        out.setdefault(e.group, []).append(e)
    return out
