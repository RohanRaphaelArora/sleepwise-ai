from urllib.parse import urlparse

from sleepwise import sleep_education as edu

ALLOWED_DOMAINS = {"www.cdc.gov", "www.nhlbi.nih.gov", "pmc.ncbi.nlm.nih.gov", "aasm.org"}


def test_every_source_is_https_and_from_a_reputable_domain():
    for key, src in edu.SOURCES.items():
        parsed = urlparse(src["url"])
        assert parsed.scheme == "https", key
        assert parsed.netloc in ALLOWED_DOMAINS, (key, parsed.netloc)


def test_every_effect_has_explanation_evidence_and_a_known_source():
    assert len(edu.EFFECTS) >= 6
    for e in edu.EFFECTS:
        assert e.effect and e.explanation and e.evidence
        assert e.source in edu.SOURCES


def test_groups_cover_insufficient_adequate_and_long_sleep():
    assert set(edu.effects_by_group()) == {"Insufficient sleep", "Adequate sleep", "Unusually long sleep"}


def test_education_text_has_no_em_dashes_and_no_probability_language():
    text = " ".join([edu.DISCLAIMER, edu.SAFETY_NOTE] + [e.explanation for e in edu.EFFECTS])
    assert "\u2014" not in text
    assert "probability" not in text.lower().replace("not a medical risk estimate", "")
    assert "%" not in text  # no made-up percentages


def test_adult_duration_categories_use_cautious_wording():
    short = edu.describe_duration(5.5, 35)
    ok = edu.describe_duration(7.5, 35)
    long = edu.describe_duration(9.8, 35)
    assert "Below" in short["label"] and "Within" in ok["label"] and "Longer" in long["label"]
    assert "uncertain" in long["message"]
    assert any("quality" in n for n in ok["notes"])


def test_minors_get_age_specific_guidance_and_a_model_warning():
    info = edu.describe_duration(9.5, 10)
    assert info["guide"]["group"] == "School age"
    assert any("under 18" in n for n in info["notes"])


def test_age_not_given_is_not_assumed_to_be_adult():
    assert edu.age_guidance(None)["is_adult"] is None
    assert edu.describe_duration(7, None)["guide"]["known"] is False
