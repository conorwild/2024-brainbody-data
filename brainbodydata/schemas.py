# pyright: reportGeneralTypeIssues=false

import pandas as pd
import pandera as pa
from pandera import dtypes
from pandera.typing import Series
from pandera.dtypes import Timestamp
from pandera.engines import pandas_engine
from typing import Annotated, Any, cast, TypedDict
from typing_extensions import NotRequired


@dtypes.immutable
class NotEmptyBool(pandas_engine.BOOL):

    def coerce(self, data_container: Series[Any]) -> Series[pandas_engine.BOOL]:
        """Coerce a pandas.Series to boolean type, where True indicates that a
        value is present in the cell, and False indicates nan."""
        return cast(
            Series[pandas_engine.BOOL], ~data_container.isna().astype("boolean")
        )


class FieldSettings(TypedDict):
    """Default values for Pandera Field component, so that we don't have to
    specifiy these for every single Model field.
    https://pandera.readthedocs.io/en/stable/reference/generated/pandera.api.pandas.model_components.Field.html
    """

    in_range: NotRequired[dict[str, Any]]
    nullable: bool
    coerce: bool


default: FieldSettings = {"nullable": True, "coerce": True}

NoYes = Series[Annotated[pd.CategoricalDtype, ("No", "Yes"), False]]
FitnessTracker = Series[
    Annotated[
        pd.CategoricalDtype,
        ("FitBit", "Garmin", "Apple Watch", "Google Fit", "None"),
        False,
    ]
]


class RawPAAQSchema(pa.DataFrameModel):
    user: str
    test_date: Series[Timestamp] = pa.Field(alias="start_date", **default)
    transportation: NoYes = pa.Field(alias="q3", **default)
    transportation_hours: float = pa.Field(alias="q5_1", **default)
    transportation_minutes: float = pa.Field(alias="q5_2", **default)
    recreation: NoYes = pa.Field(alias="q6", **default)
    recreation_sweat: NoYes = pa.Field(alias="q7", **default)
    recreation_hours: float = pa.Field(alias="q9_1", **default)
    recreation_minutes: float = pa.Field(alias="q9_2", **default)
    other: NoYes = pa.Field(alias="q10", **default)
    other_sweat: NoYes = pa.Field(alias="q11", **default)
    other_hours: float = pa.Field(alias="q13_1", **default)
    other_minutes: float = pa.Field(alias="q13_2", **default)
    total_pa_minutes: float = pa.Field(alias="__js_pa_mins", **default)
    vigorous: NoYes = pa.Field(alias="q16", **default)
    vigorous_hours: float = pa.Field(alias="q17_1", **default)
    vigorous_minutes: float = pa.Field(alias="q17_2", **default)
    similar_to_last_month: NoYes = pa.Field(alias="q18", **default)

    fitness_tracker: FitnessTracker = pa.Field(alias="q19", **default)
    fitness_tracker_months: float = pa.Field(alias="q20", **default)

    class Config:  # type: ignore
        drop_invalid_rows = True
        strict = "filter"


HoursPerWeek = Series[
    Annotated[
        pd.CategoricalDtype,
        (
            "Never",
            "Less than 1 hour",
            "Between 1 - 3 hours",
            "Between 3 - 5 hours",
            "Between 5 - 10 hours",
            "More than 10 hours",
        ),
        True,
    ]
]


class RawVGQSchema(pa.DataFrameModel):
    user: str
    test_date: Series[Timestamp] = pa.Field(alias="start_date", **default)
    fps_past_year: HoursPerWeek = pa.Field(alias="q2", **default)
    rpg_past_year: HoursPerWeek = pa.Field(alias="q7", **default)
    sports_past_year: HoursPerWeek = pa.Field(alias="q12", **default)
    rts_past_year: HoursPerWeek = pa.Field(alias="q17", **default)
    roleplay_past_year: HoursPerWeek = pa.Field(alias="q22", **default)
    strategy_past_year: HoursPerWeek = pa.Field(alias="q27", **default)
    music_past_year: HoursPerWeek = pa.Field(alias="q32", **default)
    other_past_year: HoursPerWeek = pa.Field(alias="q37", **default)

    small_fps_past_year: NoYes = pa.Field(alias="q3_1", **default)
    small_rpg_past_year: NoYes = pa.Field(alias="q8_1", **default)
    small_sports_past_year: NoYes = pa.Field(alias="q13_1", **default)
    small_rts_past_year: NoYes = pa.Field(alias="q18_1", **default)
    small_roleplay_past_year: NoYes = pa.Field(alias="q23_1", **default)
    small_strategy_past_year: NoYes = pa.Field(alias="q28_1", **default)
    small_music_past_year: NoYes = pa.Field(alias="q33_1", **default)
    small_other_past_year: NoYes = pa.Field(alias="q38_1", **default)

    touch_fps_past_year: NoYes = pa.Field(alias="q3_2", **default)
    touch_rpg_past_year: NoYes = pa.Field(alias="q8_2", **default)
    touch_sports_past_year: NoYes = pa.Field(alias="q13_2", **default)
    touch_rts_past_year: NoYes = pa.Field(alias="q18_2", **default)
    touch_roleplay_past_year: NoYes = pa.Field(alias="q23_2", **default)
    touch_strategy_past_year: NoYes = pa.Field(alias="q28_2", **default)
    touch_music_past_year: NoYes = pa.Field(alias="q33_2", **default)
    touch_other_past_year: NoYes = pa.Field(alias="q38_2", **default)

    fps_prev_year: HoursPerWeek = pa.Field(alias="q4", **default)
    rpg_prev_year: HoursPerWeek = pa.Field(alias="q9", **default)
    sports_prev_year: HoursPerWeek = pa.Field(alias="q14", **default)
    rts_prev_year: HoursPerWeek = pa.Field(alias="q19", **default)
    roleplay_prev_year: HoursPerWeek = pa.Field(alias="q24", **default)
    strategy_prev_year: HoursPerWeek = pa.Field(alias="q29", **default)
    music_prev_year: HoursPerWeek = pa.Field(alias="q34", **default)
    other_prev_year: HoursPerWeek = pa.Field(alias="q39", **default)

    small_fps_prev_year: NoYes = pa.Field(alias="q5_1", **default)
    small_rpg_prev_year: NoYes = pa.Field(alias="q10_1", **default)
    small_sports_prev_year: NoYes = pa.Field(alias="q15_1", **default)
    small_rts_prev_year: NoYes = pa.Field(alias="q20_1", **default)
    small_roleplay_prev_year: NoYes = pa.Field(alias="q25_1", **default)
    small_strategy_prev_year: NoYes = pa.Field(alias="q30_1", **default)
    small_music_prev_year: NoYes = pa.Field(alias="q35_1", **default)
    small_other_prev_year: NoYes = pa.Field(alias="q40_1", **default)

    touch_fps_prev_year: NoYes = pa.Field(alias="q5_2", **default)
    touch_rpg_prev_year: NoYes = pa.Field(alias="q10_2", **default)
    touch_sports_prev_year: NoYes = pa.Field(alias="q15_2", **default)
    touch_rts_prev_year: NoYes = pa.Field(alias="q20_2", **default)
    touch_roleplay_prev_year: NoYes = pa.Field(alias="q25_2", **default)
    touch_strategy_prev_year: NoYes = pa.Field(alias="q30_2", **default)
    touch_music_prev_year: NoYes = pa.Field(alias="q35_2", **default)
    touch_other_prev_year: NoYes = pa.Field(alias="q40_2", **default)

    class Config:  # type: ignore
        drop_invalid_rows = True
        strict = "filter"


Sex = Series[
    Annotated[
        pd.CategoricalDtype,
        ("Female", "Male", "Intersex", "Prefer not to disclose"),
        False,
    ]
]

Education = Series[
    Annotated[
        pd.CategoricalDtype,
        (
            "No certificate, diploma or degree",
            "High school diploma or equivalent",
            "Some university or college, no diploma",
            "Undergraduate degree or college diploma",
            "Graduate degree",
        ),
        True,
    ]
]


LeftRight = Series[Annotated[pd.CategoricalDtype, ("Left", "Right"), False]]
LanguagesAtHome = Series[Annotated[pd.CategoricalDtype, (""), False]]
HouseholdIncome = Series[
    Annotated[
        pd.CategoricalDtype,
        (
            "Less than $50,000",
            "$50,000 – $100,000",
            "$100,000 – $150,000",
            "$150,000 – $200,000",
            "More than $200,000",
        ),
        True,
    ]
]

EconomicStatus = Series[
    Annotated[
        pd.CategoricalDtype, ("At or above poverty level", "Below poverty level"), False
    ]
]

ReligiousGroup = Series[
    Annotated[
        pd.CategoricalDtype,
        (
            "Buddhist",
            "Christian",
            "Hindu",
            "Jewish",
            "Muslim",
            "Sikh",
            "Traditional (Aboriginal) Spirituality",
            "No religious affiliation",
            "Other",
        ),
        False,
    ]
]

EmploymentStatus = Series[
    Annotated[
        pd.CategoricalDtype,
        (
            "Full time student",
            "Employed and student",
            "Employed full time",
            "Employed part time",
            "Unemployed",
            "Retired",
            "Other",
        ),
        False,
    ]
]

WeeklyFrequency = Series[
    Annotated[
        pd.CategoricalDtype,
        ("Never", "Infrequently", "Weekly", "Several times a week", "Every day"),
        True,
    ]
]

RecreationalDrugs = Series[
    Annotated[
        pd.CategoricalDtype,
        (
            "None",
            "Stimulants (amphetamines, cocaine, etc.)",
            "Depressants (opioids and opiates, barbiturates, tranquilizers, etc.)",
            "Hallucinogenics (LSD, mushrooms, etc.)",
            "Other",
        ),
        False,
    ]
]

HowOften = Series[
    Annotated[
        pd.CategoricalDtype,
        (
            "Not at all",
            "Several days",
            "More than half the days",
            "Nearly every day",
        ),
        True,
    ]
]

ColourBlind = Series[
    Annotated[pd.CategoricalDtype, ("No", "Red / Green", "Blue / Yellow"), False]
]

ColourBlindnessType = Series[
    Annotated[
        pd.CategoricalDtype,
        (
            "Don't Know",
            "Protanomoly",
            "Deuteranopia",
            "Protanopia",
            "Tritanomaly",
            "Deuteranomaly",
        ),
        False,
    ]
]


GENDER_LABEL_MAP = {
    "nonbinary": [
        "fluid",
        "nonbinary",
        "non binary",
        "genderqueer",
        "variable",
        "questioning",
        "both",
    ],
    "other": ["neither", "agender", "other", "none", "non"],
    "female": [
        "female",
        "woman",
        "women",
        "lady",
        "femail",
        "feale",
        "femal",
        "femalw",
    ],
    "male": ["male", "mail", "man", "make"],
}

Gender = Series[Annotated[pd.CategoricalDtype, tuple(GENDER_LABEL_MAP.keys()), False]]


class RawBBQualtricsSchema(pa.DataFrameModel):
    user: str
    test_date: Series[Timestamp] = pa.Field(alias="start_date", **default)
    age: float = pa.Field(alias="q2", **default)
    sex: Sex = pa.Field(alias="q4", **default)
    gender_raw: str = pa.Field(alias="q5", **default)
    gender_cat: Gender = pa.Field(**default)
    education: Education = pa.Field(alias="q12", **default)
    exercise: WeeklyFrequency = pa.Field(alias="q16", **default)
    handedness: LeftRight = pa.Field(alias="q6", **default)
    country: str = pa.Field(alias="q7", **default)
    language: str = pa.Field(**default)
    number_languages: float = pa.Field(alias="q9", **default)
    household_income: HouseholdIncome = pa.Field(alias="q10", **default)
    employment: EmploymentStatus = pa.Field(alias="q14", **default)
    economic_status: EconomicStatus = pa.Field(alias="q11", **default)

    dx_diabetes: NotEmptyBool = pa.Field(alias="q24_1", **default)
    dx_obesity: NotEmptyBool = pa.Field(alias="q24_2", **default)
    dx_depression: NotEmptyBool = pa.Field(alias="q24_9", **default)
    dx_anxiety: NotEmptyBool = pa.Field(alias="q24_10", **default)
    dx_hypertension: NotEmptyBool = pa.Field(alias="q24_3", **default)
    dx_stroke: NotEmptyBool = pa.Field(alias="q24_4", **default)
    dx_hearing_loss: NotEmptyBool = pa.Field(alias="q24_14", **default)
    dx_heart_attack: NotEmptyBool = pa.Field(alias="q24_5", **default)
    dx_multiple_sclerosis: NotEmptyBool = pa.Field(alias="q24_13", **default)
    dx_parkinsons: NotEmptyBool = pa.Field(alias="q24_12", **default)
    dx_cardiac_problems: NotEmptyBool = pa.Field(alias="q24_11", **default)
    dx_concussion: NotEmptyBool = pa.Field(alias="q24_7", **default)
    dx_dementia_memory_problem: NotEmptyBool = pa.Field(alias="q24_6", **default)
    dx_other: NotEmptyBool = pa.Field(alias="q24_8", **default)
    dx_other_str: str = pa.Field(alias="q24_8__text", **default)
    dx_none: NotEmptyBool = pa.Field(alias="q24_15", **default)
    neuro_dx_adhd: NotEmptyBool = pa.Field(alias="q27_4", **default)
    neuro_dx_asd: NotEmptyBool = pa.Field(alias="q27_5", **default)
    neuro_dx_language: NotEmptyBool = pa.Field(alias="q27_6", **default)
    neuro_dx_cerebral_palsy: NotEmptyBool = pa.Field(alias="q27_11", **default)
    neuro_dx_dyslexia: NotEmptyBool = pa.Field(alias="q27_12", **default)
    neuro_dx_motor: NotEmptyBool = pa.Field(alias="q27_13", **default)
    neuro_dx_other: NotEmptyBool = pa.Field(alias="q27_14", **default)
    neuro_dx_other_str: str = pa.Field(alias="q27_14__text", **default)
    neuro_dx_none: NotEmptyBool = pa.Field(alias="q27_15", **default)

    num_conccussions: float = pa.Field(alias="q26", **default)
    hearing_aids: NoYes = pa.Field(alias="q25", **default)
    colourblind: str = pa.Field(alias="q29_1", **default)
    colourblind_type: str = pa.Field(alias="q29_2", **default)
    covid: NoYes = pa.Field(alias="q30", **default)

    religion_buddhist: NotEmptyBool = pa.Field(alias="q13_1", **default)
    religion_christian: NotEmptyBool = pa.Field(alias="q13_2", **default)
    religion_hindu: NotEmptyBool = pa.Field(alias="q13_3", **default)
    religion_jewish: NotEmptyBool = pa.Field(alias="q13_4", **default)
    religion_muslim: NotEmptyBool = pa.Field(alias="q13_5", **default)
    religion_sikh: NotEmptyBool = pa.Field(alias="q13_6", **default)
    religion_aboriginal: NotEmptyBool = pa.Field(alias="q13_7", **default)
    religion_none: NotEmptyBool = pa.Field(alias="q13_9", **default)
    religion_other: NotEmptyBool = pa.Field(alias="q13_8", **default)
    religion_other_str: str = pa.Field(alias="q13_8__text", **default)

    cigarette_vape: float = pa.Field(alias="q17", **default)
    caffeine: float = pa.Field(alias="q18", **default)
    alcohol: float = pa.Field(alias="q19", **default)
    cannabis: float = pa.Field(alias="q20", **default)
    rec_drugs_none: NotEmptyBool = pa.Field(alias="q21_1", **default)
    rec_drugs_stimulants: NotEmptyBool = pa.Field(alias="q21_2", **default)
    rec_drugs_depressants: NotEmptyBool = pa.Field(alias="q21_3", **default)
    rec_drugs_halucinogenics: NotEmptyBool = pa.Field(alias="q21_4", **default)
    rec_drugs_other: NotEmptyBool = pa.Field(alias="q21_5", **default)
    rec_drugs_other_str: str = pa.Field(alias="q21_5__text", **default)

    phq2_1: HowOften = pa.Field(alias="q22_1", **default)
    phq2_2: HowOften = pa.Field(alias="q22_2", **default)
    gad2_1: HowOften = pa.Field(alias="q22_3", **default)
    gad2_2: HowOften = pa.Field(alias="q22_4", **default)

    class Config:  # type: ignore
        drop_invalid_rows = True
        strict = "filter"


StudyProgress = Series[
    Annotated[
        pd.CategoricalDtype,
        (
            "consented",
            "demographics_complete",
            "PAAQ_started",
            "PAAQ_complete",
            "VGQ_complete",
            "debriefing_unavailable",
            "debriefing_complete",
        ),
        False,
    ]
]


@dtypes.immutable  # type: ignore
class TrueFalseStrFromBinary(pandas_engine.STRING):

    def coerce(  # type: ignore
        self, data_container: Series[Any]
    ) -> Series[pandas_engine.STRING]:
        return cast(
            Series[pandas_engine.STRING],
            data_container.map({1: "true", 0: "false"}),
        )


class BBMasterListSchema(pa.DataFrameModel):
    email: str = pa.Field(nullable=False)
    stage: StudyProgress = pa.Field(**default)
    user: str = pa.Field(nullable=False)
    creation_date: Series[Timestamp] = pa.Field(**default)
    giftcard_entry: NoYes = pa.Field(coerce=True, nullable=True)
    unsubscribed: TrueFalseStrFromBinary = pa.Field(coerce=True, nullable=True)
    fitness_tracker: FitnessTracker = pa.Field(coerce=True, nullable=True)
    test_date: Series[Timestamp] = pa.Field(coerce=True, nullable=True)

    class Config:  # type: ignore
        drop_invalid_rows = True
        add_missing_columns = True
        strict = "filter"
