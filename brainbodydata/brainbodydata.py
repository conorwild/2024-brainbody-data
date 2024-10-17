# pyright: reportMissingTypeArgument=false
from typing import List, Dict, Type, Literal, Union, Optional, get_args, overload

import numpy as np
import pandas as pd
import pandera as pa

from .schemas import (
    RawPAAQSchema,
    RawVGQSchema,
    RawBBQualtricsSchema,
    BBMasterListSchema,
    GENDER_LABEL_MAP,
)
from .src import (
    CogsData,
    column_map_from_aliases,
    DataladRenderTypes,
    DataSource,
    preprocess,
    QualtricsData,
    to_snakecase,
)

START_DATE = pd.Timestamp("2024-01-11")


class BBRawCogsData(DataSource):
    """This class is simply a DataSource wrapper for the wrap cognitive data. This is
    really just useful for pulling the raw data file if you need it.
    """

    @property
    def src_filename(self) -> str:
        return "brainbodystudy-cogs-raw.csv.bz2"


class BBCogsData(CogsData):

    @property
    def src_filename(self) -> str:
        return "brainbodystudy-cogs.report.pickle"


class BBQuestionnaire(QualtricsData):
    @property
    def src_filename(self) -> str:
        return "BBDemographic_Survey_June_19_2024_11_17.csv.bz2"

    @property
    def column_renamer(self) -> dict[str, str]:
        return {"client_id": "user"}

    @preprocess("post_read", priority=1)
    def _process_user_identifier(self) -> None:
        self._data.loc[:, "user"] = self._data["user"].str.lower()

    @preprocess("post_read", priority=2)
    def _combine_language_cols(self) -> None:
        self._data.loc[:, "language"] = (
            self._data.loc[:, self._data.columns.str.startswith("q8")]
            .apply(lambda r: ",".join(r.dropna()), axis=1)  # type: ignore
            .replace({"": np.nan})
        )

    @staticmethod
    def map_gender(g: Union[str, float]):
        if isinstance(g, str):
            g = g.lower().strip()

            for label, responses in GENDER_LABEL_MAP.items():
                for response in responses:
                    if response in g:
                        return label
        return np.nan

    @preprocess("post_read")
    def _map_genders(self) -> None:
        self._data = self._data.assign(gender_cat=self._data["q5"].map(self.map_gender))

    @property
    def schema(self) -> Type[pa.DataFrameModel]:
        return RawBBQualtricsSchema

    @preprocess("post_validation", priority=0)
    def _rename_all_columns(self) -> None:
        self._data = self._data.rename(columns=column_map_from_aliases(self.schema))

    @preprocess("post_validation", priority=1)
    def _score_GAD2(self) -> None:
        self._data.loc[:, "gad2"] = (
            self.data.loc[:, ["gad2_1", "gad2_2"]]
            .apply(lambda c: c.cat.codes)  # type: ignore
            .sum(axis=1)
        )
        self._data.loc[self._data["gad2"] < 0, "gad2"] = np.nan
        self._data.loc[:, "gad2_flag"] = self.data.loc[:, "gad2"] >= 3

    @preprocess("post_validation", priority=1)
    def _score_PHQ2(self) -> None:
        self._data.loc[:, "phq2"] = (
            self.data.loc[:, ["phq2_1", "phq2_2"]]
            .apply(lambda c: c.cat.codes)  # type: ignore
            .sum(axis=1)
        )
        self._data.loc[self._data["phq2"] < 0, "phq2"] = np.nan
        self._data.loc[:, "phq2_flag"] = self.data.loc[:, "phq2"] >= 3


class PAAQ(QualtricsData):
    """
    This class wraps and processes the data collected from our Qualtrics version of the
    Physical Activity Adult Questionnaire (PAAQ).  See the following references for
    background, questionnaire details, and questionnaire scoring:

    References

        1. Colley, R. C., Butler, G., Garriguet, D., Prince, S. A. & Roberts, K. C.
        Comparison of self-reported and accelerometer-measured physical activity in
        Canadian adults. Heal. Reports 29, 3–15 (2018).

        2. Colley, R. C., Butler, G., Garriguet, D., Prince, S. A. & Roberts, K. C.
        Comparison of self-reported and accelerometer-measured physical activity among
        Canadian youth. Heal. Reports 30, 3–12 (2019).

    """

    @property
    def src_filename(self) -> str:
        return "PAAQ_June_19_2024_08_20.csv.bz2"

    @property
    def column_renamer(self) -> Dict[str, str]:
        return {"client_id": "user"}

    Domain = Literal["transportation", "recreation", "other", "vigorous"]

    @property
    def domains(self) -> List[Domain]:
        return list(get_args(self.Domain))

    @preprocess("post_read")
    def _combine_tracker_instructions_fail_qs(self) -> None:
        fail_qs = ["q27", "q39", "q32", "q43"]
        self._data = self._data.assign(
            fitness_tracker_instructions_fail=~self._data[fail_qs]
            .bfill(axis=1)
            .iloc[:, 0]
            .isna()
        )

    @property
    def schema(self) -> Type[pa.DataFrameModel]:
        return RawPAAQSchema

    @preprocess("post_validation", priority=0)
    def _rename_all_columns(self) -> None:
        self._data = self._data.rename(columns=column_map_from_aliases(self.schema))

    @preprocess("post_validation", priority=1)
    def _calculate_weekly_total(self) -> None:
        for domain in self.domains:
            hrs = f"{domain}_hours"
            mins = f"{domain}_minutes"
            self._data.loc[:, f"{domain}_total"] = (  # type: ignore
                self._data[hrs] * 60 + self._data[mins]  # type: ignore
            )


class VGQ(QualtricsData):
    """
    This class wraps and processes the data collected from our Qualtrics version of the
    Video Game Questionnaire (VGQ). See the following references.

    References
        1. Bediou, B., Mayer, R., Barbara, S., Tipton, E. & Bavelier, D. Meta-Analysis
        of Action Video Game Impact on. Psychol. Bull. 144, 77–110 (2017).

    Other Links
        https://www.unige.ch/fapse/brainlearning/vgq
        https://www.unige.ch/fapse/brainlearning/download_file/view/44f93e5d-ade6-43ed-8247-10bf5bb5cff3/250

    """

    @property
    def src_filename(self) -> str:
        return "VGQ_June_19_2024_08_22.csv.bz2"

    @property
    def column_renamer(self) -> Dict[str, str]:
        return {"client_id": "user"}

    @property
    def schema(self) -> Type[pa.DataFrameModel]:
        return RawVGQSchema

    @preprocess("post_validation")
    def _rename_all_columns(self) -> None:
        self._data = self._data.rename(columns=column_map_from_aliases(self.schema))

    GameGenre = Literal[
        "fps", "rpg", "sports", "rts", "roleplay", "strategy", "music", "other"
    ]
    ScreenType = Literal["small", "touch"]
    When = Literal["past_year", "prev_year"]
    GamerType = Literal["NVGP", "AVGP_1", "AVGP_2", "AVGP_3", "AVGP_4", "Low_Tweener"]

    @property
    def action_genres(self) -> list[GameGenre]:
        return list(get_args(self.GameGenre))[0:4]

    @property
    def nonaction_genres(self) -> list[GameGenre]:
        return list(get_args(self.GameGenre))[4:]

    @property
    def all_genres(self) -> list[GameGenre]:
        return list(get_args(self.GameGenre))

    @property
    def gamer_types(self) -> list[GamerType]:
        return list(get_args(self.GamerType))

    @overload
    def columns_for(
        self,
        genres: GameGenre,
        when: When,
        screen: Optional[ScreenType] = None,
        as_codes: bool = False,
    ) -> pd.Series: ...

    @overload
    def columns_for(
        self,
        genres: List[GameGenre],
        when: When,
        screen: Optional[ScreenType] = None,
        as_codes: bool = False,
    ) -> pd.DataFrame: ...

    def columns_for(
        self,
        genres: Union[List[GameGenre], GameGenre],
        when: When,
        screen: Optional[ScreenType] = None,
        as_codes: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        is_series = isinstance(genres, str)
        genres = [genres] if isinstance(genres, str) else genres

        column_names = [f"{genre}_{when}" for genre in genres]
        if screen:
            column_names = [f"{screen}_{column}" for column in column_names]

        aliased_names = [getattr(RawVGQSchema, x) for x in column_names]  # type: ignore
        if is_series:
            aliased_names = aliased_names[0]

        selected_cols = self._data.loc[:, aliased_names]
        if as_codes:
            selected_cols = selected_cols.apply(lambda c: c.cat.codes)  # type: ignore

        return selected_cols

    def no_small_screen(self, genre: GameGenre, when: When) -> pd.Series:

        return (self.columns_for(genre, when, "small") == "No") & (
            self.columns_for(genre, when, "touch") == "No"
        )

    @preprocess("post_validation", priority=5)
    def score_VGQ(self) -> None:
        """This is the wildly complicated function used to score the results of the VGQ."""

        condition_1_nvgp = (
            self.columns_for(self.action_genres, "past_year") <= "Less than 1 hour"
        ).all(axis=1)

        condition_2_nvgp = (
            self.columns_for(self.nonaction_genres, "past_year")
            <= "Between 1 - 3 hours"
        ).all(axis=1)

        condition_3_nvgp = (
            self.columns_for(self.all_genres, "past_year", as_codes=True).sum(axis=1)
            < 5
        )

        condition_4_nvgp = (
            self.columns_for(self.action_genres, "prev_year") <= "Less than 1 hour"
        ).all(axis=1)

        condition_5_nvgp = (
            self.columns_for(self.nonaction_genres, "prev_year")
            <= "Between 1 - 3 hours"
        ).all(axis=1)

        condition_6_nvgp = (
            self.columns_for(self.all_genres, "past_year", as_codes=True).sum(axis=1)
            < 5
        )

        self._data = self._data.assign(
            NVGP=(
                condition_1_nvgp
                & condition_2_nvgp
                & condition_3_nvgp
                & condition_4_nvgp
                & condition_5_nvgp
                & condition_6_nvgp
            )
        )

        condition_1_avgp_1 = (
            (self.columns_for("fps", "past_year") >= "Between 5 - 10 hours")
            & self.no_small_screen("fps", "past_year")
        ) | (
            (self.columns_for("rpg", "past_year") >= "Between 5 - 10 hours")
            & self.no_small_screen("rpg", "past_year")
        )

        condition_2_avgp_1 = (
            self.columns_for(self.nonaction_genres, "past_year")
            <= "Between 1 - 3 hours"
        ).all(axis=1)

        self._data = self._data.assign(AVGP_1=condition_1_avgp_1 & condition_2_avgp_1)

        condition_1_avgp_2 = (
            (self.columns_for("fps", "past_year") >= "Between 3 - 5 hours")
            & self.no_small_screen("fps", "past_year")
        ) | (
            (self.columns_for("rpg", "past_year") >= "Between 3 - 5 hours")
            & self.no_small_screen("rpg", "past_year")
        )

        condition_2_avgp_2 = (
            (self.columns_for("fps", "prev_year") >= "Between 5 - 10 hours")
            & self.no_small_screen("fps", "prev_year")
        ) | (
            (self.columns_for("rpg", "prev_year") >= "Between 5 - 10 hours")
            & self.no_small_screen("rpg", "prev_year")
        )

        condition_3_avgp_2 = (
            self.columns_for(self.nonaction_genres, "past_year")
            <= "Between 1 - 3 hours"
        ).all(axis=1)

        self._data = self._data.assign(
            AVGP_2=(condition_1_avgp_2 & condition_2_avgp_2 & condition_3_avgp_2)
        )

        condition_1_avgp_3 = (
            (self.columns_for("fps", "past_year") >= "Between 3 - 5 hours")
            & self.no_small_screen("fps", "past_year")
        ) | (
            (self.columns_for("rpg", "past_year") >= "Between 3 - 5 hours")
            & self.no_small_screen("rpg", "past_year")
        )

        condition_2_avgp_3 = (
            self.columns_for("sports", "past_year") >= "Between 5 - 10 hours"
        ) & self.no_small_screen("sports", "past_year")

        condition_3_avgp_3 = (
            self.columns_for(self.nonaction_genres, "past_year")
            <= "Between 1 - 3 hours"
        ).all(axis=1)

        self._data = self._data.assign(
            AVGP_3=condition_1_avgp_3 & condition_2_avgp_3 & condition_3_avgp_3
        )

        condition_1_avgp_4 = (
            (self.columns_for("fps", "past_year") >= "Between 3 - 5 hours")
            & self.no_small_screen("fps", "past_year")
        ) | (
            (self.columns_for("rpg", "past_year") >= "Between 3 - 5 hours")
            & self.no_small_screen("rpg", "past_year")
        )

        condition_2_avgp_4 = (
            self.columns_for("rts", "past_year") >= "Between 5 - 10 hours"
        ) & self.no_small_screen("rts", "past_year")

        condition_3_avgp_4 = (
            self.columns_for(self.nonaction_genres, "past_year")
            <= "Between 1 - 3 hours"
        ).all(axis=1)

        self._data = self._data.assign(
            AVGP_4=condition_1_avgp_4 & condition_2_avgp_4 & condition_3_avgp_4
        )

        condition_1_low_tweener = (
            self.columns_for("fps", "past_year") <= "Less than 1 hour"
        )

        condition_2_low_tweener = (
            self.columns_for("rpg", "past_year") <= "Less than 1 hour"
        )

        condition_3_low_tweener = (
            (self.columns_for("sports", "past_year") <= "Between 1 - 3 hours")
            & (self.columns_for("rts", "past_year") <= "Less than 1 hour")
        ) | (
            (self.columns_for("sports", "past_year") <= "Less than 1 hour")
            & (self.columns_for("rts", "past_year") <= "Between 1 - 3 hours")
        )

        condition_4_low_tweener = (
            self.columns_for("fps", "prev_year") <= "Between 1 - 3 hours"
        )

        condition_5_low_tweener = (
            self.columns_for("rpg", "prev_year") <= "Between 1 - 3 hours"
        )

        condition_6_low_tweener = (
            self.columns_for("sports", "prev_year") <= "Between 3 - 5 hours"
        )

        condition_7_low_tweener = (
            self.columns_for("rts", "prev_year") <= "Between 3 - 5 hours"
        )

        condition_8_low_tweener = ~self._data["NVGP"]

        condition_9_low_tweener = (
            self.columns_for(self.all_genres, "past_year", as_codes=True).mean(axis=1)
            <= 10
        ) | (
            (
                self.columns_for(self.all_genres, "past_year", as_codes=True).mean(
                    axis=1
                )
                <= 1
            )
            & (
                self.columns_for(self.all_genres, "prev_year", as_codes=True).mean(
                    axis=1
                )
                <= 5
            )
        )

        condition_10_low_tweener = (
            self.columns_for(self.all_genres, "past_year") <= "Between 3 - 5 hours"
        ).all(axis=1)

        self._data = self._data.assign(
            Low_Tweener=(
                condition_1_low_tweener
                & condition_2_low_tweener
                & condition_3_low_tweener
                & condition_4_low_tweener
                & condition_5_low_tweener
                & condition_6_low_tweener
                & condition_7_low_tweener
                & condition_8_low_tweener
                & condition_9_low_tweener
                & condition_10_low_tweener
            )
        )


class BrainBodyStudy:
    def __init__(self, dl_renderer: DataladRenderTypes = "disabled") -> None:
        self._cogs = CogsData(dl_renderer=dl_renderer)
        self._qq = BBQuestionnaire(dl_renderer=dl_renderer)
        self._paaq = PAAQ(dl_renderer=dl_renderer)
        self._vgq = VGQ(dl_renderer=dl_renderer)

    @property
    def full_dataset(self) -> pd.DataFrame:
        return (
            self._cogs.scores.reset_index()
            .merge(
                self._qq.data.dropna(subset=["age"]),
                on=["user"],
                how="left",
            )
            # .merge(
            #     self._vgq.data,
            #     on=["user"],
            #     how="left",
            # )
            # .drop(columns=["user", "report", "batch_name", "index"])
            .dropna(axis=1, how="all")
            .query("test_date > '2024-01-09'")
        )


class BBMasterList(DataSource):
    """This class wraps the master contact list data. Not available to anyone other
    than the Principal Investigator; data file not stored in any online repository.
    This is used to generate the anonymized user ID list.
    """

    @property
    def src_filename(self) -> str:
        return "contact_list.csv"

    @property
    def schema(self) -> Type[pa.DataFrameModel]:
        return BBMasterListSchema

    def _fetch_data(self) -> None:
        """Override to do nothing, since we can't use datalad to get this file."""
        pass

    @preprocess("post_read", priority=-1)
    def _convert_columns_to_snakecase(self) -> None:
        self._data.columns = [to_snakecase(c) for c in self._data.columns]

    @property
    def column_renamer(self) -> dict[str, str]:
        return {"external_data_reference": "user", "giftcard": "giftcard_entry"}

    @preprocess("post_read", priority=0)
    def _process_user_identifier(self) -> None:
        self._data.loc[:, "user"] = self._data["user"].str.lower()

    @preprocess("post_validation", priority=1)
    def _remove_emails(self) -> None:
        """Remove any rows with an email address that matches one of the options
        here that we should remove. Testers, etc."""
        _EMAILS_TO_DROP = [
            "conorwild",
            "alex.xue",
            "test",
            "creyos",
            "scienceandindustrymuseum.org.uk",
            r"\+\S+@gmail",
        ]

        for expr in _EMAILS_TO_DROP:
            to_drop = self._data["email"].str.contains(expr).astype("bool")
            self._data = self._data[~to_drop]

    @preprocess("post_validation", priority=2)
    def _remove_duplicates(self) -> None:
        """If an email address appears more than once, we will use the identifier that
        corresponds to the time that they made it through the farthest (or if there is
        a tie, then the one that occurs earlier). Remember, each signup (email) is
        associated with a different anonymous study identifier ('user')
        """
        self._data = (
            self._data.set_index(["email", "stage"])
            .sort_index(ascending=False)
            .groupby(["user"], as_index=False)
            .nth(0)
            .reset_index()
        )

    @preprocess("post_validation", 3)
    def _filter_by_start_date(self) -> None:
        """Remove any contacts that were added before the date that the study actually
        launched. This will remove any other testing datasets not captured by previous
        filters.
        """
        self._data = self._data[self._data["creation_date"] >= START_DATE]

    @preprocess("post_validation", 4)
    def _strip_times(self) -> None:
        """Strip times from creation timestamps, leaving only dates, so that the data
        is a bit more anonymized"""
        self._data["creation_date"] = self._data["creation_date"].dt.date

    @preprocess("post_validation", 5)
    def _drop_emails(self) -> None:
        """Drop the emails, now that we have sorted and cleaned our contact list."""
        self._data = self._data.drop(columns=["email"])
