# pyright: reportGeneralTypeIssues=false, reportCallIssue=false

from __future__ import annotations
from abc import ABC
from pathlib import Path
import re
import sys
from typing import (
    Any,
    cast,
    Dict,
    get_args,
    List,
    Literal,
    Tuple,
    Type,
    Union,
    Optional,
    Callable,
    Protocol,
    runtime_checkable,
)

from cbspython import all_cbs_columns, TESTS, Report
import datalad.api as dl
import pandas as pd
from pandas._libs.parsers import STR_NA_VALUES
from pandas._typing import CompressionOptions
import pandera as pa
from pandera import DataFrameModel


def to_snakecase(s: str):
    """Convert a space separated and capitalized (i.e. titleized) string into
        an underscore and lower-case form (i.e., snakecase)

    Arguments:
        s {str} -- a string it title format

    Returns:
        string -- the string in snakecase form
    """

    s = s.replace("-", " ")
    s = re.sub(r"([A-Z]+)", r" \1", s)
    s = re.sub(r"([A-Z][a-z]+)", r" \1", s)
    s = "_".join(s.split()).lower()
    s = re.sub(r"[\W]+", "", s)
    return s


def column_map_from_aliases(schema: Type[DataFrameModel]) -> dict[str, str]:
    return {getattr(schema, name): name for name in schema.__annotations__.keys()}


DataladRenderTypes = Literal["generic", "json", "json_pp", "disabled", "tailored"]
DataSourceTypes = Literal["infer", "csv", "pickle"]
PreprocStage = Literal["post_read", "post_validation"]


@runtime_checkable
class PreprocessingMethod(Protocol):
    preprocess_stage: PreprocStage
    preprocess_priority: Optional[int]

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def preprocess(
    stage: PreprocStage, priority: Optional[int] = None
) -> Callable[..., PreprocessingMethod]:
    def add_property(func: Callable[..., Any]) -> PreprocessingMethod:
        func = cast(PreprocessingMethod, func)
        func.preprocess_stage = stage
        func.preprocess_priority = priority
        return func

    return add_property


class DataSource(ABC):

    @property
    def root_path(self) -> Path:
        """The absolute path to the file that instantiated this class. This
            path is used as the basis for locating data files, etc. and should
            be located in the package root. That is, within the directory that
            has the package (dataset) name and contains an __init__.py file.

        Returns:
            (pathlib.Path): The absolute path of the package contents.

        Example:
            /Users/conor/data/SCI-78-ADHD-CFP-data/SCI78_ADHD_CFP/
        """
        subclass_pathname = sys.modules[self.__class__.__module__].__file__
        return Path(cast(Path, subclass_pathname)).parent.absolute()

    @property
    def repository_path(self) -> Path:
        """The absolute path of the repository. This is different than
            the root directory because it is one level up from the package
            root. Normally, this corresponds to the folder that a dataset
            or codebase is cloned into.

        Returns:
            (pathlib.Path): The absolute path of the repository root.

        Example:
            /Users/conor/data/SCI-78-ADHD-CFP-data/
        """
        return self.root_path.parents[0]

    @property
    def data_path(self) -> Path:
        """The absolute path to the directory that contains all datafiles
            associated with this data source.

        Returns:
            (pathlib.Path): The absolute path of the data files.

        Example:
            /Users/conor/data/SCI-78-ADHD-CFP-data/SCI78_ADHD_CFP/data/
        """
        return self.root_path / "data"

    @property
    def src_pathname(self) -> Path:
        """The full path and filename (i.e.,the  pathname) of the source data.
            Note that the path is absolute and the file is assumed to be be
            located in the pre-specified data directory.

        Returns:
            (pathlib.Path): The absolute path of the source data file.

        Example:
            /Users/conor/data/SCI-78-ADHD-CFP-data/data/SCI-78-user-data.csv.gz.crypt  # noqa
        """
        return self.data_path / self.src_filename

    @property
    def src_filename(self) -> str:
        """The filename (not including path) of the file that contains
            the data for this source. The file should be located within the
            pre-specified data directory.

            This is an abstract method and MUST be overridden in subclasses,
            hence why this is declared as a property (rather than a constant).

        Returns:
            (str): A filename, e.g., "my_datafile.csv"
        """
        raise ValueError("Source filename must be specified for subclasses.")

    _VALID_SRC_TYPES: Tuple[DataSourceTypes] = get_args(DataSourceTypes)
    _VALID_COMPRESSIONS: Tuple[Any, ...] = get_args(get_args(CompressionOptions)[0])

    @staticmethod
    def _match_opts(
        s: str, opts: List[str], n_max: int = 1, n_min: int = 1
    ) -> List[str]:
        """A handy helper function for identifying if certain options
            (substrings) are present in a given string. In the context of this
            class, we use this function to identify file types and
            compression options from filenames.

        Args:
            s (str): The string to search for substrings.

            opts (list-like): Substrings to identify within the string 's'.

            n_max (int, optional): The maximum number of matches allowed.
                Defaults to 1.

            n_min (int, optional): The minimum number of matches allowed.
                Defaults to 1.

        Raises:
            ValueError: If the number of found matches is outside the bounds
                of n_max and n_min.

        Returns:
            List: A list of the substring options found within the string 's'.

        Examples:
            >> _match_opts('test.csv', ['csv', 'pickle'])
            ... ['csv']

            >> _match_opts('test.csv', ['test', 'csv'])
            ... ValueError: Must be no more than 1 of ['test', 'csv'] in test.csv

            >> _match_opts('mytest.csv', ['test', 'csv', 'other'], n_max=2)
            ... ['test', 'csv']

            >>> _match_opts('test.csv', ['tast', 'cxv'], n_max=2)
            ... ValueError: Must be at least 1 of ['tast', 'cxv'] in test.csv

            >> _match_opts('test.csv', ['tast', 'cxv'], n_min=0, n_max=2)
            ... []
        """
        found = dict(zip(opts, [f"{o}" in s for o in opts]))
        n_matches = sum(found.values())
        if n_matches < n_min:
            raise ValueError(f"Must be at least {n_min} of {opts} in {s}")
        if n_matches > n_max:
            raise ValueError(f"Must be no more than {n_min} of {opts} in {s}")

        return [opt for opt, is_present in found.items() if is_present]

    def _infer_src_type(self) -> DataSourceTypes:
        """Infers the file type of the data source from the filename. There
            must be one, one only one, of the valid source types in the
            filename. For example, 'test.csv' will return 'csv', but
            'test.csv.pickle' or 'test' will raise an errors.

        Returns:
            (str): One of _VALID_SRC_TYPES
        """
        matches = self._match_opts(
            str(self.src_filename),
            [f".{t}" for t in self._VALID_SRC_TYPES[1:]],
            n_max=1,
            n_min=1,
        )
        return cast(DataSourceTypes, matches[0].strip("."))

    def _infer_compression_type(self) -> CompressionOptions:
        """Infers the type of file compression of the data source from the
            filename. There can be at most one of the valid compression types,
            or none. For example, 'test.csv.bz2' will return 'bz2', 'test.csv'
            will return None, and 'test.csv.bz2.gz' will raise an error.

        Returns:
            (str): One of _VALID_COMPRESSIONS, or None
        """
        matches = self._match_opts(
            str(self.src_filename),
            [f".{c}" for c in self._VALID_COMPRESSIONS[1:]],
            n_max=1,
            n_min=0,
        )
        if len(matches) == 0:
            compression = None
        else:
            compression = matches[0].strip(".")

        return cast(CompressionOptions, compression)

    def _preprocessing(self, stage: PreprocStage, **kwargs: Any) -> None:
        cls = type(self)
        cls_attrs = [(name, getattr(cls, name)) for name in dir(cls)]
        preproc_names = (
            name
            for name, attr in cls_attrs
            if isinstance(attr, PreprocessingMethod)
            and getattr(attr, "preprocess_stage", False) == stage
        )
        preproc_methods = [getattr(self, name) for name in preproc_names]
        preproc_methods.sort(
            key=lambda f: (f.preprocess_priority is None, f.preprocess_priority),  # type: ignore
            reverse=False,
        )
        for method in preproc_methods:
            method(**kwargs)

    def __init__(
        self,
        dl_renderer: DataladRenderTypes = "json_pp",
        src_type: DataSourceTypes = "infer",
        compression: CompressionOptions = "infer",
        keep_only_mandatory: bool = False,
        validate: bool = True,
        preprocessing: bool = True,
        **preproc_kwargs: Any,
    ) -> None:
        """Constructor for the DataSource Class.

        Args:
            dl_renderer (str, optional): Datalad result renderer option, passed
                to any datalad api commands. See e.g., https://docs.datalad.org/en/stable/generated/datalad.api.get.html
                Defaults to 'json_pp'.

            src_type (str, optional): Specify the format of the data source.
                Can be one of _VALID_SRC_TYPES. If 'infer', the type is
                inferred from the filename.
                Defaults to 'infer'.

            compression (str, optional): Specifies any compression applied to
                the data source file (e.g., bzip2). Can be one of
                _VALID_COMPRESSIONS (including None). If 'infer', the
                compression type is inferred from the filename.
                Defaults to 'infer'.

            keep_only_mandatory (bool, optional): If True, then will only keep
                columns that are specified in the mandatory columns property.
                Defaults to False.

            validate (bool, optional): Do we apply any validation rules, like a
                schema (if supplied). If False, no validation is applied.
                Defaults to True.

            preprocessing (bool, optional): If True, then we apply both post-read
                and post-validation processing functions. If False, skip all
                preprocessing steps.
                Defaults to True.

            preproc_kwargs: dict[str, Any], optional: Any keyword arguments to be
                passed to preprocessing methods.
        """
        self._dl_renderer: DataladRenderTypes = dl_renderer
        self._keep_only_mandatory = keep_only_mandatory
        self._do_validation = validate
        self._do_preprocessing = preprocessing

        if src_type == "infer":
            self._src_type = self._infer_src_type()
        else:
            self._src_type = src_type

        if compression == "infer":
            self._compression = self._infer_compression_type()
        else:
            self._compression = compression

        self._fetch_data()
        self._data = self.read_data()

        if self._do_preprocessing:
            self._preprocessing("post_read", **preproc_kwargs)

        if self._do_validation:
            self._validate_data()

        if self._do_preprocessing:
            self._preprocessing("post_validation", **preproc_kwargs)

        if self._keep_only_mandatory and self.mandatory_columns:
            self._data = self._data.loc[:, self.mandatory_columns]

    @property
    def dl_renderer(self) -> DataladRenderTypes:
        return self._dl_renderer

    @dl_renderer.setter
    def dl_renderer(self, value: DataladRenderTypes) -> None:
        self._dl_renderer = value

    _INDEX_COLUMNS: List[str]

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def _fetch_data(self) -> None:
        """Fetch the data (with datalad) before reading it. Can be overriden
        in subclasses that might not need datalad, or use other methods to
        make sure that data are present and intact. Note, this version is
        specialized to fetch from a dataverse.
        """
        dataset = dl.Dataset(self.data_path)
        if not dataset.is_installed():
            dl.get(self.data_path, get_data=False)  # type: ignore

        data_siblings = dataset.siblings(result_renderer=self._dl_renderer)

        if "dataverse-storage" not in [s["name"] for s in data_siblings]:
            dataset.siblings(action="enable", name="dataverse-storage")

        dl.get(  # type: ignore
            self.src_pathname,
            dataset=self.data_path,
            result_renderer=self._dl_renderer,
        )

    @property
    def _read_data_kwargs(self) -> dict[str, Any]:
        """Keyword arguments to be passed either to read_csv or read_pickle.
        Should be overriden in subclasses.

        Returns:
            dict[str, str]: Key word arguments
        """
        return {}

    def read_data(self) -> pd.DataFrame:
        """Reads the data file, but does not apply any processing. Can be
            overridden in subclasses to customize how the data are handled.

        Returns:
            (pandas.DataFrame): Data from the source file as directly read by
                Pandas into a dataframe.
        """
        if self._src_type == "csv":
            return pd.read_csv(self.src_pathname, **self._read_data_kwargs)
        elif self._src_type == "pickle":
            return pd.read_pickle(self.src_pathname, **self._read_data_kwargs)
        else:
            raise TypeError

    @preprocess("post_read")
    def _convert_created_at(self) -> None:
        if "created_at" in self._data.columns:
            self._data.loc[:, "created_at"] = pd.to_datetime(self._data["created_at"])

    # This dict is used to rename columns in the dataframe immediately after
    # reading in the data, but before data validation. Override in subclasses
    # to apply any dataset-specific renaming.
    @property
    def column_renamer(self) -> Dict[str, str]:
        return {}

    @preprocess("post_read", priority=-1)
    def _rename_columns(self, rename_columns: Optional[dict[str, str]] = None) -> None:
        """Rename columns in self._data, also checks that the columns exist.

        Args:
            rename_columns (dict[str, str], Optional): A map where keys are original
            column names, and values are their new names. If not provided, will use
            the self.column_renamer property

        Raises:
            pd.errors.InvalidColumnName: If the original column is not present in the
            dataframe this will be raised.

        Returns:
            None
        """
        if rename_columns is None:
            rename_columns = self.column_renamer

        for original, _ in rename_columns.items():
            if original not in self._data.columns:
                raise pd.errors.InvalidColumnName(f"Data does not contain '{original}'")

        self._data = self._data.rename(columns=rename_columns)

    @property
    def schema(self) -> Optional[Type[pa.DataFrameModel]]:
        return None

    def _validate_data(self) -> None:
        """Perform some validation steps on the internal dataframe. Returns
        nothing if everything passes, otherwise errors will be raised.

        FUTURE SELF: look into Pandera for dataframe validation
            https://pandera.readthedocs.io/en/stable/try_pandera.html
        """
        assert self._data is not None
        if self.schema:
            self._data = cast(pd.DataFrame, self.schema(self._data, lazy=True))
        else:
            if self.mandatory_columns is not None:
                for c in self.mandatory_columns:
                    if c not in self._data:
                        raise pd.errors.InvalidIndexError(
                            f"Data does not contain column '{c}'."
                        )

    @property
    def mandatory_columns(self) -> Union[None, List[str]]:
        """This method returns a list-like of strings that must be
        present in the data (as columns). This must be overriden in
        subclasses.
        """
        pass

    @property
    def frozen_name(self) -> str:
        return self.__class__.__name__

    @property
    def frozen_file(self) -> Path:
        return self.data_path / f"{self.frozen_name}.frozen.pickle"

    def freeze(
        self,
        datalad_commit: bool = False,
        commit_message: Optional[str] = None,
        **datalad_kwargs: Any,
    ) -> None:

        if self.frozen_file.exists():
            dl.unlock(self.frozen_file)  # type: ignore

        self._data.to_pickle(self.frozen_file)
        if datalad_commit:
            dl.save(self.frozen_file, message=commit_message, **datalad_kwargs)  # type: ignore

    @classmethod
    def from_frozen(cls, *args: Any, **kwargs: Any) -> DataSource:
        instance = cls.__new__(cls, *args, **kwargs)
        dl.get(instance.frozen_file)  # type: ignore
        instance._data = pd.read_pickle(instance.frozen_file)
        return instance


class CogsData(DataSource):
    def __init__(
        self, *args: Any, abbreviate_columns: bool = True, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._abbreviate_columns = abbreviate_columns

    @property
    def mandatory_columns(self) -> List[str]:
        return ["user_id", "session_data", "test_id", "cogs_created_at", "single_score"]

    @property
    def column_renamer(self) -> Dict[str, str]:
        return {
            "created_at": "cogs_created_at",
        }

    @property
    def report(self) -> Report:
        return Report(self._data)

    @property
    def excluded_features(self) -> List[str]:
        return ["better_than_chance", "duration_ms", "attempted", "errors"]

    @property
    def included_features(self) -> List[str]:
        return []

    @property
    def keep_cols_in_wideform(self) -> List[str]:
        return []

    @property
    def retain_columns(self) -> List[str]:
        return []

    @staticmethod
    def _abbrev_helper(df: pd.DataFrame) -> pd.DataFrame:
        """Helper function for abbreviating columns of a dataframe. This is a
            a hacky way of dealing with the fact that some columsn might not
            be of the (testname, feature) format. Like, what do we do with a
            column named ("age", "")?

        Args:
            df (pd.DataFrame): The input dataframe

        Returns:
            pd.DataFrame: The dataframe, with collapsed column index.
        """
        score_features: List[str] = all_cbs_columns(df)
        new_cols_names: List[str] = []
        for c in df.columns:
            if c in score_features:
                new_cols_names.append(f"{TESTS[c[0]].abbrev}_{c[1]}")
            else:
                new_cols_names.append("_".join(c).strip("_"))
        df.columns = new_cols_names
        return df

    @property
    def scores(self) -> pd.DataFrame:
        scores: pd.DataFrame = self.report.to_wideform(
            include_common=True,
            exclude=self.excluded_features,
            retain_columns=self.keep_cols_in_wideform + self.included_features,
        )
        for keep in self.keep_cols_in_wideform:
            all_injected_cols = [c for c in scores if c[1] == keep]  # type: ignore
            scores[keep] = scores[all_injected_cols[0]].copy()
            scores = scores.drop(columns=all_injected_cols)
        if self._abbreviate_columns:
            scores = self._abbrev_helper(scores)

        return scores


class QuestionnaireData(DataSource):
    _QUESTION_ANSWER: re.Pattern[str] = re.compile(r"\w+\Z")

    def _strip_header(self, text: str) -> str:
        """Questionnaire data exported from creyos tend to have headers that
            might look like "medical_standards.swan.<text>". We should get rid
            of the header things to save space.

        Args:
            text (str): The questionnaire text string for a question / answer.

        Returns:
            str: The stripped string.
        """
        m = re.search(self._QUESTION_ANSWER, text)
        if m is not None:
            return m[0]
        else:
            return text

    @preprocess("post_read")
    def _strip_medical_headers(self) -> None:
        self._data.loc[:, "question"] = self._data["question"].map(self._strip_header)
        self._data.loc[:, "answer"] = self._data["answer"].map(self._strip_header)

    @preprocess("post_read")
    def _drop_unecessary_columns(self) -> None:
        self._data = self._data.drop(
            columns=["question_id", "questionnaire_id", "answer_id"]
        )

    @property
    def column_renamer(self) -> dict[str, str]:
        return {
            "user": "user_id",
            "session": "questionnaire_session_id",
            "created_at": "question_created_at",
        }

    @property
    def mandatory_columns(self) -> List[str]:
        return ["user_id", "questionnaire_session_id"]

    @property
    def answers(self) -> pd.DataFrame:
        return self._data.drop(columns=["questionnaire_score"]).set_index(
            [
                "user_id",
                "questionnaire_session_id",
                "questionnaire_type",
                "question_position",
            ]
        )

    @property
    def scores(self) -> pd.DataFrame:
        return (
            self._data.set_index(["user_id", "questionnaire_session_id"])
            .groupby(["user_id", "questionnaire_session_id"], as_index=True)
            .nth(0)
            .loc[
                :, ["questionnaire_score", "questionnaire_type", "question_created_at"]
            ]
            .rename(columns={"question_created_at": "questionnaire_created_at"})
        )


class QualtricsData(DataSource):

    _INDEX_COLUMNS: List[str] = ["user"]

    @property
    def mandatory_columns(self) -> List[str]:
        return [
            "start_date",
            "end_date",
            "progress",
            "status",
            "duration_in_seconds",
            "finished",
            "recorded_date",
            "response_id",
            "distribution_channel",
        ]

    @property
    def _read_data_kwargs(self) -> dict[Any, Any]:
        return {
            "keep_default_na": False,
            "na_values": [s for s in STR_NA_VALUES if s not in ["None"]],
        }

    @preprocess("post_read", priority=0)
    def _convert_columns_to_snakecase(self) -> None:
        self._data.columns = [to_snakecase(c) for c in self._data.columns]

    @preprocess("post_read", priority=1)
    def _format_header_rows(self) -> None:
        self._data.loc[0, :] = self._data.loc[0, :].apply(to_snakecase)
        self._question_map: dict[str, str] = self._data.iloc[0].to_dict()
        self._data = self._data.drop([0, 1]).reset_index(drop=True)

    @preprocess("post_read")
    def _convert_dates(self) -> None:
        for c in ["start_date", "end_date"]:
            self._data[c] = pd.to_datetime(self._data[c])

    @preprocess("post_validation")
    def set_index(self) -> None:
        self._data = self._data.set_index(self._INDEX_COLUMNS)

    @property
    def question_map(self) -> dict[str, str]:
        return self._question_map
