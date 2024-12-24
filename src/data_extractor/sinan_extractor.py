from __future__ import annotations

import logging
from collections import defaultdict
import time
from typing import TYPE_CHECKING

import pandas as pd
from pysus.ftp.databases.sinan import SINAN

if TYPE_CHECKING:
    from pysus.data.local import ParquetSet
    from typing import Literal

from typing import TYPE_CHECKING, Literal

from pathlib import Path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
ERRORS = []
DEFAULT_PATH = "data"

SELECTED_DISEASES = [
    # "BOTU",
    # "CHAG",
    # "CHIK",
    # "COLE",
    # "COQU",
    # "DERM",
    # "DIFT",
    # "ESQU",
    # "EXAN",
    # "FMAC",
    # "FTIF",
    # "HANS",
    "HANT",
    # "HEPA",
    # "INFL",
    # "LEIV",
    # "LEPT",
    # "LERD",
    "LTAN",
    # "MALA",
    # "MENI",
    # "NTRA",
    # "PEST",
    # "PFAN",
    # "PNEU",
    # "RAIV",
    # "SDTA",
    # "SIFA",
    # "SIFC",
    # "SIFG",
    # "SRC",
    # "TETA",
    "TETN",
    "TOXC",
    # "TOXG",
    # "TRAC",
    "TUBE",
    # "VARC",
    # "ZIKA",
]


class SinanDataExtractor:
    timestamp_column = "DT_NOTIFIC"
    time_serie_key = "ID_MUNICIP"

    def __init__(
        self,
        disease_codes: list[str],
        path: str,
        frequency: str = "ME",
        split_rate=None,
    ) -> None:
        self.path = path if path else DEFAULT_PATH
        self.disease_codes = disease_codes if disease_codes else SELECTED_DISEASES
        self.frequency = frequency
        self.sinan = SINAN().load()
        self.split_rate = split_rate

    def extract(self) -> None:
        logger.info(f"Extracting SINAN data to {self.path}")
        cases: list[ParquetSet] = self.sinan.download(
            self.sinan.get_files(self.disease_codes), f"{self.path}/raw"
        )
        self._preprocessing(cases)
        logger.info(f"Finished extracting SINAN data to {self.path}")

    def _get_disease_code_by_filename(self, case: ParquetSet) -> str:
        filename = case.path.split("/")[-1]
        filename_without_extension = filename.split(".")[0]
        disease_code = "".join(filter(str.isalpha, filename_without_extension))
        assert disease_code is not None
        return disease_code

    def _get_dataset_year_by_filename(self, case: ParquetSet) -> str:
        filename = case.path.split("/")[-1]
        filename_without_extension = filename.split(".")[0]
        return "".join(filter(str.isdigit, filename_without_extension))

    def _preprocessing(self, cases: list[ParquetSet]) -> None:
        files_by_disease = defaultdict(list)

        for case in cases:
            disease_code = self._get_disease_code_by_filename(case)
            files_by_disease[disease_code].append(case)

        for disease_name in files_by_disease.keys():
            try:
                df_merged = pd.DataFrame()
                for case in files_by_disease[disease_name]:
                    dataset_year = self._get_dataset_year_by_filename(case)
                    logger.info(
                        f"Processing disease dataset: {disease_name} - {dataset_year}"
                    )
                    try:
                        df: pd.DataFrame = case.to_dataframe()

                        df = df[[self.timestamp_column, self.time_serie_key]]  # type: ignore
                        # Getting State code as the key
                        df[self.time_serie_key] =  pd.to_numeric((
                            df[self.time_serie_key].astype(str, errors="raise").str[:2]
                        ), downcast='unsigned')
                        df = self.treat_edge_cases(df, disease_name, dataset_year)
                        df_merged = pd.concat([df_merged, df], ignore_index=True)
                    except Exception as e:
                        logger.error(
                            f"Failed to load dataset: {disease_name} - {dataset_year}"
                        )
                        logger.error(f"Error: {e}", exc_info=True)
                        ERRORS.append((disease_name, dataset_year))
                        continue

                # Group by unique_id and ds and sum the cases
                grouped_df = (
                    df_merged.groupby(
                        [
                            pd.Grouper(key=self.timestamp_column, freq=self.frequency),
                            self.time_serie_key,
                        ]
                    )
                    .size()
                    .reset_index(name="y")
                )

                grouped_df = self._fill_missing_date(grouped_df).sort_values(
                    by=[self.time_serie_key, self.timestamp_column]
                )

                # Check if the two dates are too far apart
                max_date = grouped_df[self.timestamp_column].max()
                min_date = grouped_df[self.timestamp_column].min()
                logger.info(
                    f"Max date: {max_date}; Min date: {min_date} of {disease_name}"
                )

                if (max_date.year - min_date.year) > 25:
                    raise ValueError(
                        "There are more than 24 years between the max and min date in the dataset, it should be a problem"
                    )

                grouped_df.rename(
                    columns={
                        self.timestamp_column: "ds",
                        self.time_serie_key: "unique_id",
                    },
                    inplace=True,
                )

                SAVE_PATH = f"{self.path}/processed"
                Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
                grouped_df.to_parquet(
                    f"{SAVE_PATH}/{disease_name}_{self.frequency}.parquet", index=False
                )
                grouped_df.to_csv(
                    f"{SAVE_PATH}/{disease_name}_{self.frequency}.csv",
                    index=False,
                )
                logger.info(f"Saved: {disease_name} in {SAVE_PATH}")
            except Exception as e:
                logger.error(f"Failed to process disease dataset: {disease_name}")
                logger.error(f"Error: {e}", exc_info=True)
                ERRORS.append(disease_name)
                continue

    def treat_edge_cases(
        self, df: pd.DataFrame, disease_code: str, dataset_year: str
    ) -> pd.DataFrame:
        inconsistent_indices = self.get_inconsistent_dates(df)
        return self.fix_inconsistent_dates(df, inconsistent_indices)

    def _fill_missing_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing dates with 0 cases in the provided DataFrame based on the selected frequency.

        This method ensures that all dates within the range of the dataset are represented according to the specified frequency,
        even if there are no notifications for some periods. This is important because the absence of notifications in a period
        means that period does not appear in the dataset and there are no notifications in SINAN. While this approach does not
        address the issue of underreporting, it ensures consistency in the time series data by not artificially inflating or
        deflating the number of cases. The subnotification can be a problem, but it may occur even in periods with notifications,
        so, in the same way we do not increase the number of cases, we do not decrease it.
        """
        all_dates = pd.date_range(
            start=df[self.timestamp_column].min(),
            end=df[self.timestamp_column].max(),
            freq=self.frequency,
        )
        all_municipalities = df[self.time_serie_key].unique()

        idx = pd.MultiIndex.from_product(
            [all_dates, all_municipalities],
            names=[self.timestamp_column, self.time_serie_key],
        )

        return (
            df.set_index([self.timestamp_column, self.time_serie_key])
            .reindex(idx, fill_value=0)
            .reset_index()
        )

    def get_inconsistent_dates(self, df: pd.DataFrame) -> set:
        """
        Checks for inconsistent dates using vectorized operations and logs indices where dates have more than 5 months gap.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the date column to be checked.
        """
        # Backup date column
        df[self.timestamp_column + "_backup"] = df[self.timestamp_column]
        # Convert to datetime and sort
        df[self.timestamp_column] = pd.to_datetime(
            df[self.timestamp_column], errors="coerce"
        )

        # Calculate date differences
        date_diffs = abs(df[self.timestamp_column].diff())
        next_date_diffs = abs(df[self.timestamp_column].diff(-1))

        # Find indices where gaps are > 12 months (365 days)
        prev_gaps = (date_diffs > pd.Timedelta(days=365))
        next_gaps = (next_date_diffs > pd.Timedelta(days=365))
        conversion_errors = df[self.timestamp_column].isna()
        prev_gaps.iloc[0] = False
        next_gaps.iloc[-1] = False
        inconsistent_indices = set(df.index[prev_gaps | next_gaps | conversion_errors])
        if inconsistent_indices:
            logger.warning(
                f"Inconsistent date indices with more than 12 months gap: {inconsistent_indices}"
            )

        return inconsistent_indices

    def fix_inconsistent_dates(
        self, df: pd.DataFrame, inconsistent_indices: set
    ) -> pd.DataFrame:
        """
        Fixes inconsistent dates in the specified DataFrame column by replacing them with the nearest valid dates.
        It recives the inconsistent indexes and then, identifies the indices of inconsistent dates and attempts to replace them with the nearest valid dates.
        If both previous and next valid dates are found, it uses the previous date if they are in the same month,
        or calculates the midpoint month if they are in the same year. If only one valid date is found, it uses that date.
        If no valid dates are found, it raises a ValueError.
        Parameters:
            df (pd.DataFrame): The DataFrame containing the date column to be fixed.
        Returns:
            pd.DataFrame: The DataFrame with inconsistent dates fixed.
        Raises:
            ValueError: If no valid dates are found to replace an inconsistent date, or if previous and next valid dates are in different years.
        """
        for index in inconsistent_indices:
            logger.info(f"The problematic date is: {df.loc[index, self.timestamp_column + '_backup']}")
            # Get the previous and next valid indices
            prev_valid = None
            for i in range(index - 1, 0, -1):
                if i in inconsistent_indices:
                    continue
                if (
                    prev := pd.to_datetime(
                        df.loc[i, self.timestamp_column], errors="coerce"
                    )
                ) is not pd.NaT:
                    prev_valid = (i, prev)
                    break
            next_valid = None
            for i in range(index + 1, len(df)):
                if i in inconsistent_indices:
                    continue
                if (
                    next_v := pd.to_datetime(
                        df.loc[i, self.timestamp_column], errors="coerce"
                    )
                ) is not pd.NaT:
                    next_valid = (i, next_v)
                    break

            # Use previous or next valid date if available
            if prev_valid is not None and next_valid is not None:
                if abs((next_valid[1] - prev_valid[1])) <= pd.Timedelta(days=365): # type: ignore
                    mean_date = min(prev_valid[1], next_valid[1]) + abs(((next_valid[1] - prev_valid[1])) / 2) # type: ignore
                    df.loc[index, self.timestamp_column] = mean_date
                    logger.warning(
                        f"Using midpoint of {prev_valid} and {next_valid}. It result {mean_date} for index {index}"
                    )
                else:
                    raise ValueError(
                        f"Dates have more than 365 days of distance: {prev_valid} and {next_valid}"
                    )
            elif prev_valid is not None:
                df.loc[index, self.timestamp_column] = prev_valid[1]
                logger.warning(
                    f"Using previous valid date: {prev_valid} for index {index}"
                )
            elif next_valid is not None:
                df.loc[index, self.timestamp_column] = next_valid[1]
                logger.warning(
                    f"Using next valid date: {next_valid} for index {index}"
                )
            else:
                raise ValueError(
                    f"No valid dates found to replace inconsistent date at index {index}"
                )
        return df


if __name__ == "__main__":
    extractor = SinanDataExtractor(
        disease_codes=SELECTED_DISEASES,
        path=DEFAULT_PATH,
        frequency="ME",
        split_rate=None,
    )
    extractor.extract()
    logger.error(f"Errors: {ERRORS}")
