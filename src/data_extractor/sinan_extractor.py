from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import pandas as pd
from pysus.ftp.databases.sinan import SINAN
from typing import Literal

if TYPE_CHECKING:
    from typing import List
    from pysus.data.local import ParquetSet

from pathlib import Path

DEFAULT_PATH = "data/sinan"
SELECTED_DISEASES = [
    "ANIM",
    "CHAG",
    "DENG",
    "ESQU",
    "FMAC",
    "HANT",
    "LEIV",
    "LEPT",
    "LTAN",
    "MALA",
    "PEST",
    "RAIV",
]

GeographicLevel = Literal["municipality", "region", "state"]


class SinanDataExtractor:
    timestamp_column = "DT_NOTIFIC"

    def __init__(
        self,
        disease_codes: List[str],
        path: str,
        geographic_level: GeographicLevel = "region",
        frequency: str = "MS",
    ) -> None:
        self.path = path if path else DEFAULT_PATH
        self.disease_codes = disease_codes if disease_codes else SELECTED_DISEASES
        self.frequency = frequency
        self.geographic_level = geographic_level
        match geographic_level:
            case "municipality":
                self.time_serie_key = "ID_MUNICIP"
            case "region":
                self.time_serie_key = "ID_REGIONA"
                raise ValueError("The dataset has missing data for the region level")
            case "state":
                self.time_serie_key = "SG_UF"
                raise ValueError("The dataset has missing data for the state level")


        self.sinan = SINAN().load()

    def extract(self) -> List[ParquetSet]:
        logging.info(f"Extracting SINAN data to {self.path}")
        cases: List[ParquetSet] = self.sinan.download(
            self.sinan.get_files(self.disease_codes), f"{self.path}/raw"
        )
        self._preprocessing(cases)
        logging.info(f"Finished extracting SINAN data to {self.path}")

    def _get_disease_code_by_filename(self, case: ParquetSet) -> str:
        filename = case.path.split("/")[-1]
        filename_without_extension = filename.split(".")[0]
        return "".join(filter(str.isalpha, filename_without_extension))

    def _get_dataset_year_by_filename(self, case: ParquetSet) -> str:
        filename = case.path.split("/")[-1]
        filename_without_extension = filename.split(".")[0]
        return "".join(filter(str.isdigit, filename_without_extension))

    def _preprocessing(self, cases: List[ParquetSet]) -> None:
        files_by_disease = defaultdict(list)

        for case in cases:
            disease_code = self._get_disease_code_by_filename(case)
            files_by_disease[disease_code].append(case)

        for disease_name in files_by_disease.keys():
            df_merged = pd.DataFrame()
            for case in files_by_disease[disease_name]:
                disease_code = self._get_disease_code_by_filename(case)
                dataset_year = self._get_dataset_year_by_filename(case)
                logging.info(
                    f"Processing disease dataset: {disease_code} - {dataset_year}"
                )
                df: pd.DataFrame = case.to_dataframe()

                # Check for missing values in key columns
                if df[self.time_serie_key].isna().any() or df[self.timestamp_column].isna().any():
                    raise ValueError(f"Missing values found in {self.time_serie_key} or {self.timestamp_column}")
                
                df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column])
                df[self.time_serie_key] = df[self.time_serie_key].str.extract('(\d+)').astype(int)
                df = df[[self.time_serie_key, self.timestamp_column]]
                grouped_df = (
                    df.groupby(
                        [
                            pd.Grouper(key=self.timestamp_column,
                                       freq=self.frequency),
                            self.time_serie_key,

                        ]
                    )
                    .size()
                    .reset_index(name="count_cases")
                )
                df_merged = pd.concat(
                    [df_merged, grouped_df], ignore_index=True)

            df_merged = self._fill_missing_date(df_merged).sort_values(
                by=[self.time_serie_key, self.timestamp_column])
            df_merged.rename(
                columns={self.timestamp_column: "date", self.time_serie_key: "id"}, inplace=True)
            df_merged = df_merged[['id','date','count_cases']]
            SAVE_PATH = f"{self.path}/processed"
            Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
            df_merged.to_parquet(
                f"{SAVE_PATH}/{disease_code}_{self.geographic_level}_{self.frequency}.parquet"
            )
            df_merged.to_csv(
                f"{SAVE_PATH}/{disease_code}_{self.geographic_level}_{self.frequency}.csv",
                index=False,
            )
            logging.info(f"Saved: {disease_code} in {SAVE_PATH}")
        return df_merged

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
