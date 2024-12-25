from __future__ import annotations

import logging
from collections import defaultdict
import time
from typing import TYPE_CHECKING

import pandas as pd
from pysus.ftp.databases.sinan import SINAN
import numpy as np
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
    "BOTU",
    "CHAG",
    "CHIK",
    "COLE",
    "COQU",
    "DERM",
    "DIFT",
    "ESQU",
    "EXAN",
    "FMAC",
    "FTIF",
    "HANS",
    "HANT",
    "HEPA",
    "INFL",
    "LEIV",
    "LEPT",
    "LERD",
    "LTAN",
    "MALA",
    "MENI",
    "NTRA",
    "PEST",
    "PFAN",
    "PNEU",
    "RAIV",
    "SDTA",
    "SIFA",
    "SIFC",
    "SIFG",
    "SRC",
    "TETA",
    "TETN",
    "TOXC",
    "TOXG",
    "TRAC",
    "TUBE",
    "VARC",
    "ZIKA",
    # "DENG"
]

class SinanDataExtractor:
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

                        # Getting State code as the key
                        df[self.time_serie_key] =  pd.to_numeric((
                            df[self.time_serie_key].astype(str, errors="raise").str[:2]
                        ), downcast='unsigned')
                        df = self.treat_edge_cases(df, disease_name, dataset_year)
                        df = df[[self.time_serie_key, 'ds']] # type: ignore
                        df_merged = pd.concat([df_merged, df], ignore_index=True)
                    except Exception as e:
                        logger.error(
                            f"Failed to load dataset: {disease_name} - {dataset_year}"
                        )
                        
                        if str(e) == "No objects to concatenate":
                            logger.error(f"Error: {e}")
                        else:
                            logger.error(f"Error: {e}", exc_info=True)
                        ERRORS.append((disease_name, dataset_year))
                        continue

                # Group by unique_id and ds and sum the cases
                grouped_df = (
                    df_merged.groupby(
                        [
                            pd.Grouper(key='ds', freq=self.frequency),
                            self.time_serie_key,
                        ]
                    )
                    .size()
                    .reset_index(name="y")
                )

                grouped_df = self._fill_missing_date(grouped_df).sort_values(
                    by=[self.time_serie_key, 'ds']
                )

                # Check if the two dates are too far apart
                max_date = grouped_df['ds'].max()
                min_date = grouped_df['ds'].min()
                logger.info(
                    f"Max date: {max_date}; Min date: {min_date} of {disease_name}"
                )

                if (max_date.year - min_date.year) > 25:
                    raise ValueError(
                        "There are more than 24 years between the max and min date in the dataset, it should be a problem"
                    )

                grouped_df.rename(
                    columns={
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
        return self.fix_inconsistent_dates(df)

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
            start=df['ds'].min(),
            end=df['ds'].max(),
            freq=self.frequency,
        )
        all_municipalities = df[self.time_serie_key].unique()

        idx = pd.MultiIndex.from_product(
            [all_dates, all_municipalities],
            names=['ds', self.time_serie_key],
        )

        return (
            df.set_index(['ds', self.time_serie_key])
            .reindex(idx, fill_value=0)
            .reset_index()
        )

    def fix_inconsistent_dates(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:

        original_notif = df['DT_NOTIFIC'].copy()
        df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC'], errors='coerce')

        # Print failed conversions for notification dates  
        failed_notif = df['DT_NOTIFIC'].isna() & ~original_notif.isna()
        if failed_notif.any():
            logger.warning(f"Failed to convert {failed_notif.sum()} notification dates:")
            logger.warning(original_notif[failed_notif].value_counts())
        # Check if DT_DIAG exists in the dataframe
        if 'DT_DIAG' in df.columns:
            # Convert dates to datetime using vectorized operations
            # Create backup of original dates
            original_diag = df['DT_DIAG'].copy()

            # Convert dates and check for failures
            df['DT_DIAG'] = pd.to_datetime(df['DT_DIAG'], errors='coerce')

            # Print failed conversions for diagnosis dates
            failed_diag = df['DT_DIAG'].isna() & ~original_diag.isna()
            if failed_diag.any():
                logger.warning(f"Failed to convert {failed_diag.sum()} diagnosis dates:")
                logger.warning(original_diag[failed_diag].value_counts())



            # Vectorized date difference calculation
            date_diff = df['DT_NOTIFIC'] - df['DT_DIAG']
            mask = ~date_diff.isna() & (np.abs(date_diff) > pd.Timedelta(days=365))
            if mask.any():
                problematic_dates = df[mask][['DT_DIAG', 'DT_NOTIFIC']]
                logger.warning("Found notifications with more than 12 months between diagnosis and notification dates:")
                logger.warning(f"\n{problematic_dates}")
                # Use DT_NOTIFIC for problematic dates
                df.loc[mask, 'DT_DIAG'] = df.loc[mask, 'DT_NOTIFIC']

            # Vectorized fallback to DT_NOTIFIC 
            df['ds'] = df['DT_DIAG'].fillna(df['DT_NOTIFIC'])
        else:
            # If DT_DIAG doesn't exist, just use DT_NOTIFIC
            failed_notif = df['DT_NOTIFIC'].isna() & ~original_notif.isna()
            if failed_notif.any():
                logger.warning(f"Failed to convert {failed_notif.sum()} notification dates:")
                logger.warning(original_notif[failed_notif].value_counts())
            df['ds'] = df['DT_NOTIFIC']

        # Log and drop invalid rows in one operation
        if df['ds'].isna().any():
            logger.warning(f"Dropping rows with invalid dates")
            df = df.dropna(subset=['ds'])
        # Log rows where ds is before 1990
        early_dates = df[df['ds'].dt.year < 1990]
        if not early_dates.empty:
            logger.warning(f"Found {len(early_dates)} rows with dates before 1990:")
            logger.warning(early_dates)
        # Drop rows where ds year equals ANO_NASC if ANO_NASC exists
        if 'ANO_NASC' in df.columns:
            if (invalid_year_mask := df['ds'].dt.year == df['ANO_NASC']).any():
                logger.warning(f"Dropping {invalid_year_mask.sum()} rows where date equals birth year")
                df = df[~invalid_year_mask] # type: ignore

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
