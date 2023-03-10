from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import enum
import os

import numpy as np
import pandas as pd

import commonconstants as cc
import commonfunctions as cf
import config
import filesystemutils as fs


@enum.unique
class SamplingInterval(enum.Enum):
    """
    Class used to differentiate when selecting hourly, quarter hourly, ..., data.
    """
    HOUR = 0
    QUARTER_HOUR = 1


class DataReader(ABC):
    """
    Interface for reading data from external files
    """
    def __init__(self):
        self._columns_in_final_df = list()

    @abstractmethod
    def read_data(self, absolute_path, index_col=None):
        raise NotImplementedError


class ElectricityConsumptionReader(DataReader):
    """
    Interface for reading electricity consumption data from external files
    """
    def __init__(self):
        super().__init__()
        self._columns_in_final_df = ['CUPS', 'AE', 'R1', 'R2', 'R3', 'R4']

    def read_data(self, absolute_path, index_col=None):
        raise NotImplementedError

    def _set_non_labour_days(self, df):
        """
        Adds a new column indicating if the days is non-working. It is assumed a DateTimeIndex in the dataframe
        :param df:
        :return:
        """
        df_copy = df.copy()
        # Weekends
        df_copy['is_non_labour'] = df_copy.index.weekday > 4

        def set_day_as_non_labour(day: int, month: int, year: list, campuses: list, cups: str = None):
            """
            Updates dataframe with non-working days
            :param day: Non-working day number
            :param year: year in which the day is non-working
            :param campuses: List of campuses in which the day is non-working
            :param cups: If the day is applicable only to one CUPS, then include it
            :return:
            """
            nonlocal df_copy
            # df day must be the specified in the function signature
            # I think month condition is superfluous
            date_condition = (df_copy.index.day == day) & (df_copy.index.month == month) & (df_copy.index.year == year)
            if cups is None and campuses is None:
                df_copy.loc[date_condition, 'is_non_labour'] = True
                return

            mark_cups_as_non_labour = []

            er = ExcelReader()
            excel_data = er.read_excel_cups()

            if cups is not None:
                mark_cups_as_non_labour += [cups]
            elif campuses is not None:
                mark_cups_as_non_labour += list(
                    excel_data[excel_data['CAMPUS'].isin(campuses)]['CUPS']
                )

            for marked_cups in mark_cups_as_non_labour:
                df_copy.loc[date_condition & (df_copy['CUPS'] == marked_cups), 'is_non_labour'] = True

        for y in cc.non_working_days.keys():
            year_data = cc.non_working_days[y]
            for month, days_list in year_data.items():
                if len(days_list) > 0:
                    for nwd in days_list:
                        set_day_as_non_labour(day=nwd['day'], month=month, year=y, campuses=nwd['campuses'],
                                              cups=nwd['cups'])

        return df_copy

    @abstractmethod
    def _arrange_column_names(self, df):
        """
        This function is meant to change column names of a dataframe to be in
        conjunction with those defined in self._columns_in_final_df
        :param df: dataframe to change column names
        :return: df with column names changed
        """
        raise NotImplementedError

    @abstractmethod
    def _get_company_name(self):
        """
        Returns the name of the company that provided the data
        :return: str
        """
        raise NotImplementedError

    def get_dataframe_for_year_and_month(self, year, month, sampling_interval, force_computing=False):
        """
        Retrieves the data frame for the specified year and month
        :param sampling_interval:
        :param year: year of the data
        :param month: month of the data
        :param force_computing: reindexing the data frame is time consuming, so once it is done
        one time, a new file is saved. If this file is found, it is read instead of the original. If
        force_computing is True, then the original is read even if the modified one exists and is
        found
        :return: pandas DataFrame
        """
        # Get all csvs that will compose the dataframe
        csvs = self._get_file_path_for_year_month(year, month, sampling_interval)

        # Create empty dataframe
        df = pd.DataFrame()

        year_month = f"{str(year)}{str(month).zfill(2)}"
        already_computed_df_name = f"{year_month}.csv"
        already_computed_df_folder = self._get_processed_data_source_root_path(sampling_interval, self._get_company_name())
        already_computed_df_path = os.path.join(already_computed_df_folder, year_month, already_computed_df_name)

        if force_computing or not fs.path_exists(already_computed_df_path):
            # Compute the data frame
            for csv in csvs:
                # Read csv to dataframe
                df_csv = self.read_data(csv)
                # Append it to the existing data
                df = pd.concat([df, df_csv])

            # Reindex dataframe to get datetime as index
            df = self._set_date_as_dataframe_index(df)
            # Normalize column names
            df = self._arrange_column_names(df)
            # Remove unneeded extra columns
            df = self._drop_unneeded_columns(df)
            # Set column types in order for computed df and saved df to be consisten
            df = self._set_columns_types(df)
            # Save dataframe to avoid reindexing the next time
            self._save_hourly_reindexed_dataframe(df, sampling_interval)
        else:
            df = self.read_data(already_computed_df_path, index_col='index')

        return df

    def get_dataframe_for_year_month_and_cups(self, year, month, cups, sampling_interval):
        """
        Gets a dataframe for a given year, month and cups
        :param sampling_interval:
        :param year: year of the dataframe
        :param month: month of the dataframe
        :param cups: cups of the dataframe
        :return: df containing all the information of the cups for the specified
        year and dataframe
        """
        df = self.get_dataframe_for_year_and_month(year, month, sampling_interval)
        df_cups = cf.filter_by_cups(df, cups)

        return df_cups

    def get_dataframe_for_year(self, year, sampling_interval):
        """
        Gets all monthly dataframes for the specified year as a single df
        :param sampling_interval:
        :param year: year to obtain the dataframe for
        """
        # Get all available months for the specified year
        months = self._get_available_months_for_year(year, sampling_interval)
        # Create empty data to append more to it
        year_df = pd.DataFrame()

        # Compute dataframe of each month and join them
        for month in months:
            month_df = self.get_dataframe_for_year_and_month(year, month, sampling_interval)
            year_df = pd.concat([year_df, month_df])

        return year_df

    def get_dataframe_for_year_and_cups(self, year, cups, sampling_period):
        """
        Gets a yearly dataframe for one cups
        :param sampling_period:
        :param year: year to plot the data of
        :param cups: cups to plot the data of
        :return: pd.Dataframe
        """
        year_df = self.get_dataframe_for_year(year, sampling_period)

        return cf.filter_by_cups(year_df, cups)

    def _get_cupses_of_campus(self, campus_label):
        """
        Searches all cupses belonging to a campus
        :param df: dataframe to filter
        :param campus_label: capus label to be left on df
        :return: pd.Dataframe
        """
        er = ExcelReader()
        df = er.read_excel_cups()

        df_filtered = df[df['CAMPUS'].str.startswith(campus_label)].copy()
        return list(df_filtered['CUPS'])

    def _get_file_path_for_year_month(self, year, month, sampling_interval):
        """
        Gets all file paths for the specified year and month
        :param year: year
        :param month: month
        :param sampling_interval: SamplingInterval object to specify data to read
        :return: list containing all csv files paths
        """
        year = str(year)
        month = str(month).zfill(2)

        folder_name = cf.compose_folder_name(year, month)

        data_folder_path = self._get_data_source_root_path(sampling_interval, self._get_company_name())

        file_directory = os.path.join(data_folder_path, folder_name)
        file_in_directory = list(map(
            lambda x: os.path.join(data_folder_path, folder_name, x),
            os.listdir(file_directory)
        ))

        file_in_directory = list(filter(
            lambda x: x.endswith('.csv'),
            file_in_directory
        ))

        return file_in_directory

    def _drop_unneeded_columns(self, df):
        """
        This function drops the columns that are not needed. The used columns names are
        specified in the global variable columns_in_final_df
        :param df: df to drop columns from
        :return: df with the extra columns dropped
        """
        cols_to_drop = []
        for col in list(df.columns):
            if col not in self._columns_in_final_df:
                cols_to_drop.append(col)

        return df.drop(labels=cols_to_drop, axis=1)

    def _set_columns_types(self, df):
        """
        Set the column types of the dataframe to the correct ones to be able to check
        if computed dataframes and saved dataframes are the exact same.
        :param df: df to edit
        :return: df with correct types
        """
        df['CUPS'] = df['CUPS'].astype(str)

        float_columns = self._columns_in_final_df.copy()
        float_columns.remove('CUPS')

        for fc in float_columns:
            df[fc] = df[fc].apply(lambda x: str(x).replace(",", "."))
            df[fc] = df[fc].astype(np.float64)

        return df

    def _save_hourly_reindexed_dataframe(self, df, sampling_interval):
        """
        Saves a dataframe onces it has been reindexed. The purpose of this function is not being
        in the need of reindex a dataframe every time that it must be read due to the fact of
        that operation being time consuming.
        :param sampling_interval:
        :param df: dataframe to save
        :return:
        """
        # Extract one date fo compute folder and file names
        first_date = df.index[0]
        year = str(first_date.year)
        month = str(first_date.month).zfill(2)
        month_folder = cf.compose_folder_name(year, month)
        file_name = f"{month_folder}.csv"

        root_folder_directory_to_save = self._get_processed_data_source_root_path(sampling_interval, self._get_company_name())
        directory_to_save = os.path.join(root_folder_directory_to_save, month_folder)
        path_to_save = os.path.join(directory_to_save, file_name)

        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)

        df.to_csv(path_to_save, sep=';', index_label='index')

    def _set_date_as_dataframe_index(self, df):
        """
        Using date and hour of each measure, it is created a DatetimeIndex to get an easier plotting
        :param df: dataframe to modify
        :return: df
        """
        df_copy = df.copy()
        # Create a new Series as the Hora column as a string
        new_index = df_copy['Hora'].astype(str)
        # Complete it with date and minute information
        new_index = df_copy['Fecha'].astype(str) + new_index.apply(
            lambda x: " " + x.zfill(2) + ":00"
        )
        # Convert the strings to datetime
        new_index = pd.to_datetime(new_index, dayfirst=True)
        # Set the index
        df_copy = df_copy.set_index(new_index)
        # Drop, now unneeded, Hora and Fecha columns
        df_copy = df_copy.drop(columns=["Hora", "Fecha"])

        return df_copy

    def _get_data_source_root_path(self, sampling_interval, company_name):
        """
        Selects between hourly and quarter hourly data
        :param sampling_interval: SamplingInterval object
        :param company_name: company from which the data are retrieved
        :return: Absolute path to folder containing directories with year and month
        """
        if sampling_interval == SamplingInterval.HOUR:
            return os.path.join(config.CURVAS_HORARIAS_PATH, company_name)
        elif sampling_interval == SamplingInterval.QUARTER_HOUR:
            return os.path.join(config.CURVAS_CUARTO_HORARIAS_PATH, company_name)
        else:
            raise ValueError

    def _get_processed_data_source_root_path(self, sampling_interval, company_name):
        """
        Selects between hourly and quarter hourly data. The
        :param sampling_interval: SamplingInterval object
        :param company_name: company from which the data are retrieved
        :return: Absolute path to folder containing directories with year and month of the
        already processed data (saved and reindexed dataframes)
        """
        if sampling_interval == SamplingInterval.HOUR:
            return os.path.join(config.CURVAS_HORARIAS_TRATADAS_PATH, company_name)
        elif sampling_interval == SamplingInterval.QUARTER_HOUR:
            return os.path.join(config.CURVAS_CUARTO_HORARIAS_TRATADAS_PATH, company_name)
        else:
            raise ValueError

    def _get_available_months_for_year(self, year, sampling_interval):
        """
        Gets the months of the year that present data. It consideres that theres is data only
        if a folder exists
        :param sampling_interval:
        :param year: year to get the data from
        :return: list of the months containing data
        """
        year = str(year)

        # Get folders
        path_to_data = self._get_data_source_root_path(sampling_interval, self._get_company_name())

        folders = os.listdir(path_to_data)
        folders = list(
            filter(
                lambda x: x.startswith(year),
                folders
            )
        )

        # Initialize list to return
        months = []

        # Iterate over folders
        for folder in folders:
            # Minimal check of name pattern
            if len(folder) == 6:
                # Month is the last two digits
                month = int(folder[4:])
                if month not in months:
                    months.append(month)

        months.sort()
        return months

    def get_hourly_day_data_from_month_dataframe(self, day, df):
        """
        Extracts consumption information rom one day. Due to the way energy is measured, readings for one day are
        composed of the hours 1-23 of that day plus the hour 0 of the next one
        :param day: day to extract
        :param df: dataframe to extract the day from
        :return: pd.Dataframe
        """

        # Extract year and month of the studied dataframe
        year_to_extract = list(df.index.year)[0]
        month_to_extract = list(df.index.month)[0]

        # Filter datarame by day
        df_copy = df[(df.index.day == day) & (df.index.month == month_to_extract)].copy()
        final_hour = datetime(year_to_extract, month_to_extract, day, 23)

        # Drop hour zero
        hour_zero_index = df_copy[df_copy.index.hour == 0].index
        df_copy = df_copy.drop(index=hour_zero_index)

        # Compute next hour
        next_day_first_hour = final_hour + timedelta(hours=1)
        # Find next hour and append it to dataframe
        next_day_first_hour_row = df[df.index == next_day_first_hour].copy()

        df_copy = pd.concat([df_copy, next_day_first_hour_row])

        return df_copy


class CsvElectricityConsumptionReader(ElectricityConsumptionReader):
    def read_data(self, absolute_path, index_col=None):
        """
        Reads as a dataframe the specified csv
        :param absolute_path: absolute path to csv
        :param index_col: Specifies the column that is used as de dataframe index.
        It is used when reading an already processed csv file
        :return: pandas dataframe
        """
        if index_col is None:
            df = pd.read_csv(absolute_path, delimiter=";")
        else:
            df = pd.read_csv(absolute_path, delimiter=";", index_col=index_col)
            df.index = pd.to_datetime(df.index)

        df = self._set_non_labour_days(df)
        return df

    def _arrange_column_names(self, df):
        """
        See description on parent class
        """
        raise NotImplementedError

    def _get_company_name(self):
        """
        See description on parent class
        """
        raise NotImplementedError


class NexusCsvReader(CsvElectricityConsumptionReader):
    def _arrange_column_names(self, df):
        """
        NOT TESTED METHOD
        """
        return df

    def _get_company_name(self):
        return config.NEXUS_NAME


class IberdrolaCsvReader(CsvElectricityConsumptionReader):
    def _arrange_column_names(self, df):
        cols = list(df.columns)

        cols[cols.index('CONSUMO kWh')] = 'AE'
        cols[cols.index('REACT Q1')] = 'R1'
        cols[cols.index('REACT Q2')] = 'R2'
        cols[cols.index('REACT Q3')] = 'R3'
        cols[cols.index('REACT Q4')] = 'R4'

        df.columns = cols

        return df

    def _get_company_name(self):
        return config.IBERDROLA_NAME

    def _set_date_as_dataframe_index(self, df):
        """
        Using date and hour of each measure, it is created a DatetimeIndex to get an easier plotting
        :param df: dataframe to modify
        :return: df
        """
        df_copy = df.copy()
        # Create a new Series as the Hora column as a string
        new_index = df_copy['FECHA-HORA'].astype(str)
        # Convert the strings to datetime
        new_index = pd.to_datetime(new_index, dayfirst=True)
        # Set the index
        df_copy = df_copy.set_index(new_index)
        # Drop, now unneeded, Hora and Fecha columns
        df_copy = df_copy.drop(columns=["FECHA-HORA"])

        return df_copy


class ExcelReader(DataReader):
    def read_data(self, absolute_path, index_col=None):
        """
        See description on parent class
        """
        return pd.read_excel(config.EXCEL_CUPS_LOCATION_PATH)

    def read_excel_cups(self):
        """
        Reads the excel that contains CUPS information
        :return: pd.DataFrame
        """
        # Read contents from the excel file that containts each CUPS and its location
        excel_cups= self.read_data(config.EXCEL_CUPS_LOCATION_PATH)
        # Remove unneeded 1P at the end of some CUPS
        excel_cups['CUPS'] = excel_cups['CUPS'].apply(lambda x: x if not x.endswith('1P') else x[:-2])
        # Conserve only rows that have a valid CUPS
        cups_pattern = r"ES\d+[a-zA-Z]+"
        excel_cups_filtered = excel_cups[excel_cups['CUPS'].str.match(cups_pattern)]

        return excel_cups_filtered.copy()

    def get_location_name_of_cups(self, cups):
        """
        Given a CUPS, returns its location name
        :param cups:
        :return:
        """
        df = self.read_excel_cups()
        return df[df['CUPS'].str.contains(cups)]['UBICACION'].iloc[0]


if __name__ == '__main__':
    nexus_reader = NexusCsvReader()
    iberdrola_reader = IberdrolaCsvReader()

    borrar = nexus_reader.get_dataframe_for_year_and_month(2021, 7, SamplingInterval.HOUR)
    pass

    # cups = 'ES0021000010784658ER'
    # df_nexus = nexus_reader.get_dataframe_for_year_month_and_cups(2021, 4, cups, SamplingInterval.HOUR)
    # df_iberdrola = iberdrola_reader.get_dataframe_for_year_month_and_cups(2021, 4, cups, SamplingInterval.HOUR)

    year = 2021
    month = 4
    cups = "ES0021000010784658ER"
    si = SamplingInterval.HOUR

    df = iberdrola_reader.get_dataframe_for_year_month_and_cups(year, month, cups, si)

    extracted_day = iberdrola_reader.get_hourly_day_data_from_month_dataframe(30, df)

