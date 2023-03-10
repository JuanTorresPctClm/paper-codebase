from datetime import datetime
from meteostat import Point, Hourly
import pandas as pd

import commonconstants as cc

albacete = Point(38.97914249075293, -1.8547650185417761)


def get_hourly_temperature(location: Point, start: datetime, end: datetime):
    """
    :param location: (latitud, longitude, [heigh]) of the point whose data are wanted to be known
    :param start: start date
    :param end: end date
    :return: DataFrame with hourly temperature values
    """
    data = Hourly(location, start, end)
    data = data.fetch()
    return data['temp']


def get_processed_temperature_by_month(location: Point, start: datetime, end: datetime, function: str = 'mean',
                                       day_hour_start: int = 0, day_hour_end: int = 23, index_as_month_labels=True):
    """
    Process hourly temperature data and groups it by month
    :param index_as_month_labels:
    :param location:
    :param start:
    :param end:
    :param function: function to be applied to the DataFrame
    :param day_hour_start: this parameter and day_hour_end can be used to restrict the hours of the day that are processed
    :param day_hour_end: see day_hour_start
    :return: DataFrame with 12 rows with the processed data
    """
    hourly_temp = get_hourly_temperature(location, start, end)
    index_num = list(
        range(1, max(hourly_temp.index.month) + 1)
    )

    if index_as_month_labels:
        index = [cc.month_dictionary[x] for x in index_num]

    col_name = f'{function}_temp'
    month_temp_df = pd.DataFrame(index=index, columns=[col_name])

    for i in index_num:
        condition = (hourly_temp.index.month == i) & \
                    (hourly_temp.index.hour >= day_hour_start) & \
                    (hourly_temp.index.hour <= day_hour_end)

        if function == 'mean':
            month_temp_df[col_name].iloc[i - 1] = hourly_temp[condition].mean()
        elif function == 'max':
            month_temp_df[col_name].iloc[i - 1] = hourly_temp[condition].max()
        elif function == 'min':
            month_temp_df[col_name].iloc[i - 1] = hourly_temp[condition].min()
        else:
            raise ValueError(f"Function {function} is not implemented")

    return month_temp_df


if __name__ == '__main__':
    start = datetime(2021, 1, 1)
    end = datetime(2021, 12, 31)
    temp = get_processed_temperature_by_month(albacete, start, end, function='mean')