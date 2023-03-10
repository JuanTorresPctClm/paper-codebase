from datetime import datetime

import matplotlib as mpl
import matplotlib.dates as mpldates
from matplotlib.dates import DateFormatter, MONTHLY, rrulewrapper
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean

import commonconstants as cc
import readdata as rd
import weather


def _get_total_active_power_by_building_and_month_for_year(year:int, cupses:list, sampling_interval: rd.SamplingInterval):
    """
    Computes and returns the total active power consumption for the given year and cupses
    :return: dict[building_name][month_number] = total kWh
    """
    # Dictionary to store monthly consumptions for each cups
    consumptions = {}

    for cups in cupses:
        er = rd.ExcelReader()
        cups_location_name = er.get_location_name_of_cups(cups)

        # Dictionary to store month consumption for the given cups
        cups_month_dfs = {}

        # Get months of the year that have data available
        max_month_available = max(data_reader._get_available_months_for_year(year=year, sampling_interval=sampling_interval))

        # For each month compute its dataframe and store it
        for month in range(1, max_month_available + 1):
            df = data_reader.get_dataframe_for_year_month_and_cups(year=year, month=month, cups=cups,
                                                                   sampling_interval=sampling_interval)
            cups_month_dfs[month] = df.sum()['AE']

        consumptions[cups_location_name] = cups_month_dfs

    consumptions_df = pd.DataFrame(consumptions)
    consumptions_df = consumptions_df.rename(cc.month_dictionary)
    return consumptions_df


def _set_number_of_subdivisions_for_axis(axes: mpl.axes.Axes, axis: str, subdivisions: int):
    """
    Subdivides the specified axis of the given axes to the given number of subdivisions
    :param axes: Axes that contains the axis to set the subdivisions
    :param axis: str specifying the axis name, x or y
    :param subdivisions: Integer indicating the desired number of subdivisions
    :return:
    """
    if axis == 'x':
        a = axes.xaxis
    elif axis == 'y':
        a = axes.yaxis
    else:
        raise ValueError(f'No {axis} exists.')

    a.set_major_locator(plt.MaxNLocator(subdivisions))
    a.set_minor_locator(plt.MaxNLocator(subdivisions * 2))


def _set_thousands_separator_in_axis(axis, ax):
    """
    axis = x or y
    """
    if axis == 'x':
        ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    elif axis == 'y':
        ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    else:
        raise ValueError(f"There is no {axis} axis")


def _get_single_fig_and_ax(**params):
    """
    Creates a figure and an axes according the given params. The idea is
    to define some dictionaries of params and be exchangeable
    :param params: dictionary of params. These can be used for formatting purposes.
    :return: fig, axes
    """
    fig, ax = plt.subplots(**params)

    return fig, ax


def instant_active_power_year_consumption_cups(year: int, cups: str, sampling_interval: rd.SamplingInterval,
                                               data_reader: rd.ElectricityConsumptionReader, figax=None, **params):
    """
    Plots every single data point in the year for the given cups
    :param data_reader:
    :param sampling_interval:
    :param year: year to retrieve data from
    :param cups: cups whose data is wanted to be plotted
    :param figax: tuple containing figure and axes in which the plot is wanted to be plotted
    :return: Nothing
    """
    year = int(year)
    df = data_reader.get_dataframe_for_year_and_cups(year, cups, sampling_interval)

    if figax is None:
        fig, ax = _get_single_fig_and_ax()
    else:
        fig, ax = figax

    er = rd.ExcelReader()
    legend_label = er.get_location_name_of_cups(cups)

    ax.plot(df.index, df['AE'], label=legend_label, linewidth=.8)
    fig.suptitle(f"Year {year} electricity consumption")
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=.5)
    ax.set_ylabel("Electricity consumption (kWh)")
    _set_number_of_subdivisions_for_axis(ax, 'y', 17)

    # Date formatting
    date_form = DateFormatter("%d-%b")
    ax.xaxis.set_major_formatter(date_form)
    rrule = rrulewrapper(MONTHLY, bymonthday=(1, 15))
    rrule_locator = mpldates.RRuleLocator(rrule)
    ax.xaxis.set_major_locator(rrule_locator)

    # Compute x axis limits in order not to plot empty spaces at the beginning nor the end
    min_x_axis_lim, max_x_axis_lim = None, None

    if (min_x_axis_lim is None) or (df.index.min() < min_x_axis_lim):
        min_x_axis_lim = df.index.min()

    if (max_x_axis_lim is None) or (df.index.max() > max_x_axis_lim):
        max_x_axis_lim = df.index.max()

    ax.set_xlim(min_x_axis_lim, max_x_axis_lim)
    ax.set_ylim(0)


def instant_active_power_year_consumption_cups_multiple_rows(year: int, cups: str, sampling_interval: rd.SamplingInterval,
                                                             data_reader: rd.ElectricityConsumptionReader, figax=None,
                                                             **params):
    """
    Plots every single data point in the year for the given cups
    :param data_reader:
    :param sampling_interval:
    :param year: year to retrieve data from
    :param cups: cups whose data is wanted to be plotted
    :param figax: tuple containing figure and axes in which the plot is wanted to be plotted
    :return: Nothing
    """
    year = int(year)
    df = data_reader.get_dataframe_for_year_and_cups(year, cups, sampling_interval)

    if figax is None:
        fig, ax = _get_single_fig_and_ax()
    else:
        fig, ax = figax

    er = rd.ExcelReader()
    legend_label = er.get_location_name_of_cups(cups)

    if 'color' not in params.keys():
        ax.plot(df.index, df['AE'], label=legend_label, linewidth=.8)
    else:
        ax.plot(df.index, df['AE'], label=legend_label, linewidth=.8, color=params['color'])

    fig.suptitle(f"Year {year} electricity consumption")
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=.5)
    fig.supylabel("Electricity consumption (kWh)")

    _set_number_of_subdivisions_for_axis(ax, 'y', 5)

    # Date formatting
    date_form = DateFormatter("%d-%b")
    ax.xaxis.set_major_formatter(date_form)
    rrule = rrulewrapper(MONTHLY, bymonthday=(1, 15))
    rrule_locator = mpldates.RRuleLocator(rrule)
    ax.xaxis.set_major_locator(rrule_locator)

    # Compute x axis limits in order not to plot empty spaces at the beginning nor the end
    min_x_axis_lim, max_x_axis_lim = None, None

    if (min_x_axis_lim is None) or (df.index.min() < min_x_axis_lim):
        min_x_axis_lim = df.index.min()

    if (max_x_axis_lim is None) or (df.index.max() > max_x_axis_lim):
        max_x_axis_lim = df.index.max()

    ax.set_xlim(min_x_axis_lim, max_x_axis_lim)
    ax.set_ylim(0)


def total_yearly_consumption_of_campus(year, campus_label, sampling_interval, data_reader, figax=None, **params):
    """
    Plots the aggregate consumption of every building belonging to the specified campus
    """
    cupses = data_reader._get_cupses_of_campus(campus_label)

    er = rd.ExcelReader()
    consumption_column_name = 'Consumption'
    consumption_df = pd.DataFrame(columns=[consumption_column_name])

    for cups in cupses:
        df = data_reader.get_dataframe_for_year_and_cups(year, cups, sampling_interval)
        name = er.get_location_name_of_cups(cups)

        consumption_df.loc[name] = df['AE'].sum()

    consumption_df = consumption_df.sort_values(by=consumption_column_name, ascending=True)

    # Convert values to MWh
    consumption_df[consumption_column_name] = consumption_df[consumption_column_name] / 1000

    # Sort values from higher to lower
    consumption_df = consumption_df.sort_values(consumption_column_name, ascending=False)

    if figax is None:
        fig, ax = _get_single_fig_and_ax(layout='constrained')
    else:
        fig, ax = figax

    fig.suptitle(f'Energy consumption in {year} for buildings in {campus_label} campus')
    consumption_label = "Consumption (MWh)"

    # ax.set_xlabel(consumption_label)
    # x_axis = list(consumption_df.index)
    # ax.barh(x_axis, consumption_df[consumption_column_name])
    # ax.ticklabel_format(axis='x', style='plain')
    # ax.grid(True, axis='x')

    consumption_df.plot.bar(legend=False, ax=ax)
    ax.set_ylabel(consumption_label)
    ax.set_xlabel('Building')
    ax.grid(True, axis='y')
    fig.align_xlabels()
    # ax.tick_params(axis='x', labelrotation=45, bottom=True, top=False)

    ax.set_axisbelow(True)
    # Include thousands separator
    _set_thousands_separator_in_axis(axis='y', ax=ax)


def yearly_consumption_by_month_and_cupses(year: int, cupses: list, sampling_interval: rd.SamplingInterval,
                                           data_reader: rd.ElectricityConsumptionReader, figax=None, **params):
    """
    Annual bar plot of the selected cupses. It represents the summed value
    """
    # Determine figure and axes to plot the data to
    if figax is None:
        fig, ax = _get_single_fig_and_ax()
    else:
        fig, ax = figax

    consumptions_df = _get_total_active_power_by_building_and_month_for_year(year=year, cupses=cupses,
                                                                             sampling_interval=sampling_interval) / 1000
    consumptions_df.plot.bar(ax=ax)

    ax.legend(loc='upper right')
    ax.tick_params(axis='x', labelrotation=0, bottom=True, top=False)
    ax.grid(True, axis='y')
    ax.set_axisbelow(True)
    _set_number_of_subdivisions_for_axis(ax, 'y', 11)
    ax.set_ylabel("Monthly electricity consumption (MWh)")
    fig.suptitle("Consumption per month and building")
    _set_thousands_separator_in_axis(axis='y', ax=ax)


def deviation_yearly_consumption_by_month_and_cupses_with_respect_to_mean(year, cupses, sampling_interval, data_reader,
                                                                          figax=None, **params):
    """
    Annual bar plot of the deviation with respect to the mean of the selected cupses. It represents the summed value
    """
    # Determine figure and axes to plot the data to
    if figax is None:
        fig, ax = _get_single_fig_and_ax()
    else:
        fig, ax = figax

    consumptions_df = _get_total_active_power_by_building_and_month_for_year(year=year, cupses=cupses,
                                                                             sampling_interval=sampling_interval)
    consumptions = consumptions_df.to_dict()

    annual_means = {k: mean(month_dict.values()) for k, month_dict in consumptions.items()}
    deviations = {k: {month: 100 * (total_month - annual_means[k]) / annual_means[k] for month, total_month in month_dict.items()} for k, month_dict in consumptions.items()}

    deviations_df = pd.DataFrame(deviations)
    deviations_df = deviations_df.rename(cc.month_dictionary)
    deviations_df.plot.bar(ax=ax)

    ax.legend(loc='upper right')
    ax.tick_params(axis='x', labelrotation=0, bottom=True, top=False)
    ax.grid(True, axis='y')
    ax.set_axisbelow(True)
    _set_number_of_subdivisions_for_axis(ax, 'y', 7)
    ax.set_ylabel("Deviation (%)")
    fig.suptitle("Deviation in consumption with respect to annual mean")


def temperature(location, year, figax):
    # Determine figure and axes to plot the data to
    if figax is None:
        fig, ax = _get_single_fig_and_ax()
    else:
        fig, ax = figax

    start = datetime(int(year), 1, 1)
    end = datetime(int(year), 12, 31)
    temperatures = weather.get_hourly_temperature(location, start=start, end=end)

    month_nums = [x for x in range(1, 13)]
    dfs = []

    for num in month_nums:
        label = cc.month_dictionary[num]
        dfs.append((label, temperatures[temperatures.index.month == num]))

    fig.suptitle(f'Temperatures during year {year}')
    ax.set_ylabel('Temperature [ºC]')
    ax.grid(True, axis='y')

    labels = list(map(
        lambda x: x[0],
        dfs
    ))
    data = list(map(
        lambda x: x[1],
        dfs
    ))
    ax.boxplot(data, labels=labels, sym='.')


def yearly_electricity_use_intensity_by_month_and_cupses(year, cupses, sampling_interval, data_reader, figax=None, **params):
    """
    Annual bar plot of the selected cupses. It represents the summed value
    """
    # Determine figure and axes to plot the data to
    if figax is None:
        fig, ax = _get_single_fig_and_ax()
    else:
        fig, ax = figax

    consumptions_df = _get_total_active_power_by_building_and_month_for_year(year=year, cupses=cupses,
                                                                             sampling_interval=sampling_interval)
    er = rd.ExcelReader()
    areas_df = er.read_excel_cups()[['UBICACION', 'SUPERFICIE']]
    areas_df = areas_df[areas_df['UBICACION'].isin(consumptions_df.columns)]
    areas_df = areas_df.set_index('UBICACION').transpose()

    # Merge dataframes to be able to divide everything by SUPERFICIE
    consumptions_df = pd.concat([consumptions_df, areas_df])
    consumptions_df = consumptions_df / consumptions_df.loc['SUPERFICIE']
    # Drop SUPERFICIE column in order not to plot it
    consumptions_df = consumptions_df.drop('SUPERFICIE')

    consumptions_df.plot.bar(ax=ax)

    ax.legend(loc='upper right')
    ax.tick_params(axis='x', labelrotation=0, bottom=True, top=False)
    ax.grid(True, axis='y')
    ax.set_axisbelow(True)
    _set_number_of_subdivisions_for_axis(ax, 'y', 7)
    ax.set_ylabel("Electricity Use Intensity (kWh/m$^2$)")
    fig.suptitle("Electricity Use Intensity per month and building")


def superposed_daily_values_and_mean_for_month(year, month, cups, labour_days, sampling_interval, data_reader,
                                               color=None, figax=None, **params):
    """
    Plots all daily consumption data of a month superposed with one another and their mean
    USE superposed_daily_values.mplstyle FOR NEXT TO EACH OTHER FIGURES IN WORD
    :param year: Year to represent
    :param month: Month to represent
    :param labour_days: if True labour days are plotted, otherwise non-labour days are plotted
    :param cups: cups to represent
    :return:
    """
    # Determine figure and axes to plot the data to
    if figax is None:
        fig, ax = _get_single_fig_and_ax()
    else:
        fig, ax = figax

    if color is None:
        color = 'C0'

    month_label = cc.month_dictionary[month]

    # Plot title
    labour_label = f"{' non' if not labour_days else ''}"
    figure_title = f"{month_label}-{year}{labour_label} working days"
    fig.suptitle(figure_title)

    df = data_reader.get_dataframe_for_year_month_and_cups(year=year, month=month, cups=cups,
                                                           sampling_interval=sampling_interval)

    # Axis labels
    ax.set_xlabel("Day hour (h)")
    ax.set_ylabel("Electricity consumption (kWh)")

    # Index to synchronise dataframes and compute mean
    index = [x for x in range(0, 24)]

    # X axis values in figure
    x_axis_plot = [x + 1 for x in index]

    # Get all days in month
    days = [x for x in range(1, max(df.index.day) + 1)]
    # Create dataframe to store day values and later compute their mean
    df_to_compute_mean = pd.DataFrame(index=index)

    def get_df_for_day_and_cups(day, cups):
        # ...for a specific day
        df_day = df[df.index.day == day]
        # Check if it is a labour day
        if labour_days:
            df_day = df_day[df_day['is_non_labour'] == False]
        else:
            df_day = df_day[df_day['is_non_labour'] == True]

        return df_day

    for day in days:
        df_day = get_df_for_day_and_cups(day, cups)

        if len(df_day[df_day.index.duplicated()]) == 0:
            df_to_compute_mean.insert(day - 1, f"d{day}", df_day.set_index(df_day.index.hour)['AE'])

    # Compute mean
    df_to_compute_mean['mean'] = df_to_compute_mean.mean(axis=1)
    # Check deviations of each day with respect to the mean
    for day in days:
        df_day_unprocessed = get_df_for_day_and_cups(day, cups)
        df_day = df_day_unprocessed.set_index(df_day_unprocessed.index.hour)
        df_day['deviation'] = (
                (df_to_compute_mean['mean'] - df_day['AE']) * 100 / df_to_compute_mean['mean']
        ).abs()

        if len(df_day_unprocessed) > 0 and len(df_day_unprocessed) == len(x_axis_plot):
            ax.plot(x_axis_plot, df_day['AE'], color=color, alpha=.1)
            # ax.plot(x_axis_plot, df_day['deviation'], alpha=.1)

    # Plot mean
    er = rd.ExcelReader()
    label = er.get_location_name_of_cups(cups)
    ax.plot(x_axis_plot, df_to_compute_mean['mean'], color=color, linewidth=2,
            label=label)

    ax.legend(loc='upper left', framealpha=0.3)
    ax.grid(True, axis='y', alpha=.5)
    ax.set_xlim([1, 24])

    _set_number_of_subdivisions_for_axis(axes=ax, axis='x', subdivisions=13)
    _set_number_of_subdivisions_for_axis(axes=ax, axis='y', subdivisions=10)


def plot_everything():
    """
    Shows all figures
    :return:
    """
    plt.show()


if __name__ == '__main__':
    pass
    # for month in range(1, 13):
    #     plt.style.use('../styles/superposed_daily_values.mplstyle')
    #     data_reader = rd.NexusCsvReader()
    #     fig, ax = _get_single_fig_and_ax(layout='constrained')
    #     # fig, axes = plt.subplots(4, 1, sharex=True, layout='constrained')
    #     cupses = ['ES0021000000202888ZW', 'ES0021000000262250LR', 'ES0021000000262246HC', 'ES0021000000262247HK']
    #
    #     colors = ['C0', 'C1', 'C2', 'C3']
    #
    #     for i in range(0, 4):
    #         # instant_active_power_year_consumption_cups_multiple_rows(year=2021, cups=cupses[i],
    #         #                                                          sampling_interval=rd.SamplingInterval.HOUR,
    #         #                                                          data_reader=data_reader, figax=(fig, axes[i]),
    #         #                                                          color=colors[i])
    #
    #         # instant_active_power_year_consumption_cups(year=2021, cups=cupses[i], sampling_interval=rd.SamplingInterval.HOUR,
    #         #                                            data_reader=data_reader, figax=(fig, ax))
    #
    #         superposed_daily_values_and_mean_for_month(year=2021, month=month, cups=cupses[i], labour_days=True,
    #                                                    sampling_interval=rd.SamplingInterval.HOUR, data_reader=data_reader,
    #                                                    color=colors[i], figax=(fig, ax))
    #
    #
    #     # plot_everything()
    #
    #     name = f"{month}_laborables_{cc.month_dictionary[month]}_2021_fuente_grande"
    #     png_name = f"C:\\Users\\jtorres\\OneDrive\\consumos_uclm\\Consumo edificios UCLM\\Gráficas corregidas\\png\\laborables\\{name}"
    #     svg_name = f"C:\\Users\\jtorres\\OneDrive\\consumos_uclm\\Consumo edificios UCLM\\Gráficas corregidas\\svg\\laborables\\{name}"
    #     fig.savefig(f"{png_name}.png", format="png")
    #     fig.savefig(f"{svg_name}.svg", format="svg")

