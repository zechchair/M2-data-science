"""Plotting referendum results in pandas.

In short, we want to make beautiful map to report results of a referendum. In
some way, we would like to depict results with something similar to the maps
that you can find here:
https://github.com/x-datascience-datacamp/datacamp-assignment-pandas/blob/main/example_map.png

To do that, you will load the data as pandas.DataFrame, merge the info and
aggregate them by regions and finally plot them on a map using `geopandas`.
"""
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def load_data():
    """Load data from the CSV files referundum/regions/departments."""
    referendum = pd.read_csv('data/referendum.csv', sep=';')
    regions = pd.read_csv('data/regions.csv', sep=',')
    departments = pd.read_csv('data/departments.csv', sep=',')

    return referendum, regions, departments


def merge_regions_and_departments(regions, departments):
    """Merge regions and departments in one DataFrame.

    The columns in the final DataFrame should be:
    ['code_reg', 'name_reg', 'code_dep', 'name_dep']
    """
    df_merged = regions.merge(departments, how='inner',
                              left_on='code', right_on='region_code')
    df_merged = df_merged.drop(['code_x', 'slug_x', 'slug_y'], axis=1)\
        .rename(columns={
            "region_code": "code_reg", "code_y": "code_dep", "name_y":
            "name_dep", "name_x": "name_reg"})
    df_merged = df_merged[['code_reg', 'name_reg', 'code_dep', 'name_dep']]

    return df_merged


def merge_referendum_and_areas(referendum, regions_and_departments):
    """Merge referendum and regions_and_departments in one DataFrame.

    You can drop the lines relative to DOM-TOM-COM departments, and the
    french living abroad.
    """
    referendum['Department code'] = referendum['Department code'].replace(
        ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
        ['01', '02', '03', '04', '05', '06', '07', '08', '09'])
    df = referendum.merge(regions_and_departments, how='inner',
                          left_on='Department code', right_on='code_dep')

    return df


def compute_referendum_result_by_regions(referendum_and_areas):
    """Return a table with the absolute count for each region.

    The return DataFrame should be indexed by `code_reg` and have columns:
    ['name_reg', 'Registered', 'Abstentions', 'Null', 'Choice A', 'Choice B']
    """
    df1 = referendum_and_areas.groupby(
        by=['code_reg', 'name_reg'], as_index=False)[
        ['name_reg', 'Registered', 'Abstentions', 'Null',
            'Choice A', 'Choice B']].sum()
    df1 = df1.set_index('code_reg')

    return df1


def plot_referendum_map(referendum_result_by_regions):
    """Plot a map with the results from the referendum.

    * Load the geographic data with geopandas from `regions.geojson`.
    * Merge these info into `referendum_result_by_regions`.
    * Use the method `GeoDataFrame.plot` to display the result map. The results
      should display the rate of 'Choice A' over all expressed ballots.
    * Return a gpd.GeoDataFrame with a column 'ratio' containing the results.
    """
    df = gpd.read_file('data/regions.geojson')
    df1 = df.merge(referendum_result_by_regions, how='inner',
                   left_on='nom', right_on='name_reg')
    df1['ratio'] = df1['Choice A'] / \
        (df1['Choice A']+df1['Choice B'])
    df1.plot('ratio')

    return df1


if __name__ == "__main__":

    referendum, df_reg, df_dep = load_data()
    regions_and_departments = merge_regions_and_departments(
        df_reg, df_dep
    )
    referendum_and_areas = merge_referendum_and_areas(
        referendum, regions_and_departments
    )
    referendum_results = compute_referendum_result_by_regions(
        referendum_and_areas
    )
    print(referendum_results)

    plot_referendum_map(referendum_results)
    plt.show()
