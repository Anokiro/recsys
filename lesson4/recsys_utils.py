"""
utils

"""
import pandas as pd


def prefilter_items(data: pd.DataFrame,
                    group_col: str = None,
                    popular_col: str = None,
                    time_col: str = None,
                    price_col: str = None,
                    top_popular_filter_del: int = None,
                    top_popular_filter_choose: int = None,
                    top_unpopular_filter: int = None,
                    time_unpopular_filter: int = None,
                    chip_item_filter: int = None,
                    exp_item_filter: int = None) -> pd.DataFrame:
    """
Заменяет Item на фиктивный в датасете по набору условий
    """

    n_filter_start = (data[group_col] == 999999).sum()

    popularity = data.groupby(group_col)[popular_col].sum().reset_index()

    if top_popular_filter_choose:  # оставляет N самых полулярных товаров

        top = popularity.sort_values(popular_col, ascending=False).head(
            top_popular_filter_choose).item_id.tolist()
        data.loc[~data[group_col].isin(top), group_col] = 999999

    if top_popular_filter_del:  # фильтр самых популярных  (которые и так купят)

        top = popularity.sort_values(popular_col, ascending=False).head(
            top_popular_filter_del).item_id.tolist()
        data.loc[data[group_col].isin(top), group_col] = 999999

    if top_unpopular_filter:  # фильтр самых популярных товаров

        bottom = popularity.sort_values(
            popular_col, ascending=True).head(top_unpopular_filter).item_id.tolist()
        data.loc[data[group_col].isin(bottom), group_col] = 999999

    if time_unpopular_filter:  # фильтр товаров, которые не продавались за последние N месяцев

        actuality = data.groupby(group_col)[time_col].min().reset_index()
        top_actual = actuality[actuality[time_col] > 365].item_id.tolist()
        data.loc[data[group_col].isin(top_actual), group_col] = 999999

    if chip_item_filter:  # Фильт товаров, которые стоят < N$

        low_price = data[data[price_col] < chip_item_filter].item_id.tolist()
        data.loc[data[group_col].isin(low_price), group_col] = 999999

    if exp_item_filter:  # Фильт товаров, которые стоят > N$ (дорогих)

        high_price = data[data[price_col] > exp_item_filter].item_id.tolist()
        data.loc[data[group_col].isin(high_price), group_col] = 999999

    n_filter = (data[group_col] == 999999).sum() - n_filter_start
    print(f'Отфильтровано {n_filter} записей')

    return data

