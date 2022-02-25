"""
recommenders

"""
import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True):
        # your_code.
        self.data = data

        self.user_item_matrix = self.prepare_matrix()  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(
            self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)

        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    def prepare_matrix(self, index: str = 'user_id', columns: str = 'item_id', values: str = 'quantity'):
        # your_code

        user_item_matrix = pd.pivot_table(self.data,
                                          index=index,
                                          columns=columns,
                                          values=values,
                                          aggfunc='count',
                                          fill_value=0)

        self.user_item_matrix = user_item_matrix.astype(float)
        return user_item_matrix

    def prepare_dicts(self, user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    def fit(self, user_item_matrix, n_factors=50, regularization=0.7, iterations=5, num_threads=2):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def get_similar_items_recommendation(self, item, N=5):

        #     """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # your_code
        rec_list_s = []
        recs = self.model.similar_items(self.itemid_to_id[item], N=N + 1)
        for ii in range(1, len(recs)):
            rec_list_s.append(self.id_to_itemid[recs[ii][0]])

        return rec_list_s

    def get_similar_users(self, user, N=5):
        #     """Находим похожих юзеров"""

        rec_list_u = []
        recs = self.model.similar_users(self.userid_to_id[user], N=N + 1)
        for ii in range(1, len(recs)):
            rec_list_u.append(self.id_to_userid[recs[ii][0]])

        return rec_list_u

    def get_similar_users_recommendation(self, user, N=5):
        #     """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        # your_code

        similars_list = self.get_similar_users(user, N=5)

        data_ = self.data[['user_id', 'item_id']].loc[self.data['user_id'].isin(similars_list)]
        rec_list_items = data_['item_id'].value_counts()[:N].index.tolist()

        return rec_list_items

#         assert len(res) == N, 'Количество рекомендаций != {}'.format(N)


