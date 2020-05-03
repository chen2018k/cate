from algobase import AlgoBase
from prediction import PredictionImpossible
import numpy as np
from six import iteritems
import heapq
import random

class SymmetricAlgo(AlgoBase):
    """This is an abstract class aimed to ease the use of symmetric algorithms.
    A symmetric algorithm is an algorithm that can can be based on users or on
    items indifferently, e.g. all the algorithms in this module.
    When the algo is user-based x denotes a user and y an item. Else, it's
    reversed.
    """

    def __init__(self, sim_options={}, verbose=True, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.verbose = verbose

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        return self

    def switch(self, u_stuff, i_stuff):
        """Return x_stuff and y_stuff depending on the user_based field."""

        if self.sim_options['user_based']:
            return u_stuff, i_stuff
        else:
            return i_stuff, u_stuff

class KNNWithMeans(SymmetricAlgo):


    def __init__(self,dataset,base_line=False, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):


        SymmetricAlgo.__init__(self, sim_options=sim_options,
                               verbose=verbose, **kwargs)

        self.k = k
        self.min_k = min_k
        self.list_of_cats = dataset.item_weight
        self.taste_score_data = dataset.user_weight
        self.base_line = base_line
        self.user_mean = dataset.all_user_means





    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)

        self.sim = self.compute_similarities()

        self.means = np.zeros(self.n_x)
        for x, ratings in iteritems(self.xr):
            self.means[x] = np.mean([r for (_, r) in ratings])

        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            # print('inside error part')
            details = {}
            details['was_impossible'] = True
            # details['default predict1'] = self.default_prediction()
            raw_uid = self.trainset.to_raw_uid(u)
            # raw_item_id = self.trainset.to_raw_iid(i)
            details['mean predict'] = self.user_mean[raw_uid]
            # print(self.list_of_cats)

            bias = np.dot(self.list_of_cats[i], self.taste_score_data[raw_uid])

            # print(bias)

            est = self.user_mean[raw_uid]

            if (self.user_mean[raw_uid] < self.trainset.global_mean) :
                # tmp += bias
                est = est - (est * bias)
            else :
                # tmp -= bias
                est = est + (est * bias)

            # details['predict'] = est
            # print(est)
            return est, details
        
            # raise PredictionImpossible('User and/or item is unkown.')

        # x, y = self.switch(u, i)
        # neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        # k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])
        # k_neighbors = neighbors[:self.k]

        # if user_based == False then:
            # x = i
            # y = u

        est = self.means[u]

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        # for (nb, sim, r) in k_neighbors:
        #     side_info = {}
        #     # if user_based == False then:
        #         # nb = item_inner_id
        #     if sim > 0:

        #         if self.base_line:
        #             # sim += 0.1005
        #             sum_sim += sim
        #             sum_ratings += (r - self.means[nb]) * sim
        #             #sum_ratings += (r) * sim
        #             actual_k += 1
        #         else:
        #             if self.sim_options['user_based'] == True:
        #                 raw_item_id = self.trainset.to_raw_iid(i)
        #                 raw_uid = self.trainset.to_raw_uid(nb)
        #                 result = np.dot(self.list_of_cats[raw_item_id], self.taste_score_data[raw_uid])
        #             else:
        #                 result = np.dot(self.list_of_cats[nb], self.taste_score_data[y]) # y is the user in the item_based

        #             # result += 0.3
        #             sum_sim += result
        #             sum_ratings += (r - self.means[nb]) * result
        #             #sum_ratings += r * result * 100 + 121212
        #             actual_k += 1

        # if actual_k < self.min_k:
        #     sum_ratings = 0

        # try:
        #     est += sum_ratings / sum_sim
        raw_item_id = self.trainset.to_raw_iid(i)
        raw_uid = self.trainset.to_raw_uid(u)
        result = np.dot(self.list_of_cats[raw_item_id], self.taste_score_data[raw_uid])

        # result = random.uniform(0.064140899, 0.28747576886)




        # if (self.user_mean[raw_uid] < self.trainset.global_mean) :
        #     # tmp += bias
        #     est = est - (est * result)
        # else :
        #     # tmp -= bias
        #     est = est + (est * result)
        est = est + (est * result)
        # k = 0.15
        # if (result > k) :
        #     est = est - (est * result)
        # if (result < k) :
        #     est = est + (est * result)
        # except ZeroDivisionError:
        #     pass  # return mean

        details = {'actual_k': actual_k,'total neighbor weigth_score':result}
        #add list: error, result, user's taste ,movie's cate
        return est, details

