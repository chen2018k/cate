from algobase import AlgoBase
from prediction import PredictionImpossible
import numpy as np
from six import iteritems
import heapq


class OurMethod(AlgoBase):


    def __init__(self, k=0.05, verbose=True, **kwargs):


        AlgoBase.__init__(self, **kwargs)
        self.verbose = verbose

        self.k = k
        # self.list_of_cats = dataset.item_weight
        # self.taste_score_data = dataset.user_weight
        # self.user_mean = dataset.all_user_means
        



    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.list_of_cats = trainset.item_weight
        self.taste_score_data = trainset.user_weight
        self.user_mean = trainset.all_user_means
        
        return self

    def estimate(self, u, i):
        details = {}
        details['was_impossible'] = False
        raw_uid = self.trainset.to_raw_uid(u)
        est = self.user_mean[raw_uid]
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            details['was_impossible'] = True
            # details['default predict1'] = self.default_prediction()
            details['mean predict'] = self.user_mean[raw_uid]
            item_weight = self.list_of_cats[i]
        else:
           item_weight = self.list_of_cats[self.trainset.to_raw_iid(i)]

        result = np.dot(item_weight, self.taste_score_data[raw_uid])

        # if (est < self.trainset.global_mean) :
        #     est = est + est * (result - 0.05)
        # else :
        est = est + est * (result + self.k)
        
        details['total neighbor weigth_score'] = result
        #add list: error, result, user's taste ,movie's cate
        return est, details