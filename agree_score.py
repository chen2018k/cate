from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from six import get_unbound_function as guf
from surprise import Dataset
import pandas as pd
import numpy as np
from surprise.model_selection import KFold
import numpy as np
from six import iteritems
import heapq
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise import *
from surprise.model_selection import train_test_split
import copy

from surprise.prediction_algorithms.optimize_baselines import baseline_als
from surprise.prediction_algorithms.optimize_baselines import baseline_sgd
from surprise import similarities as sims


from collections import defaultdict
import os
import matplotlib
import plotly

file_path_save_data = 'data/processed/'  # don't forget to create this folder before running the scrypt
datasetname = 'ml-100k'  # valid datasetnames are 'ml-latest-small', 'ml-20m', and 'jester'
data1 = Dataset.load_builtin(datasetname)
       
path = '../ml-100k/u.item'
df = pd.read_csv(path, sep="|", encoding="iso-8859-1", names=['id','name','date','space','url','cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18','cat19'])
list_of_cats = {}

df1 = df[['id','cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18','cat19']]
for row in df.itertuples(index=True, name='Pandas'):
    id = str(getattr(row, "id"))
    cate_x = [getattr(row, "cat1"),getattr(row, "cat2"),getattr(row, "cat3"),getattr(row, "cat4"),getattr(row, "cat5"),getattr(row, "cat6"),getattr(row, "cat7"),getattr(row, "cat8"),getattr(row, "cat9"),getattr(row, "cat10"),getattr(row, "cat11"),getattr(row, "cat12"),getattr(row, "cat13"),getattr(row, "cat14"),getattr(row, "cat15"),getattr(row, "cat16"),getattr(row, "cat17"),getattr(row, "cat18"),getattr(row, "cat19"),]
    list_of_cats[id] = cate_x

def into_rate(cate,rate):
    for index in range(len(cate)):
        if cate[index] == 1:
            cate[index] = rate
    return cate

def dist(vec1,vec2,common_number):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dist = np.sqrt(np.sum(np.square(vec1 - vec2))/common_number)
    return dist

def common_number(vec1,vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    i = 0
    common_set_length = 0
    while (i < len(vec1)):
        a_val = vec1[i]
        b_val = vec2[i]

        if a_val != 0 and b_val != 0:
            common_set_length += 1
        i += 1

    return common_set_length



from collections import namedtuple


class PredictionImpossible(Exception):
    """Exception raised when a prediction is impossible.

    When raised, the estimation :math:`\hat{r}_{ui}` is set to the global mean
    of all ratings :math:`\mu`.
    """

    pass


class Prediction(namedtuple('Prediction',
                            ['uid', 'iid', 'r_ui', 'est', 'details'])):
    """A named tuple for storing the results of a prediction.

    It's wrapped in a class, but only for documentation and printing purposes.

    Args:
        uid: The (raw) user id. See :ref:`this note<raw_inner_note>`.
        iid: The (raw) item id. See :ref:`this note<raw_inner_note>`.
        r_ui(float): The true rating :math:`r_{ui}`.
        est(float): The estimated rating :math:`\\hat{r}_{ui}`.
        details (dict): Stores additional details about the prediction that
            might be useful for later analysis.
    """

    __slots__ = ()  # for memory saving purpose.

    def __str__(self):
        s = 'user: {uid:<10} '.format(uid=self.uid)
        s += 'item: {iid:<10} '.format(iid=self.iid)
        if self.r_ui is not None:
            s += 'r_ui = {r_ui:1.2f}   '.format(r_ui=self.r_ui)
        else:
            s += 'r_ui = None   '
        s += 'est = {est:1.2f}   '.format(est=self.est)
        s += str(self.details)

        return s

class AlgoBase(object):
    """Abstract class where is defined the basic behavior of a prediction
    algorithm.

    Keyword Args:
        baseline_options(dict, optional): If the algorithm needs to compute a
            baseline estimate, the ``baseline_options`` parameter is used to
            configure how they are computed. See
            :ref:`baseline_estimates_configuration` for usage.
    """

    def __init__(self, **kwargs):

        self.bsl_options = kwargs.get('bsl_options', {})
        self.sim_options = kwargs.get('sim_options', {})
        if 'user_based' not in self.sim_options:
            self.sim_options['user_based'] = True
        self.skip_train = False

        if (guf(self.__class__.fit) is guf(AlgoBase.fit) and
           guf(self.__class__.train) is not guf(AlgoBase.train)):
            warnings.warn('It looks like this algorithm (' +
                          str(self.__class__) +
                          ') implements train() '
                          'instead of fit(): train() is deprecated, '
                          'please use fit() instead.', UserWarning)

    def train(self, trainset):
        '''Deprecated method: use :meth:`fit() <AlgoBase.fit>`
        instead.'''

        warnings.warn('train() is deprecated. Use fit() instead', UserWarning)

        self.skip_train = True
        self.fit(trainset)

        return self

    def fit(self, trainset):
        """Train an algorithm on a given training set.

        This method is called by every derived class as the first basic step
        for training an algorithm. It basically just initializes some internal
        structures and set the self.trainset attribute.

        Args:
            trainset(:obj:`Trainset <surprise.Trainset>`) : A training
                set, as returned by the :meth:`folds
                <surprise.dataset.Dataset.folds>` method.

        Returns:
            self
        """

        # Check if train method is overridden: this means the object is an old
        # style algo (new algo only have fit() so self.__class__.train will be
        # AlgoBase.train). If true, there are 2 possible cases:
        # - algo.fit() was called. In this case algo.train() was skipped which
        #   is bad. We call it and skip this part next time we enter fit().
        #   Then return immediatly because fit() has already been called by
        #   AlgoBase.train() (which has been called by algo.train()).
        # - algo.train() was called, which is the old way. In that case,
        #   the skip flag will ignore this.
        # This is fairly ugly and hacky but I did not find anything better so
        # far, in order to maintain backward compatibility... See
        # tests/test_train2fit.py for supported cases.
        if (guf(self.__class__.train) is not guf(AlgoBase.train) and
                not self.skip_train):
            self.train(trainset)
            return
        self.skip_train = False

        self.trainset = trainset

        # (re) Initialise baselines
        self.bu = self.bi = None

        return self

    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False, sum = 0):
        """Compute the rating prediction for given user and item.

        The ``predict`` method converts raw ids to inner ids and then calls the
        ``estimate`` method which is defined in every derived class. If the
        prediction is impossible (e.g. because the user and/or the item is
        unkown), the prediction is set according to :meth:`default_prediction()
        <surprise.prediction_algorithms.algo_base.AlgoBase.default_prediction>`.

        Args:
            uid: (Raw) id of the user. See :ref:`this note<raw_inner_note>`.
            iid: (Raw) id of the item. See :ref:`this note<raw_inner_note>`.
            r_ui(float): The true rating :math:`r_{ui}`. Optional, default is
                ``None``.
            clip(bool): Whether to clip the estimation into the rating scale,
                that was set during dataset creation. For example, if
                :math:`\\hat{r}_{ui}` is :math:`5.5` while the rating scale is
                :math:`[1, 5]`, then :math:`\\hat{r}_{ui}` is set to :math:`5`.
                Same goes if :math:`\\hat{r}_{ui} < 1`.  Default is ``True``.
            verbose(bool): Whether to print details of the prediction.  Default
                is False.

        Returns:
            A :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>` object
            containing:

            - The (raw) user id ``uid``.
            - The (raw) item id ``iid``.
            - The true rating ``r_ui`` (:math:`\\hat{r}_{ui}`).
            - The estimated rating (:math:`\\hat{r}_{ui}`).
            - Some additional details about the prediction that might be useful
              for later analysis.
        """

        # Convert raw ids to inner ids
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = 'UKN__' + str(iid)

        details = {}
        try:
            est = self.estimate(iuid, iiid)

            # If the details dict was also returned
            if isinstance(est, tuple):
                est, details = est

            details['was_impossible'] = False

        except PredictionImpossible as e:
            est = self.default_prediction()
            details['was_impossible'] = True
            details['r_ui'] = r_ui
            details['error1'] = 0
            details['error2'] = abs(est - r_ui)
            tmp = est
            details['predict'] = tmp
            if (int(iid) < 1645) :
                bias = np.dot(self.list_of_cats[int(iiid[5:])], self.taste_score_data[iuid]) * 10
                if (user_mean[int(uid)] < 3.53) :
                    tmp += bias
                else :
                    tmp -= bias

                details['predict'] = tmp
                details['error1'] = abs(r_ui - tmp)
                est = tmp




        # clip estimate into [lower_ bound, higher_bound]
        if clip:
            lower_bound, higher_bound = self.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)

        pred = Prediction(uid, iid, r_ui, est, details)

        if verbose:
            print(pred)

        return pred

    def default_prediction(self):
        '''Used when the ``PredictionImpossible`` exception is raised during a
        call to :meth:`predict()
        <surprise.prediction_algorithms.algo_base.AlgoBase.predict>`. By
        default, return the global mean of all ratings (can be overridden in
        child classes).

        Returns:
            (float): The mean of all ratings in the trainset.
        '''

        return self.trainset.global_mean

    def test(self, testset, verbose=False):
        """Test the algorithm on given testset, i.e. estimate all the ratings
        in the given testset.

        Args:
            testset: A test set, as returned by a :ref:`cross-validation
                itertor<use_cross_validation_iterators>` or by the
                :meth:`build_testset() <surprise.Trainset.build_testset>`
                method.
            verbose(bool): Whether to print details for each predictions.
                Default is False.

        Returns:
            A list of :class:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>` objects
            that contains all the estimated ratings.
        """

        # The ratings are translated back to their original scale.
        predictions = [self.predict(uid,
                                    iid,
                                    r_ui_trans,
                                    verbose=verbose)
                       for (uid, iid, r_ui_trans) in testset]
        return predictions

    def compute_baselines(self, verbose=False):
        """Compute users and items baselines.

        The way baselines are computed depends on the ``bsl_options`` parameter
        passed at the creation of the algorithm (see
        :ref:`baseline_estimates_configuration`).

        This method is only relevant for algorithms using :func:`Pearson
        baseline similarty<surprise.similarities.pearson_baseline>` or the
        :class:`BaselineOnly
        <surprise.prediction_algorithms.baseline_only.BaselineOnly>` algorithm.

        Args:
            verbose(bool) : if ``True``, print status message. Default is
                ``False``.

        Returns:
            A tuple ``(bu, bi)``, which are users and items baselines."""

        # Firt of, if this method has already been called before on the same
        # trainset, then just return. Indeed, compute_baselines may be called
        # more than one time, for example when a similarity metric (e.g.
        # pearson_baseline) uses baseline estimates.
        if self.bu is not None:
            return self.bu, self.bi

        method = dict(als=baseline_als,
                      sgd=baseline_sgd)

        method_name = self.bsl_options.get('method', 'als')

        try:
            if verbose:
                print('Estimating biases using', method_name + '...')
            self.bu, self.bi = method[method_name](self)
            return self.bu, self.bi
        except KeyError:
            raise ValueError('Invalid method ' + method_name +
                             ' for baseline computation.' +
                             ' Available methods are als and sgd.')

    def compute_similarities(self, verbose=False):
        """Build the similarity matrix.

        The way the similarity matrix is computed depends on the
        ``sim_options`` parameter passed at the creation of the algorithm (see
        :ref:`similarity_measures_configuration`).

        This method is only relevant for algorithms using a similarity measure,
        such as the :ref:`k-NN algorithms <pred_package_knn_inpired>`.

        Args:
            verbose(bool) : if ``True``, print status message. Default is
                ``False``.

        Returns:
            The similarity matrix."""

        construction_func = {'cosine': sims.cosine,
                             'msd': sims.msd,
                             'pearson': sims.pearson,
                             'pearson_baseline': sims.pearson_baseline,
                             }

        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        min_support = self.sim_options.get('min_support', 1)

        args = [n_x, yr, min_support]

        name = self.sim_options.get('name', 'msd').lower()
        if name == 'pearson_baseline':
            shrinkage = self.sim_options.get('shrinkage', 100)
            bu, bi = self.compute_baselines()
            if self.sim_options['user_based']:
                bx, by = bu, bi
            else:
                bx, by = bi, bu

            args += [self.trainset.global_mean, bx, by, shrinkage]
        if name == 'agreement':
                trainset = self.sim_options['trainset']
                beta = self.sim_options['beta']
                epsilon = self.sim_options['epsilon']
                args = [trainset, beta, epsilon]
                sim = construction_func[name](*args)
                return sim
        try:
            if verbose:
                print('before similarity calculation')
                print('Computing the {0} similarity matrix...'.format(name))
            # sim = Parallel(n_jobs=2)(delayed(construction_func[name])(*args) for i in range(10))
            if name == 'agreement':
                trainset = self.sim_options['trainset']
                beta = self.sim_options['beta']
                epsilon = self.sim_options['epsilon']
                args = [trainset, beta, epsilon]
                sim = construction_func[name](*args)
                return sim
            else:
                sim = construction_func[name](*args)
            if verbose:
                print('Done computing similarity matrix.')
            return sim
        except KeyError:
            raise NameError('Wrong sim name ' + name + '. Allowed values ' +
                            'are ' + ', '.join(construction_func.keys()) + '.')

    def get_neighbors(self, iid, k):
        """Return the ``k`` nearest neighbors of ``iid``, which is the inner id
        of a user or an item, depending on the ``user_based`` field of
        ``sim_options`` (see :ref:`similarity_measures_configuration`).

        As the similarities are computed on the basis of a similarity measure,
        this method is only relevant for algorithms using a similarity measure,
        such as the :ref:`k-NN algorithms <pred_package_knn_inpired>`.

        For a usage example, see the :ref:`FAQ <get_k_nearest_neighbors>`.

        Args:
            iid(int): The (inner) id of the user (or item) for which we want
                the nearest neighbors. See :ref:`this note<raw_inner_note>`.

            k(int): The number of neighbors to retrieve.

        Returns:
            The list of the ``k`` (inner) ids of the closest users (or items)
            to ``iid``.
        """

        if self.sim_options['user_based']:
            all_instances = self.trainset.all_users
        else:
            all_instances = self.trainset.all_items

        others = [(x, self.sim[iid, x]) for x in all_instances() if x != iid]
        others.sort(key=lambda tple: tple[1], reverse=True)
        k_nearest_neighbors = [j for (j, _) in others[:k]]

        return k_nearest_neighbors

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


    def __init__(self,user_mean,list_of_cats,taste_score_data,base_line=False, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):


        SymmetricAlgo.__init__(self, sim_options=sim_options,
                               verbose=verbose, **kwargs)

        self.k = k
        self.min_k = min_k
        self.list_of_cats = list_of_cats
        self.taste_score_data = taste_score_data
        self.base_line = base_line
        self.user_mean = user_mean



    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)

        self.sim = self.compute_similarities(verbose=self.verbose)

        self.means = np.zeros(self.n_x)
        for x, ratings in iteritems(self.xr):
            self.means[x] = np.mean([r for (_, r) in ratings])

        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            if (int(i[5:]) > 1645) :
                raise PredictionImpossible('User and/or item is unkown.')
            else:
                raise PredictionImpossible('weight score is ')

        x, y = self.switch(u, i)
        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        #k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])
        k_neighbors = neighbors[:self.k]

        # if user_based == False then:
            # x = i
            # y = u

        est = self.means[x]

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
            side_info = {}
            # if user_based == False then:
                # nb = item_inner_id
            if sim > 0:

                if self.base_line:
                    # sim += 0.1005
                    sum_sim += sim
                    sum_ratings += (r - self.means[nb]) * sim
                    #sum_ratings += (r) * sim
                    actual_k += 1
                else:
                    if self.sim_options['user_based'] == True:
                        result = np.dot(self.list_of_cats[i], self.taste_score_data[nb])
                    else:
                        result = np.dot(self.list_of_cats[nb], self.taste_score_data[y]) # y is the user in the item_based

                    # result += 0.3
                    sum_sim += result
                    sum_ratings += (r - self.means[nb]) * result
                    #sum_ratings += r * result * 100 + 121212
                    actual_k += 1

        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est += sum_ratings / sum_sim
            #est = result
        except ZeroDivisionError:
            pass  # return mean

        test = np.dot(self.list_of_cats[y], self.taste_score_data[x])
        details = {'actual_k': actual_k,'weigth_score':test}
        #add list: error, result, user's taste ,movie's cate
        return est, details


t_mae = 0
t_rmse = 0
k = 5
kf = KFold(n_splits=k, random_state=100)


for trainset, testset in kf.split(data1):
    taste_score_data = {}
    items_taste_score_data = {}
    user_mean = {}
    sim_taste = []
#compute each user's taste (by this user's history item list)
    for user, item_list in trainset.ur.items():
        user_rating = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for item_inner_id,rating in item_list:
            raw_item_id = trainset.to_raw_iid(item_inner_id)

            itemx_cate = copy.deepcopy(list_of_cats[raw_item_id])

            itemx_rating = into_rate(itemx_cate, rating)

            user_rating = [a + b for a, b in zip(user_rating, itemx_rating)]

    #get the proportion of each cate

        user_rating = (user_rating / np.sum(user_rating)).tolist()
        taste_score_data[user] = user_rating

# compute each user's mean rating
    for user, item_list in trainset.ur.items():
        sum = 0
        count = 0
        for item_inner_id,rating in item_list:
            sum += rating
            count += 1
        tmp = sum / count
        user_mean[user] = tmp


    #get the proportion of each cate


#compute each item's cate (by its used user' rating and user's taste)
    for item, user_list in trainset.ir.items():
        user_ratings = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for user, rating in user_list:
            weighted_rating = [taste * rating for taste in taste_score_data[user]]
            user_ratings = [a + b for a, b in zip(weighted_rating, user_ratings)]

    # get the proportion of each cate
        itemx_rating = (user_ratings/np.sum(user_ratings)).tolist()
        items_taste_score_data[item] = itemx_rating

    # print(items_taste_score_data['1348'])




#---------------------------------main part---------------------------------#

    user_based = True  # changed to False to do item-absed CF

    #for normal case-------------------------------------------------------#
    sim_options = {'name': 'pearson', 'user_based': user_based}
    algo = KNNWithMeans(user_mean,items_taste_score_data,
                        taste_score_data,
                        sim_options=sim_options)

    #for agreement case
    # epsilon = 1
    # lambdak = 0.5
    # beta = 2.5
    # sim_options = {'name': 'agreement',
    #                'user_based': user_based,
    #                'trainset':trainset,
    #                'beta':beta,
    #                'epsilon':epsilon,
    #                'lambdak':lambdak}
    #
    # algo = KNNWithMeans(items_taste_score_data,
    #                     taste_score_data,
    #                     base_line=False,
    #                     sim_options=sim_options)

    algo.fit(trainset)
    predictions = algo.test(testset)

# Then compute RMSE
    t_rmse += accuracy.rmse(predictions)
    t_mae += accuracy.mae(predictions)

    # print(len(predictions))
    # for i in range(500):
    #     print(predictions[i])
    list = []
    for item in predictions:
        if (item[4]["was_impossible"] == True) :
            list.append(item)
    print(len(list))
    for i in range(len(list)):
        print(list[i])


#---------------------------------show result---------------------------------#

print("\nMEAN_RMSE:" + str(t_rmse/k))
print("MEAN_MAE:" + str(t_mae/k))


