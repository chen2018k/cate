"""
The :mod:`dataset <surprise.dataset>` module defines the :class:`Dataset` class
and other subclasses which are used for managing datasets.

Users may use both *built-in* and user-defined datasets (see the
:ref:`getting_started` page for examples). Right now, three built-in datasets
are available:

* The `movielens-100k <http://grouplens.org/datasets/movielens/>`_ dataset.
* The `movielens-1m <http://grouplens.org/datasets/movielens/>`_ dataset.
* The `Jester <http://eigentaste.berkeley.edu/dataset/>`_ dataset 2.

Built-in datasets can all be loaded (or downloaded if you haven't already)
using the :meth:`Dataset.load_builtin` method.
Summary:

.. autosummary::
    :nosignatures:

    Dataset.load_builtin
    Dataset.load_from_file
    Dataset.load_from_folds
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import sys
import os
import itertools

from six.moves import input

from surprise.reader import Reader
from builtin_datasets import download_builtin_dataset
from builtin_datasets import BUILTIN_DATASETS
from surprise.trainset import Trainset
import pandas as pd
import copy
import numpy as np
from six import iteritems

def convert(listx):
    #listx : ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy']
    list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if '(no genres listed)' in listx:
        list[0] = 1
    if 'Action' in listx:
        list[1] = 1
    if 'Adventure' in listx:
        list[2] = 1
    if 'Animation' in listx:
        list[3] = 1
    if 'Children' in listx:
        list[4] = 1
    if 'Comedy' in listx:
        list[5] = 1
    if 'Crime' in listx:
        list[6] = 1
    if 'Documentary' in listx:
        list[7] = 1
    if 'Drama' in listx:
        list[8] = 1
    if 'Fantasy' in listx:
        list[9] = 1
    if 'Film-Noir' in listx:
        list[10] = 1
    if 'Horror' in listx:
        list[11] = 1
    if 'Musical' in listx:
        list[12] = 1
    if 'Mystery' in listx:
        list[13] = 1
    if 'Romance' in listx:
        list[14] = 1
    if 'Sci-Fi' in listx:
        list[15] = 1
    if 'Thriller' in listx:
        list[16] = 1
    if 'War' in listx:
        list[17] = 1
    if 'Western' in listx:
        list[18] = 1
    return list


class Dataset:

    """Base class for loading datasets.

    Note that you should never instantiate the :class:`Dataset` class directly
    (same goes for its derived classes), but instead use one of the three
    available methods for loading datasets."""

    def __init__(self, reader):

        self.reader = reader

    @classmethod
    def load_builtin(cls, name='ml-100k', prompt=True):
        """Load a built-in dataset.

        If the dataset has not already been loaded, it will be downloaded and
        saved. You will have to split your dataset using the :meth:`split
        <DatasetAutoFolds.split>` method. See an example in the :ref:`User
        Guide <cross_validate_example>`.

        Args:
            name(:obj:`string`): The name of the built-in dataset to load.
                Accepted values are 'ml-100k', 'ml-1m', and 'jester'.
                Default is 'ml-100k'.
            prompt(:obj:`bool`): Prompt before downloading if dataset is not
                already on disk.
                Default is True.

        Returns:
            A :obj:`Dataset` object.

        Raises:
            ValueError: If the ``name`` parameter is incorrect.
        """

        try:
            dataset = BUILTIN_DATASETS[name]
        except KeyError:
            raise ValueError('unknown dataset ' + name +
                             '. Accepted values are ' +
                             ', '.join(BUILTIN_DATASETS.keys()) + '.')

        # if dataset does not exist, offer to download it
        if not os.path.isfile(dataset.path):
            answered = not prompt
            while not answered:
                print('Dataset ' + name + ' could not be found. Do you want '
                      'to download it? [Y/n] ', end='')
                choice = input().lower()

                if choice in ['yes', 'y', '', 'omg this is so nice of you!!']:
                    answered = True
                elif choice in ['no', 'n', 'hell no why would i want that?!']:
                    answered = True
                    print("Ok then, I'm out!")
                    sys.exit()

            download_builtin_dataset(name)

        reader = Reader(**dataset.reader_params)

        return cls.load_from_file(file_path=dataset.path, reader=reader, item_path=dataset.item_path)

    @classmethod
    def load_from_file(cls, file_path, reader, item_path=None):
        """Load a dataset from a (custom) file.

        Use this if you want to use a custom dataset and all of the ratings are
        stored in one file. You will have to split your dataset using the
        :meth:`split <DatasetAutoFolds.split>` method. See an example in the
        :ref:`User Guide <load_from_file_example>`.


        Args:
            file_path(:obj:`string`): The path to the file containing ratings.
            reader(:obj:`Reader <surprise.reader.Reader>`): A reader to read
                the file.
        """
        if item_path is not None:
            df, category_list = cls.read_items_genre(item_path=item_path)
            return DatasetAutoFolds(ratings_file=file_path, reader=reader, item_df=df, category_list=category_list)
        return DatasetAutoFolds(ratings_file=file_path, reader=reader)

    @classmethod
    def load_from_folds(cls, folds_files, reader):
        """Load a dataset where folds (for cross-validation) are predefined by
        some files.

        The purpose of this method is to cover a common use case where a
        dataset is already split into predefined folds, such as the
        movielens-100k dataset which defines files u1.base, u1.test, u2.base,
        u2.test, etc... It can also be used when you don't want to perform
        cross-validation but still want to specify your training and testing
        data (which comes down to 1-fold cross-validation anyway). See an
        example in the :ref:`User Guide <load_from_folds_example>`.


        Args:
            folds_files(:obj:`iterable` of :obj:`tuples`): The list of the
                folds. A fold is a tuple of the form ``(path_to_train_file,
                path_to_test_file)``.
            reader(:obj:`Reader <surprise.reader.Reader>`): A reader to read
                the files.

        """

        return DatasetUserFolds(folds_files=folds_files, reader=reader)

    @classmethod
    def load_from_df(cls, df, reader):
        """Load a dataset from a pandas dataframe.

        Use this if you want to use a custom dataset that is stored in a pandas
        dataframe. See the :ref:`User Guide<load_from_df_example>` for an
        example.

        Args:
            df(`Dataframe`): The dataframe containing the ratings. It must have
                three columns, corresponding to the user (raw) ids, the item
                (raw) ids, and the ratings, in this order.
            reader(:obj:`Reader <surprise.reader.Reader>`): A reader to read
                the file. Only the ``rating_scale`` field needs to be
                specified.
        """

        return DatasetAutoFolds(reader=reader, df=df)

    @classmethod
    def read_items_genre(self, item_path):
        if "100k" in item_path:
            print('reading ml-100k genres')
            df = pd.read_csv(item_path, sep="|", encoding="iso-8859-1", names=['id','name','date','space','url','cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18','cat19'])
            category_list = {}
            df1 = df[['id','cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18','cat19']]
            for row in df.itertuples(index=True, name='Pandas'):
                id = str(getattr(row, "id"))
                cate_x = [getattr(row, "cat1"),getattr(row, "cat2"),getattr(row, "cat3"),getattr(row, "cat4"),getattr(row, "cat5"),getattr(row, "cat6"),getattr(row, "cat7"),getattr(row, "cat8"),getattr(row, "cat9"),getattr(row, "cat10"),getattr(row, "cat11"),getattr(row, "cat12"),getattr(row, "cat13"),getattr(row, "cat14"),getattr(row, "cat15"),getattr(row, "cat16"),getattr(row, "cat17"),getattr(row, "cat18"),getattr(row, "cat19"),]
                category_list[id] = cate_x
            return df1, category_list
        if "small" in item_path:
            print('reading ml-latest-small genres')
            df = pd.read_csv(item_path, sep=",", encoding="iso-8859-1", names=['id','name','CATE'])
            df1 = df[['id','CATE']]
            category_list = {}
            for row in df.itertuples(index=True, name='Pandas'):
                id = str(getattr(row, "id"))
                cate = getattr(row, "CATE")
                # change cate(str) into the list of cate(list)
                cate = cate.split('|')
                # set a map function from cate(word) to cate(0or1)
                cate = convert(cate)
                category_list[id] = cate

            return df1, category_list
        if "1m" in item_path:
            print('reading ml-1m genres')
            df = pd.read_csv(item_path, sep="::", engine='python', encoding="iso-8859-1", names=['id', 'name', 'CATE'])
            df1 = df[['id', 'CATE']]
            category_list = {}
            for row in df.itertuples(index=True, name='Pandas'):
                id = str(getattr(row, "id"))
                cate = getattr(row, "CATE")
                # change cate(str) into the list of cate(list)
                cate = cate.split('|')
                # set a map function from cate(word) to cate(0or1)
                cate = convert(cate)
                category_list[id] = cate
            return df1, category_list



    def get_allusers_weights_and_means(self):
        print('getting allusers weights')
        trainset = self.build_full_trainset()
        taste_score_data = {}
        items_taste_score_data = {}
        all_user_means = {}
    #compute each user's taste (by this user's history item list)
        for user, item_list in trainset.ur.items():
            raw_uid = trainset.to_raw_uid(user)
            user_rating = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for item_inner_id,rating in item_list:
                raw_item_id = trainset.to_raw_iid(item_inner_id)
                itemx_cate = copy.deepcopy(self.category_list[raw_item_id])
                for index in range(len(itemx_cate)):
                    if itemx_cate[index] == 1:
                        itemx_cate[index] = rating
                user_rating = [a + b for a, b in zip(user_rating, itemx_cate)]
            user_rating = (user_rating / np.sum(user_rating)).tolist()
            taste_score_data[raw_uid] = user_rating #This is the first mistake inner id was used

        #compute each item's cate (by its used user' rating and user's taste)
        for item, user_list in trainset.ir.items():
            user_ratings = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            raw_item_id = trainset.to_raw_iid(item)
            for user, rating in user_list:
                weighted_rating = [taste * rating for taste in taste_score_data[trainset.to_raw_uid(user)]]
                user_ratings = [a + b for a, b in zip(weighted_rating, user_ratings)]

        # get the proportion of each cate
            itemx_rating = (user_ratings/np.sum(user_ratings)).tolist()
            items_taste_score_data[raw_item_id] = itemx_rating

        #it will only work for users
        for x, ratings in iteritems(trainset.ur):
            raw_uid = trainset.to_raw_uid(x)
            all_user_means[raw_uid] = np.mean([r for (_, r) in ratings])
        return taste_score_data, items_taste_score_data, all_user_means


    def read_ratings(self, file_name):
        """Return a list of ratings (user, item, rating, timestamp) read from
        file_name"""

        with open(os.path.expanduser(file_name)) as f:
            raw_ratings = [self.reader.parse_line(line) for line in
                           itertools.islice(f, self.reader.skip_lines, None)]
        return raw_ratings

    def construct_trainset(self, raw_trainset):

        raw2inner_id_users = {}
        raw2inner_id_items = {}

        current_u_index = 0
        current_i_index = 0

        ur = defaultdict(list)
        ir = defaultdict(list)

        # user raw id, item raw id, translated rating, time stamp
        for urid, irid, r, timestamp in raw_trainset:
            try:
                uid = raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                raw2inner_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                raw2inner_id_items[irid] = current_i_index
                current_i_index += 1

            ur[uid].append((iid, r))
            ir[iid].append((uid, r))

        n_users = len(ur)  # number of users
        n_items = len(ir)  # number of items
        n_ratings = len(raw_trainset)

        trainset = Trainset(ur,
                            ir,
                            n_users,
                            n_items,
                            n_ratings,
                            self.reader.rating_scale,
                            raw2inner_id_users,
                            raw2inner_id_items)

        if self.item_weight is not None:
            trainset.user_weight = self.user_weight
            trainset.item_weight = self.item_weight 
            trainset.all_user_means =  self.all_user_means
        return trainset

    def construct_testset(self, raw_testset):

        return [(ruid, riid, r_ui_trans)
                for (ruid, riid, r_ui_trans, _) in raw_testset]


class DatasetUserFolds(Dataset):
    """A derived class from :class:`Dataset` for which folds (for
    cross-validation) are predefined."""

    def __init__(self, folds_files=None, reader=None):

        Dataset.__init__(self, reader)
        self.folds_files = folds_files

        # check that all files actually exist.
        for train_test_files in self.folds_files:
            for f in train_test_files:
                if not os.path.isfile(os.path.expanduser(f)):
                    raise ValueError('File ' + str(f) + ' does not exist.')


class DatasetAutoFolds(Dataset):
    """A derived class from :class:`Dataset` for which folds (for
    cross-validation) are not predefined. (Or for when there are no folds at
    all)."""

    def __init__(self, ratings_file=None, reader=None, df=None, item_df=None, category_list=None):

        Dataset.__init__(self, reader)
        self.has_been_split = False  # flag indicating if split() was called.
        self.user_weight = None
        self.item_weight = None
        self.all_user_means = None

        if ratings_file is not None:
            self.ratings_file = ratings_file
            self.raw_ratings = self.read_ratings(self.ratings_file)
            if item_df is not None:
                self.item_df = item_df
            if category_list is not None:
                self.category_list = category_list
            
            user_weight, item_weight, all_user_means = self.get_allusers_weights_and_means()
            self.user_weight = user_weight
            self.item_weight = item_weight
            self.all_user_means = all_user_means
        elif df is not None:
            self.df = df
            self.raw_ratings = [(uid, iid, float(r), None)
                                for (uid, iid, r) in
                                self.df.itertuples(index=False)]
            if item_df is not None:
                self.item_df = item_df
            if category_list is not None:
                self.category_list = category_list
        else:
            raise ValueError('Must specify ratings file or dataframe.')

    def build_full_trainset(self):
        """Do not split the dataset into folds and just return a trainset as
        is, built from the whole dataset.

        User can then query for predictions, as shown in the :ref:`User Guide
        <train_on_whole_trainset>`.

        Returns:
            The :class:`Trainset <surprise.Trainset>`.
        """

        return self.construct_trainset(self.raw_ratings)
