import util
import proposedmethods
from surprise import KNNWithMeans
from surprise import KNNBasic
from surprise.model_selection import KFold
import copy
import numpy as np
import pandas as pd
from surprise.model_selection import GridSearchCV


import random

# my_seed = 100
# random.seed(my_seed)
# np.random.seed(my_seed)

from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import accuracy



file_path_save_data = util.file_path_save_data
datasetname = util.datasetname
data1 = util.data1
list_of_cats = util.list_of_cats


print(len(data1.user_weight))
print(len(data1.all_user_means))
print(len(data1.item_weight))

# trainset, testset = train_test_split(data1, test_size=.25)

# # # # Run 5-fold cross-validation and print results

# <<<<<<< HEAD
# user_based = True  # changed to False to do item-absed CF
# sim_options = {'name': 'pearson', 'user_based': user_based}
#
#
#
# algo = knnwithmeans.KNNWithMeans(data1, sim_options=sim_options)
#
# # algo = KNNWithMeans(sim_options=sim_options)
# # algo = KNNBasic(sim_options=sim_options)
# cross_validate(algo, data1, measures=['RMSE', 'MAE'], cv=5, verbose=True)
#
# =======

algo = proposedmethods.OurMethod()
# algo = KNNWithMeans(sim_options = {'name': 'pearson', 'user_based': True})
# algo = KNNBasic(sim_options = {'name': 'pearson', 'user_based': True})
# cross_validate(algo, data1, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# >>>>>>> ea184beb873c9d3dcdb773ca13bf52851a00892c
# # # Train the algorithm on the trainset, and predict ratings for the testset
# algo.fit(trainset)
# predictions = algo.test(testset)
# print(predictions)
# list = []
# for item in predictions:
#     print(item[4]['was_impossible'])
#     if (item[4]['was_impossible'] == True) :
#         list.append(item)
#         print(item)
#
# for i in range(len(list)):
#     print(list[i])
# exit()
# # # # Then compute RMSE
# accuracy.rmse(predictions)
# accuracy.mae(predictions)

import multiprocessing
num_cores = multiprocessing.cpu_count()
#param_grid = {'k':[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]}
param_grid = {'k':[0]}
gs = GridSearchCV(proposedmethods.OurMethod, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1)
gs.fit(data1)

# best RMSE score
print(gs.best_score['rmse'])

# # combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

print(gs.best_score['mae'])
print(gs.best_params['mae'])


# df = pd.DataFrame(predictions, columns=['user_id', 'item_id', 'ratings_ui', 'estimated_Ratings', 'details'])
# print(df)
# df.to_csv(file_path_save_data+'Predictions_'+datasetname+'.csv', sep='\t', encoding='utf-8')

# # # For KNNBASIC
# kf = KFold(n_splits=5, random_state = 100)
#
# algo1 = KNNBasic(sim_options=sim_options)
#
# t_rmse = 0
# t_mae = 0
# k = 5
#
# for trainset, testset in kf.split(data1):
#
#     # train and test algorithm.
#     algo1.fit(trainset)
#     predictions = algo1.test(testset)
#
#     # Compute and print Root Mean Squared Error
#     t_rmse += accuracy.rmse(predictions)
#     t_mae += accuracy.mae(predictions)
# print('for msd')
# print("\nMEAN_RMSE:" + str(t_rmse / k))
# print("MEAN_MAE:" + str(t_mae / k))
       

