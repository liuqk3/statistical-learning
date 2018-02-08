# created on Jan. 2018
# @author Qiankun Liu

import numpy as np

def probability(movies, ratings, train_users,fold, mode):
    # this function process the data, compute the table of probability using Naive Bayes
    #
    # input:
    # movies: 2-D array, str type, each row is the information of of a movie, [movieID, title, genres]
    # ratings: 2-D array, str type, each row is a rating on a movie, [userID, movieID, rating, timestamp]
    # train_users: 2-D array, each row is the information of a user, [userID, gender, age, occupation, zip-code]
    # fold: the fold-th cross validation
    # mode: str, 'gender' or 'age', which determines this function to predict the gender or age of test user
    #
    # output:
    # prob: 3-D array, the table of probability using Naive Baye

    if mode == 'gender':
        num_movies = np.shape(movies)[0]
        num_genders = 2
        num_stars = 5
        prob = np.zeros((num_genders,num_stars,num_movies))
        for movie_idx in np.arange(0,num_movies):
            movieID = movies[movie_idx,0]
            rating_tmp_idx = np.where(ratings[:,1] == movieID)[0]#[i for i, ID in enumerate(ratings[:,1]) if ID == movieID]
            rating_tmp = ratings[rating_tmp_idx,:]  # the ratings on this movie
            for rating_idx in np.arange(0,np.shape(rating_tmp)[0]):
                one_rating = rating_tmp[rating_idx,:]
                if one_rating[0] in train_users[:,0]: # if the rating is rated by a user in the training data
                    # find the information of this user
                    user_idx = np.where(train_users[:,0] == one_rating[0])[0][0] # find the index of this user in the training data
                    user_info = train_users[user_idx,:]

                    rate_star = one_rating[2]
                    star_idx = int(rate_star) - 1
                    gender_idx = 0 # Female user
                    if user_info[1] == 'M': # Male user
                        gender_idx = 1

                    prob[gender_idx,star_idx,movie_idx] += 1

            prob[:,:,movie_idx] +=1 # handle missing values
            prob_sum = np.sum(prob[:,:,movie_idx],axis=1)
            prob[:,:,movie_idx] = np.transpose(np.transpose(prob[:,:,movie_idx])/prob_sum)
    elif mode == 'age':
        num_movies = np.shape(movies)[0]
        ages = np.array(['1', '18', '25', '35', '45', '50', '56'])
        num_age = len(ages)
        num_stars = 5
        prob = np.zeros((num_age, num_stars, num_movies))
        for movie_idx in np.arange(0, num_movies):
            movieID = movies[movie_idx, 0]
            rating_tmp_idx = np.where(ratings[:, 1] == movieID)[0]
            rating_tmp = ratings[rating_tmp_idx, :]  # the ratings on this movie
            for rating_idx in np.arange(0, np.shape(rating_tmp)[0]):
                one_rating = rating_tmp[rating_idx, :]
                if one_rating[0] in train_users[:, 0]:  # if the rating is made by a user in the training data
                    # find the information of this user
                    user_idx = np.where(train_users[:, 0] == one_rating[0])[0][0]  # find the index of this user in the training data
                    user_info = train_users[user_idx, :]

                    rate_star = one_rating[2]
                    star_idx = int(rate_star) - 1
                    age_idx = np.where(ages == user_info[2])[0][0]
                    prob[age_idx, star_idx, movie_idx] += 1

            prob[:, :, movie_idx] += 1  # handle missing values
            prob_sum = np.sum(prob[:, :, movie_idx], axis=1)
            prob[:, :, movie_idx] = np.transpose(np.transpose(prob[:, :, movie_idx]) / prob_sum)

    fname = '.\cache\prob_' + mode + '_' + str(fold) + '.npy'
    np.save(fname,prob)
    return prob

def predict(movies, ratings, train_users, test_users, prob, mode):
    # this function return the predicted results
    #
    # inputs:
    # movies: 2-D array, str type, each row is the information of of a movie, [movieID, title, genres]
    # ratings: 2-D array, str type, each row is a rating on a movie, [userID, movieID, rating, timestamp]
    # train_users: 2-D array, each row is the information of a user, [userID, gender, age, occupation, zip-code]
    # test_users: 2-D array, each row is the information of a user, [userID, gender, age, occupation, zip-code]
    # prob: 3-D array, the table of probability computed using function 'probability'
    # mode: str, 'gender' or 'age', which determines this function to predict the gender or age of test user
    #
    # output:
    # prediction: 1-D array, str type, the predicted gender or age of test users

    if mode == 'gender':
        gender = np.array([70, 77])
        num_train_users = np.shape(train_users)[0]
        num_train_f = len(np.where(train_users[:,1] == 'F')[0])
        num_train_m = len(np.where(train_users[:,1] == 'M')[0])

        num_test_users = np.shape(test_users)[0]
        prediction = np.zeros((num_test_users,))
        prediction = prediction.astype(int)
        for user_idx in np.arange(0,num_test_users):
            p_f = 1.0
            p_m = 1.0
            user_info = test_users[user_idx,:]
            rating_tmp_idx = np.where(ratings[:,0] == user_info[0])[0] # the ratings made by this user
            rating_tmp = ratings[rating_tmp_idx,:]
            for rating_idx in np.arange(0,np.shape(rating_tmp)[0]):
                one_rating = rating_tmp[rating_idx,:]
                movie_idx = np.where(movies[:,0] == one_rating[1])[0][0]
                star_idx = int(one_rating[2]) - 1
                p_f = p_f * prob[0,star_idx,movie_idx]
                p_m = p_m * prob[1,star_idx,movie_idx]
            p_f = p_f * num_train_f / num_train_users
            p_m = p_m * num_train_m / num_train_users
            if p_f >= p_m: # female
                prediction[user_idx] = gender[0]
            else: # male
                prediction[user_idx] = gender[1]

        prediction = np.array([chr(prediction[i]) for i, g in enumerate(prediction)]) # convert to char
    elif mode == 'age':
        ages = np.array([1, 18, 25, 35, 45, 50, 56])
        num_train_users = np.shape(train_users)[0]
        num_age = len(ages)
        num_per_age = np.zeros((num_age,))
        for age_idx in np.arange(0,num_age): # find the number of different ages
            num_per_age[age_idx] = len(np.where(train_users[:,2] == str(ages[age_idx])))

        num_test_users = np.shape(test_users)[0]
        prediction = np.zeros((num_test_users,))
        prediction = prediction.astype(int)
        for user_idx in np.arange(0, num_test_users):
            p = np.ones((num_age,))
            user_info = test_users[user_idx, :]
            rating_tmp_idx = np.where(ratings[:, 0] == user_info[0])[0]  # the ratings made by this user
            rating_tmp = ratings[rating_tmp_idx, :]
            for rating_idx in np.arange(0, np.shape(rating_tmp)[0]):
                one_rating = rating_tmp[rating_idx, :]
                movie_idx = np.where(movies[:, 0] == one_rating[1])[0][0]
                star_idx = int(one_rating[2]) - 1

                p = p * prob[:,star_idx, movie_idx]

            p = p * num_per_age / num_train_users
            pre_idx = np.where(p == np.max(p))[0][0]
            prediction[user_idx] = int(ages[pre_idx])

        prediction = np.array([str(prediction[i]) for i, g in enumerate(prediction)])  # convert to string
    return prediction

def error_rate(prediction, ground_truth, mode):
    # this function return the error rate
    #
    # inputs:
    # prediction: 1-D array, str type, the predicted gender or age of test users
    # ground_truth: 1-D array, str type, the ground truth gender or age of test users
    # mode: str, 'gender' or 'age', which determines this function to predict the gender or age of test user
    #
    # output:
    # error: a float type number, the error rate of prediction

    if mode == 'gender':
        num_wrong = 0
        for idx in np.arange(0, len(prediction)):
            if prediction[idx] != ground_truth[idx]:
                num_wrong += 1
        error = num_wrong / float(len(prediction))
    elif mode == 'age':
        ages = np.array(['1', '18', '25', '35', '45', '50', '56'])
        num_wrong = 0
        for idx in np.arange(0, len(prediction)):
            gt_idx = np.where(ages == ground_truth[idx])[0][0]
            pre_idx = np.where(ages == prediction[idx])[0][0]

            # if gt_idx != pre_idx:
            #     num_wrong += 1

            num_wrong = num_wrong + np.abs(gt_idx - pre_idx)
        error = num_wrong / float(len(prediction))
    return error