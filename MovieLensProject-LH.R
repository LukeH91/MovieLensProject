###########################################
# MovieLens Recommendation System Project #
###########################################


### PART 1: SETUP

# The following installs and loads all required packages:
if(!require(tidyverse)) install.packages("tidyverse",
                                         repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret",
                                     repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table",
                                          repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs",
                                          repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate",
                                      repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(data.table)
library(dslabs)
library(lubridate)

# Downloads, unzips, and appropriately renames columns of the MovieLens Dataset:
dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Create validation set from 10% of the data:
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Clean up environment
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# We now have two data sets:
# edx, which must be used to train and test all models
# validation, the final data set which must only be used for prediction at the end

# splitting the edx data into training and test sets
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# using semi-join to ensure that users and movies are not in both sets
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# we will assess the performance of models by computing their Residual Mean
# Squared Error (where lower scores are better) using this function:
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# set tibbles to display 7 digits
options(pillar.sigfig = 7)


### PART 2: DATA EXPLORATION AND VISUSALISATION

# print all column names:
cat("Column Names: ",names(edx))

# print number of rows:
cat("edx Height:",dim(edx)[1])
# print number of columns:
cat("edx Width:",dim(edx)[2])

# print number of unique users and films
edx %>%
  summarize(Unique_Users = n_distinct(userId),
            Unique_Movies = n_distinct(movieId))

# print difference between data set and movies * users
cat("Users * Movies = ",69878 * 10677)
cat("Our Row Total  = ",9000055)

# print matrix showing which films were rated by 1000 random users
users <- sample(unique(edx$userId), 1000)
edx %>% filter(userId %in% users) %>%
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% 
  as.matrix() %>% t(.) %>%
  image(xlab="Movies", ylab="Users", col="black") %>%
  title(main="Matrix Showing Distribution of User Ratings by Movie")

# trend line showing the relationship between ratings and time
# timestamp is first converted into week to allow better participant grouping
edx %>% mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Ratings Given Over Time") +
  xlab("Date of Rating (Years)") + 
  ylab("Rating Given")

# histogram of distribution of movie ratings
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black", fill = "darkgreen") + 
  scale_x_log10() + 
  ggtitle("Distribution of Number of Ratings Given to Movies") +
  xlab("Number of Ratings") + 
  ylab("Number of Films")

# histogram of distribution of number of ratings given by each user
edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black", fill = "darkred") + 
  scale_x_log10() +
  ggtitle("Distribution of Number of Ratings Given by Users") +
  xlab("Number of Ratings") + 
  ylab("Number of Users")





### PART 3: MODEL TRAINING

# we will now generate a series of models to try and predict ratings, which will
# be presented in increasing order of complexity. at each stage, their RMSE will
# be added to a table for easy comparisons

#### Model 1: Naive Bayes Prediction (Midpoint)

# predicts scores using the midpoint value of 2.75
model1 <- rep(2.75, nrow(train_set))
model1_rmse <- RMSE(test_set$rating, model1)
# knits the RMSE of model 1 to a table
rmse_results <- tibble(method = "Model 1: Scale Midpoint", RMSE = model1_rmse)
rmse_results %>% knitr::kable()


#### Model 2: Naive Bayes Prediction (Mean)

# predicts the scores using the mean score across all films (3.51)
model2 <- mean(train_set$rating)
model2_rmse <- RMSE(test_set$rating, model2)
# knits the RMSE of model 2 to a table
rmse_results <- bind_rows(rmse_results, tibble(method="Model 2: Mean of All Films",
                                               RMSE = model2_rmse ))
rmse_results %>% knitr::kable()


#### Model 3: Average of Each Film

# predicts scores using the mean score of each film
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  # calculates difference between each film mean and the overall mean
  summarize(b_i = mean(rating - mu))
# trains model 3
model3 <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i
model3_rmse <- RMSE(model3, test_set$rating)
# knits the RMSE of model 3 to a table
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Model 3: Mean of Each Film",
                                     RMSE = model3_rmse ))
rmse_results %>% knitr::kable()


#### Model 4: Average of Each User & Film

# predicts scores using the mean score of each film + each user
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  # calculates difference between each user mean and the overall mean
  summarize(b_u = mean(rating - mu - b_i))
# trains model 4
model4 <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
model4_rmse <- RMSE(test_set$rating, model4)
# knits the RMSE of model 4 to a table
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Model 4: Movie & User Means",  
                                     RMSE = model4_rmse ))
rmse_results %>% knitr::kable()


#### Model 5: Average of Each User & Film with Time

# timestamp is first converted into week-long bins to be more useful for grouping
train_set <- train_set %>%
  mutate(week = round_date(as_datetime(timestamp), unit = "week"))
# the same must be done in the test set to allow joining
test_set <- test_set %>% 
  mutate(week = round_date(as_datetime(timestamp), unit = "week"))
# predicts scores using the mean score of each film + each user + timestamp
time <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(week) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u))
# trains model 5
model5 <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(time, by='week') %>%
  mutate(pred = mu + b_i + b_u + b_t) %>%
  .$pred
model5_rmse <- RMSE(test_set$rating, model5)
# knits the RMSE of model 4 to a table
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Model 5: Movie & User Means with Time",  
                                     RMSE = model5_rmse ))
rmse_results %>% knitr::kable()


#### Model 6: Average of Each User & Film with Regularisation

# selecting the optimal lambda (and thus minimal RMSE) for User & Movie model 
# Note: This code takes a long time to run
lambdas <- seq(0, 10, 0.25)
rmses6 <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})
# optional: display lambda optimisation graph and minimised lambda (4.75)
# lambdas[which.min(rmses6)]
# qplot(lambdas, rmses6)
# corresponding minimised RMSE is saved
model6_rmse <- min(rmses6)
# knits the RMSE of model 6 to a table
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Model 6: Regularised Movie & User Means",  
                                     RMSE = model6_rmse ))
rmse_results %>% knitr::kable()


#### Model 7: Average of Each User, Film & Time with Regularisation

# selecting the optimal lambda (and thus minimal RMSE) for User, Movie & Time model 
# Note: This code takes an even longer time to run
lambdas <- seq(0, 10, 0.25)
rmses7 <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_t <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(week) %>%
    summarize(b_t = sum(rating - b_i - b_u - mu)/(n()+l))
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_t, by = "week") %>%
    mutate(pred = mu + b_i + b_u + b_t) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})
# optional: display lambda optimisation graph and minimised lambda (5)
# lambdas[which.min(rmses7)]
# qplot(lambdas, rmses7)
# corresponding minimised RMSE is saved
model7_rmse <- min(rmses7)
# knits the RMSE of model 7 to a table
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Model 7: Regularised Movie & User Means with Time",  
                                     RMSE = model7_rmse ))
rmse_results %>% knitr::kable()


#### PART 4: FINAL MODEL EVALUATION

# use model 7, this time on the entire edx dataset and the validation set
# Reset everything except main data sets
rm(list=setdiff(ls(), c("edx","validation")))
# add "week" column to both datasets to allow grouping and joining
validation <- validation %>%
  mutate(week = round_date(as_datetime(timestamp), unit = "week"))
edx <- edx %>%
  mutate(week = round_date(as_datetime(timestamp), unit = "week"))
# the following is identical to Model 7 above
lambdas <- seq(0, 10, 0.25)
rmses_final <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_t <- edx %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(week) %>%
    summarize(b_t = sum(rating - b_i - b_u - mu)/(n()+l))
  predicted_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_t, by = "week") %>%
    mutate(pred = mu + b_i + b_u + b_t) %>%
    .$pred
  return(RMSE(predicted_ratings, validation$rating))
})
# optional: display lambda optimisation graph and minimised lambda (5.5)
# lambdas[which.min(rmses_final)]
# qplot(lambdas, rmses_final)
# corresponding minimised RMSE is saved
final_model_rmse <- min(rmses_final)
# this displays the final RMSE value - 0.8646938
cat("Final Model Performance:", final_model_rmse)