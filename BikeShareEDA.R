# Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(dplyr)
library(skimr)
library(DataExplorer)
library(GGally)
library(patchwork)
library(glmnet)
library(rpart)

# Read in data
test <- vroom("C:/STAT348/KaggleBikeShare/test.csv")
train <- vroom("C:/STAT348/KaggleBikeShare/train.csv")

# Clean data
train <- train %>% select(-registered,-casual)
train$count <- log(train$count)

# EDA
glimpse(test)
glimpse(train)
skim(test)
skim(train)
plot_intro(test)
plot_intro(train)
plot_correlation(test)
plot_correlation(train)
plot_bar(test)
plot_bar(train)
plot_histogram(test)
plot_histogram(train)
plot_missing(test)
plot_missing(train)
ggpairs(test)
ggpairs(train)

p1 <- ggplot(test, aes(x = weather)) + geom_bar()
p2 <- ggplot(test, aes(x = humidity)) + geom_histogram(bins = 30)
p3 <- ggplot(test, aes(x = temp)) + geom_histogram(bins = 30)
p4 <- ggplot(test, aes(x = windspeed)) + geom_histogram(bins = 30)
(p1+p2)/(p3+p4)

# My Recipe
my_recipe <- recipe(count ~ ., data = train) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_mutate(season = factor(season)) %>%
  step_date(datetime, features = "dow") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())
prepped_recipe <- prep(my_recipe)
baked_data <- bake(prepped_recipe, new_data = test)
  
# Linear Regression
bikes.lm <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression") %>%
  fit(formula = count ~ . -datetime, data = train)

bike_predictions <- predict(bikes.lm, new_data = test)
bike_predictions

kaggle_submission <- bike_predictions %>%
  bind_cols(., test) %>%
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")

# Workflow
bike.lm <- linear_reg() %>%
set_engine("lm") %>%
set_mode("regression")

bike_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(bike.lm) %>%
fit(data=train)

lin_preds <- predict(bike_workflow, new_data = test) %>%
  bind_cols(test) %>%
  mutate(pred_count = exp(.pred))

head(baked_data, 5)

kaggle_submission2 <- lin_preds %>%
  transmute(
    datetime = as.character(format(datetime)),
    count = pred_count)


vroom_write(x = kaggle_submission2, file = "./MyFirstRecipe.csv", delim = ",")

# Penalized Regression
preg_model <- linear_reg(penalty=.005, mixture=.9) %>%
  set_engine("glmnet")
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model) %>%
  fit(data=train)
kaggle_submission3 <- predict(preg_wf, new_data = test) %>%
  bind_cols(test %>% select(datetime)) %>%
  mutate(count = exp(.pred)) %>%
  transmute(
    datetime = as.character(format(datetime)),
    count = count)

vroom_write(kaggle_submission3, file = "./glmnet_preds5.csv", delim = ",")

preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>%
  set_engine("glmnet")

preg_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(preg_model)

grid_of_tuning_params <- grid_regular(penalty(),
                                      mixture(),
                                      levels = 4)


folds <- vfold_cv(train, v = 10, repeats=1)

CV_results <- preg_wf %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse,mae))

collect_metrics(CV_results) %>%
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

bestTune <- CV_results %>%
  select_best(metric="rmse")

final_wf <-
  preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

final_preds <- final_wf %>%
  predict(new_data = test) %>%
  bind_cols(test %>% select(datetime)) %>%
  mutate(count = exp(.pred)) %>%
  transmute(
    datetime = as.character(format(datetime)),
    count = count)

vroom_write(final_preds, file = "./tuning_parameters.csv", delim = ",")

# Regression Tree
tree_model <- decision_tree(tree_depth = tune(),
                            cost_complexity = tune(),
                            min_n=tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

tree_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(tree_model)

tree_grid <- grid_regular(
  tree_depth(),
  cost_complexity(),
  min_n(),
  levels = 3  # adjust for how large you want the search space
)

tree_folds <- vfold_cv(train, v = 10)

tree_results <- tree_wf %>%
  tune_grid(
    resamples = tree_folds,
    grid = tree_grid,
    metrics = metric_set(rmse, mae)
  )

best_tree <- tree_results %>%
  select_best(metric = "rmse")

final_tree_wf <- tree_wf %>%
  finalize_workflow(best_tree) %>%
  fit(data = train)

tree_preds <- final_tree_wf %>%
  predict(new_data = test) %>%
  bind_cols(test %>% select(datetime)) %>%
  mutate(count = exp(.pred)) %>%  # back-transform log(count)
  transmute(
    datetime = as.character(format(datetime)),
    count = count
  )

vroom_write(tree_preds, file = "./tree_submission.csv", delim = ",")


