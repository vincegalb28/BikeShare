library(tidyverse)
library(tidymodels)
library(vroom)
library(dplyr)
library(skimr)
library(DataExplorer)
library(GGally)
library(patchwork)

test <- vroom("C:/STAT348/KaggleBikeShare/test.csv")
train <- vroom("C:/STAT348/KaggleBikeShare/train.csv")

test$season <- as.factor(test$season)
test$holiday <- as.factor(test$holiday)
test$workingday <- as.factor(test$workingday)
test$weather <- as.factor(test$weather)
train$season <- as.factor(train$season)
train$holiday <- as.factor(train$holiday)
train$workingday <- as.factor(train$workingday)
train$weather <- as.factor(train$weather)


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

train <- train %>% select(-registered,-casual)
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
