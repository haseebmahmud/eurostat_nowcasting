# Nowcasting Competition - PPI
# Approach 2: Machine learning methods (5-model)

# Team name: Delphi
# Team member: Haseeb Mahmud
# Address: Zähringerstraße 17, 65189 Wiesbaden, Germany
# Email: haseeb.mahmud@gmail.com
# Phone: +4917657860809


# LIBRARIES & SETUP ----

# Time Series ML
library(tidymodels)
library(modeltime)
library(modeltime.ensemble)

# Timing & Parallel Processing
library(tictoc)
library(future)
library(doFuture)

# Core 
library(tidyquant)
library(tidyverse)
library(timetk)
library(lubridate)

# * Parallel Processing ----

registerDoFuture()
n_cores <- parallel::detectCores()
plan(
  strategy = cluster,
  workers  = parallel::makeCluster(n_cores)
)
# 
plan(sequential)

# 1.0 DATA ----

# * Read Data ----
raw <- read_csv("00_Data/Sept_2022/sts_inpr_m__custom_3447457_linear.csv")

# Convert date/time variable into Date class
raw$TIME_PERIOD <- as.Date(paste(raw$TIME_PERIOD,"-01",sep=""))

# Clean raw data
data <- raw %>% 
  select(geo, TIME_PERIOD, OBS_VALUE) %>%
  rename(time = TIME_PERIOD,
         value = OBS_VALUE) 

data %>%
  group_by(geo) %>%
  plot_time_series(
    time, value,
    .facet_ncol  = 3,
    .smooth      = FALSE, 
    .interactive = FALSE
  )

# * Full Data ----
full_data_tbl <- data %>%
  
  #select(time, geo, value) %>%
  group_by(geo) %>%
  pad_by_time(time, .by = "month", .pad_value = 0) %>%
  ungroup() %>%
  
  # Global Features / Transformations / Joins
  mutate(value = log1p(value)) %>%
  
  # Group-Wise Feature Transformations
  group_by(geo) %>%
  future_frame(time, .length_out = 6, .bind_data = TRUE) %>%
  ungroup() %>%
  
  # Lags & Rolling Features / Fourier
  #mutate(geo = as_factor(geo)) %>%
  group_by(geo) %>%
  group_split() %>%
  map(.f = function(df) {
    df %>%
      arrange(time) %>%
      #tk_augment_fourier(time, .periods = c(1, 6)) %>%
      tk_augment_lags(value, .lags = 6) %>%
      tk_augment_slidify(
        value_lag6,
        .f       = ~ mean(.x, na.rm = TRUE),
        .period  = c(1, 3, 6),
        .partial = TRUE, 
        .align   = "center"
      )
  }) %>%
  bind_rows() %>%
  
  rowid_to_column(var = "rowid")

full_data_tbl

# * Data Prepared ----

data_prepared_tbl <- full_data_tbl %>%
  filter(!is.na(value)) %>%
  drop_na()

data_prepared_tbl

# * Future Data ----
future_tbl <- full_data_tbl %>%
  filter(is.na(value))

future_tbl %>% filter(is.nan(value_lag6_roll_6))

future_tbl <- future_tbl %>%
  mutate(
    across(.cols = contains("_lag"), 
           .fns  = function(x) ifelse(is.nan(x), NA, x))
  ) %>%
  
  # FIX 1 ----
# Use replace_na() to fill NA values with zero
# fill(contains("_lag"), .direction = "up")
mutate(across(.cols = contains("_lag"), .fns = ~replace_na(.x, 0)))
# END FIX 1 ----

future_tbl %>% filter(is.na(value_lag6_roll_6))

# 2.0 TIME SPLIT ----

splits <- data_prepared_tbl %>%
  time_series_split(time, assess = 6, cumulative = TRUE)

splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(time, value)

# 3.0 RECIPE ----

# * Clean Training Set ----
# - With Panel Data, need to do this outside of a recipe
# - Transformation happens by group

train_cleaned <- training(splits) %>%
  group_by(geo) %>%
  mutate(value = ts_clean_vec(value, period = 6)) %>%
  ungroup()

# training(splits) %>%
train_cleaned %>%
  group_by(geo) %>%
  plot_time_series(
    time, value, 
    .facet_ncol  = 4, 
    .smooth      = FALSE, 
    .interactive = FALSE
  )


# * Recipe Specification ----

train_cleaned

recipe_spec <- recipe(value ~ ., data = train_cleaned) %>%
  update_role(rowid, new_role = "indicator") %>%
  step_timeseries_signature(time) %>%
  step_rm(matches("(.xts$)|(.iso$)|(hour)|(minute)|(second)|(am.pm)")) %>%
  step_normalize(time_index.num, time_year) %>%
  step_other(geo) %>%
  step_dummy(all_nominal(), one_hot = TRUE)

recipe_spec %>% prep() %>% juice() %>% glimpse()

# 4.0 MODELS ----

# * PROPHET ----

wflw_fit_prophet <- workflow() %>%
  add_model(
    spec = prophet_reg() %>% set_engine("prophet")
  ) %>%
  add_recipe(recipe_spec) %>%
  fit(train_cleaned)


# * XGBOOST ----

wflw_fit_xgboost <- workflow() %>%
  add_model(
    spec = boost_tree(mode = "regression") %>% set_engine("xgboost")
  ) %>%
  # HARDHAT 1.0.0 FIX ----
# add_recipe(recipe_spec %>% update_role(date, new_role = "indicator")) %>%
add_recipe(recipe_spec %>% step_rm(time)) %>%
  fit(train_cleaned)


# * PROPHET BOOST ----

wflw_fit_prophet_boost <- workflow() %>%
  add_model(
    spec = prophet_boost(
      seasonality_daily  = FALSE, 
      seasonality_weekly = FALSE, 
      seasonality_yearly = FALSE
    ) %>% 
      set_engine("prophet_xgboost")
  ) %>%
  add_recipe(recipe_spec) %>%
  fit(train_cleaned)



# * SVM ----
# 
# wflw_fit_svm <- workflow() %>%
#   add_model(
#     spec = svm_rbf(mode = "regression") %>% set_engine("kernlab")
#   ) %>%
#   # HARDHAT 1.0.0 FIX ----     
# # add_recipe(recipe_spec %>% update_role(date, new_role = "indicator")) %>%     
# add_recipe(recipe_spec %>% step_rm(time)) %>%
#   fit(train_cleaned)



# * RANDOM FOREST ----

wflw_fit_rf <- workflow() %>%
  add_model(
    spec = rand_forest(mode = "regression") %>% set_engine("ranger")
  ) %>%
  # HARDHAT 1.0.0 FIX ----
# add_recipe(recipe_spec %>% update_role(date, new_role = "indicator")) %>%
add_recipe(recipe_spec %>% step_rm(time)) %>%
  fit(train_cleaned)
# 
# # * NNET ----
# 
# wflw_fit_nnet <- workflow() %>%
#   add_model(
#     spec = mlp(mode = "regression") %>% set_engine("nnet")
#   ) %>%
#   # HARDHAT 1.0.0 FIX ----
# # add_recipe(recipe_spec %>% update_role(date, new_role = "indicator")) %>%
# add_recipe(recipe_spec %>% step_rm(time)) %>%
#   fit(train_cleaned)

# * MARS ----

wflw_fit_mars <- workflow() %>%
  add_model(
    spec = mars(mode = "regression") %>% set_engine("earth")
  ) %>%
  # HARDHAT 1.0.0 FIX ----
# add_recipe(recipe_spec %>% update_role(date, new_role = "indicator")) %>%
add_recipe(recipe_spec %>% step_rm(time)) %>%
  fit(train_cleaned)


# * ACCURACY CHECK ----

submodels_1_tbl <- modeltime_table(
  wflw_fit_prophet,
  wflw_fit_xgboost,
  wflw_fit_prophet_boost,
  #wflw_fit_svm,
  wflw_fit_rf,
  #wflw_fit_nnet,
  wflw_fit_mars
)

submodels_1_tbl %>%
  modeltime_accuracy(testing(splits)) %>%
  arrange(rmse)


# 5.0 HYPER PARAMETER TUNING ---- 

# * RESAMPLES - K-FOLD ----- 

set.seed(123)
resamples_kfold <- train_cleaned %>% vfold_cv(v = 5)

resamples_kfold %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(time, value, .facet_ncol = 2)


# * XGBOOST TUNE ----

# ** Tunable Specification

model_spec_xgboost_tune <- boost_tree(
  mode            = "regression", 
  mtry            = tune(),
  trees           = tune(),
  min_n           = tune(),
  tree_depth      = tune(),
  learn_rate      = tune(),
  loss_reduction  = tune()
) %>% 
  set_engine("xgboost")

wflw_spec_xgboost_tune <- workflow() %>%
  add_model(model_spec_xgboost_tune) %>%
  add_recipe(recipe_spec %>% update_role(time, new_role = "indicator"))

# ** Tuning

tic()
set.seed(123)
tune_results_xgboost <- wflw_spec_xgboost_tune %>%
  tune_grid(
    resamples  = resamples_kfold,
    param_info = parameters(wflw_spec_xgboost_tune) %>%
      update(
        learn_rate = learn_rate(range = c(0.001, 0.400), trans = NULL)
      ),
    grid = 10,
    control = control_grid(verbose = TRUE, allow_par = TRUE)
  )
toc()


# ** Results

tune_results_xgboost %>% show_best("rmse", n = Inf)


# ** Finalize

wflw_fit_xgboost_tuned <- wflw_spec_xgboost_tune %>%
  finalize_workflow(select_best(tune_results_xgboost, "rmse")) %>%
  fit(train_cleaned)

# * RANGER TUNE ----

# ** Tunable Specification

model_spec_rf_tune <- rand_forest(
  mode    = "regression",
  mtry    = tune(),
  trees   = tune(),
  min_n   = tune()
) %>% 
  set_engine("ranger")

wflw_spec_rf_tune <- workflow() %>%
  add_model(model_spec_rf_tune) %>%
  add_recipe(recipe_spec %>% update_role(time, new_role = "indicator"))

# ** Tuning

tic()
set.seed(123)
tune_results_rf <- wflw_spec_rf_tune %>%
  tune_grid(
    resamples = resamples_kfold,
    grid      = 5,
    control   = control_grid(verbose = TRUE, allow_par = TRUE)
  )
toc()

# ** Results

tune_results_rf %>% show_best("rmse", n = Inf)

# ** Finalize

wflw_fit_rf_tuned <- wflw_spec_rf_tune %>%
  finalize_workflow(select_best(tune_results_rf, "rmse")) %>%
  fit(train_cleaned)

# * EARTH TUNE ----

# ** Tunable Specification

model_spec_earth_tune <- mars(
  mode        = "regression",
  num_terms   = tune(),
  prod_degree = tune()
) %>%
  set_engine("earth")

wflw_spec_earth_tune <- workflow() %>%
  add_model(model_spec_earth_tune) %>%
  add_recipe(recipe_spec %>% update_role(time, new_role = "indicator"))

# ** Tuning

tic()
set.seed(123)
tune_results_earth <- wflw_spec_earth_tune %>%
  tune_grid(
    resamples = resamples_kfold, 
    grid      = 10,
    control   = control_grid(allow_par = TRUE, verbose = TRUE)
  )
toc()


# ** Results
tune_results_earth %>% show_best("rmse")


# ** Finalize
wflw_fit_earth_tuned <- wflw_spec_earth_tune %>%
  finalize_workflow(tune_results_earth %>% select_best("rmse")) %>%
  fit(train_cleaned)

# 6.0 EVALUATE PANEL FORECASTS  -----

# * Model Table ----

submodels_2_tbl <- modeltime_table(
  wflw_fit_xgboost_tuned,
  wflw_fit_rf_tuned,
  wflw_fit_earth_tuned
) %>%
  update_model_description(1, "XGBOOST - Tuned") %>%
  update_model_description(2, "RANGER - Tuned") %>%
  update_model_description(3, "EARTH - Tuned") %>%
  combine_modeltime_tables(submodels_1_tbl)


# * Calibration ----
calibration_tbl <- submodels_2_tbl %>%
  modeltime_calibrate(testing(splits))

# * Accuracy ----
calibration_tbl %>% 
  modeltime_accuracy() %>%
  arrange(rmse)

# * Forecast Test ----

calibration_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = data_prepared_tbl,
    keep_data   = TRUE 
  ) %>%
  group_by(geo) %>%
  plot_modeltime_forecast(
    .facet_ncol         = 4, 
    .conf_interval_show = FALSE,
    .interactive        = TRUE
  )


# 7.0 RESAMPLING ----
# - Assess the stability of our models over time
# - Helps us strategize an ensemble approach

# * Time Series CV ----

resamples_tscv <- train_cleaned %>%
  time_series_cv(
    assess      = 6,
    skip        = 6,
    cumulative  = TRUE, 
    slice_limit = 4
  )

resamples_tscv %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(time, value)

# * Fitting Resamples ----

model_tbl_tuned_resamples <- submodels_2_tbl %>%
  modeltime_fit_resamples(
    resamples = resamples_tscv,
    control   = control_resamples(verbose = TRUE, allow_par = TRUE)
  )


# * Resampling Accuracy Table ----

model_tbl_tuned_resamples %>%
  modeltime_resample_accuracy(
    metric_set  = metric_set(rmse, rsq),
    summary_fns = list(mean = mean, sd = sd)
  ) %>%
  arrange(rmse_mean)


# * Resampling Accuracy Plot ----

model_tbl_tuned_resamples %>%
  plot_modeltime_resamples(
    .metric_set  = metric_set(mae, rmse, rsq),
    .point_size  = 4, 
    .point_alpha = 0.8,
    .facet_ncol  = 1
  )


# 8.0 ENSEMBLE PANEL MODELS -----

# * Average Ensemble ----

submodels_2_ids_to_keep <- c(6, 7, 2, 1)

ensemble_fit <- submodels_2_tbl %>%
  filter(.model_id %in% submodels_2_ids_to_keep) %>%
  # FIX 2 ----
# - use type = "median" to reduce effect of crazy spikes 
#   & overfitting from prophet w/ regressors
ensemble_average(type = "median")
# END FIX 2 ----


model_ensemble_tbl <- modeltime_table(
  ensemble_fit
)

# * Accuracy ----

model_ensemble_tbl %>%
  modeltime_accuracy(testing(splits))


# * Forecast ----

forecast_ensemble_test_tbl <- model_ensemble_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = data_prepared_tbl,
    keep_data   = TRUE
  ) %>%
  mutate(
    across(.cols = c(.value, value), .fns = expm1)
  )

forecast_ensemble_test_tbl %>%
  group_by(geo) %>%
  plot_modeltime_forecast(
    .facet_ncol = 4
  )

forecast_ensemble_test_tbl %>%
  filter(.key == "prediction") %>%
  select(geo, .value, value) %>%
  # group_by(geo) %>%
  summarize_accuracy_metrics(
    truth      = value, 
    estimate   = .value,
    metric_set = metric_set(mae, rmse, rsq)
  )

# * Refit ----

data_prepared_tbl_cleaned <- data_prepared_tbl %>%
  group_by(geo) %>%
  mutate(value = ts_clean_vec(value, period = 7)) %>%
  ungroup()

model_ensemble_refit_tbl <- model_ensemble_tbl %>%
  modeltime_refit(data_prepared_tbl_cleaned)

model_ensemble_refit_tbl %>%
  modeltime_forecast(
    new_data    = future_tbl,
    actual_data = data_prepared_tbl,
    keep_data   = TRUE 
  ) %>%
  mutate(
    .value    = expm1(.value),
    value = expm1(value)
  ) %>%
  group_by(geo) %>%
  plot_modeltime_forecast(
    .facet_ncol   = 4,
    .y_intercept  = 0
  )

# * Turn OFF Parallel Backend
plan(sequential)

# 9.0 RECAP ----
# - You:
#     1. Prepared 20 Time Series Groups
#     2. Modeled Panel Data
#     3. Hyper Parameter Tuned 
#     4. Resampled & Evaluated Accuracy Over Time
#     5. Ensembled the models using a strategy based on resample stability
#     6. RMSE 143, MAE 46, RSQ 0.40
# - This code can work for 10,000 Time Series. 
#     1. Only expense is Hyper Parameter Tuning
#     2. Watch Saving Ensembles & Models - Memory Size 


