# Nowcasting Competition - PVI

# Team name: Delphi
# Team member: Haseeb Mahmud
# Address: Zähringerstraße 17, 65189 Wiesbaden, Germany
# Email: haseeb.mahmud@gmail.com
# Phone: +4917657860809

# Submission for: September 2022


# 00. Preamble ----

# Disable scientific notation
options(scipen = 999, digits = 4)

# Set the language = EN
Sys.setenv(LANG = "en")

# Loading libraries

library(readr)
library(tidyverse)
library(fable)
library(tsibble)
library(tsibbledata)
library(lubridate)

# Data open 

raw <- read_csv("00_Data/sts_inpr_m__custom_3447457_linear.csv")

data <- raw %>% 
  select(geo, TIME_PERIOD, OBS_VALUE) %>%
  mutate(TIME_PERIOD = yearmonth(TIME_PERIOD)) 

data <- as_tsibble(data, index = TIME_PERIOD, key = geo)

fit <- data %>%
  fill_gaps(OBS_VALUE = 0L) %>%
  model(
    ets = ETS(OBS_VALUE),
    arima = ARIMA(OBS_VALUE),
    theta = THETA(OBS_VALUE)
  ) %>%
  mutate(
    average = (ets + arima + theta) / 3
  ) 

fc <- fit %>%
  forecast(h = 2) 

accuracy_table <- accuracy(fit)

