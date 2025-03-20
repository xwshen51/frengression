## Import packages
library(glmnet)
library(zeallot) #enable %<-%
library(mvtnorm)
library(causl)
library(survivl)
library(npcausal)
library(dplyr)
library(haven)
library(survival)
library(tidyr)
library(purrr)
library(stringr)


## baseline ####
bsl_vars <- c("SEX", "AGE", "RACE", "SMOKER", "DIABDUR", "BMIBL", "HBA1CBL",
              "HDL1BL", "LDL1BL", "CHOL1BL", "TRIG1BL", "CREATBL")

mh_vars <- c("MIFL","STROKEFL","STENFL", "NEPSCRFL", "KIDFL")

## longitudinal ####
lb_vars <- c("HBA1C", "EGFRCKD")
vs_vars <- "BMI"

## time to event ####
tte_vars <- c("ALDTHTM", "MACEEVTM", "MACEMITM")


# Load script that goes through both folders - credit to Jens
path = "../../../../Project/LEADER/Box/"
folder = list(start = paste0(path, "Analysis Ready Datasets/SAS_analysis/"),
              fin = paste0(path, "Analysis Ready Datasets/R_analysis/"))

load_func <- function(ds, folder){
  if (file.exists(paste0(folder$fin, ds, ".rds"))){
    out <- readRDS(paste0(folder$fin, ds, ".rds"))
  } else {
    out <- haven::read_sas(paste0(folder$start, ds, ".sas7bdat"))
    saveRDS(out, paste0(folder$fin, ds, ".rds"))
  }
  return(out)
}

# Load in tables 
adsl <- load_func("adsl", folder) # demographics
advs <- load_func("advs", folder) # vitals
adtte <- load_func("adtte", folder) # primary outcomes
adlb <- load_func("adlb", folder) # labs; can be slow


df_bsl <- adsl %>%
  filter(FASFL == "Y") %>% 
  select(USUBJID, ARM, all_of(bsl_vars), all_of(mh_vars))

df_long <- rbind(
  adlb %>% 
    filter(FASFL == "Y") %>% 
    filter(PARAMCD %in% lb_vars) %>% 
    mutate(DS = "adlb") %>% 
    select(DS, USUBJID, ABLFL, AVISIT, AVISITN, ADY,
           PARAM, PARAMCD, AVALU, AVAL, CHG, PCHG, DTYPE),
  advs %>% 
    filter(FASFL == "Y") %>% 
    filter(PARAMCD %in% vs_vars) %>% 
    mutate(DS = "advs") %>% 
    select(DS, USUBJID, ABLFL, AVISIT, AVISITN, ADY,
           PARAM, PARAMCD, AVALU, AVAL, CHG, PCHG, DTYPE)
)

death <- adtte %>% 
  filter(FASFL == "Y") %>% 
  filter(PARAMCD == "ALDTHTM") %>% 
  mutate(death = ifelse(CNSR == 1, 0, 1))

df_out <- adtte %>% 
  filter(FASFL == "Y") %>% 
  filter(PARAMCD %in% tte_vars) %>% 
  mutate(event = ifelse(CNSR == 1, 0, 1)) %>% 
  select(USUBJID, PARAM, PARAMCD, AVAL, event) %>% 
  left_join(death %>% select(USUBJID, death)) %>% 
  mutate(death = ifelse(event == 1, 0, death))

out <- list(
  df_bsl = df_bsl,
  df_long = df_long, 
  df_out = df_out
)

generate_bsl <- function(){
    # Step 1. Convert ARM into treatment indicator:
    df_bsl <- df_bsl %>%
    mutate(ARM = ifelse(ARM == "Liraglutide", 1,
                        ifelse(ARM == "Placebo", 0, NA)))

    # Step 2. Process all covariate columns (all except USUBJID and ARM):
    # Create an empty list to hold processed columns
    cov_list <- list()
    # Get the names of the columns to process
    cov_names <- setdiff(names(df_bsl), c("USUBJID", "ARM"))

    # Loop over each covariate column
    for(col in cov_names) {
    column_data <- df_bsl[[col]]
    
    # If the column is a character or factor type:
    if(is.character(column_data) || is.factor(column_data)) {
        # Check if the column is binary with "N" and "Y"
        if(all(unique(as.character(column_data)) %in% c("N", "Y"))) {
        cov_list[[col]] <- ifelse(column_data == "Y", 1, 0)
        } else {
        # Otherwise, create dummy variables for a multi-level categorical variable.
        # model.matrix creates one column per level (no intercept).
        dummies <- model.matrix(~ column_data - 1)
        # Drop the last dummy column if more than one dummy variable is created.
        if(ncol(dummies) > 1){
            dummies <- dummies[, -ncol(dummies), drop = FALSE]
        }
        # Optionally, rename dummy columns to include original column name as prefix
        dummy_names <- paste(col, sub("column_data", "", colnames(dummies)), sep = "_")
        colnames(dummies) <- dummy_names
        # Add each dummy column into our list
        for(j in seq_along(dummy_names)) {
            cov_list[[ dummy_names[j] ]] <- dummies[, j]
        }
        }
    } else {
        # For numeric columns, just include them unchanged
        cov_list[[col]] <- column_data
    }
    }

    # Combine the processed covariate columns into one data frame:
    cov_df <- as.data.frame(cov_list)

    # Step 3. Reorder the covariate columns so that binary ones come first.
    # Here we define a binary column as one that (ignoring NAs) only takes values 0 and 1.
    is_binary <- sapply(cov_df, function(x) {
    vals <- unique(x[!is.na(x)])
    length(vals) == 2 && all(sort(vals) == c(0, 1))
    })
    # Order: binary columns first, then the rest
    cov_df <- cov_df[, c(names(cov_df)[is_binary], names(cov_df)[!is_binary])]

    # Step 4. Rename all covariate columns as X_1, X_2, …, X_p
    # new_names <- paste0("X", seq_along(cov_df))
    # colnames(cov_df) <- new_names

    # Combine USUBJID, ARM with the newly processed covariates
    df_bsl_converted <- cbind(df_bsl[, c("USUBJID", "ARM")], cov_df)
    return(df_bsl_converted)
}

generate_outcome<-function(){

  # Determine maximum follow-up time and number of intervals (each of 6 months)
  max_time <- max(df_out$AVAL)
  num_intervals <- ceiling(max_time / 6)

  # Separate the outcomes:
  # MACE outcome (use PARAMCD "MACEEVTM")
  mace <- df_out %>% 
    filter(PARAMCD == "MACEEVTM") %>% 
    select(USUBJID, time = AVAL, event)

  # Death outcome (use PARAMCD "ALDTHTM")
  death <- df_out %>% 
    filter(PARAMCD == "ALDTHTM") %>% 
    select(USUBJID, time = AVAL, event = death)

  # Non-fatal MI outcome (use PARAMCD "MACEMITM")
  mi <- df_out %>% 
    filter(PARAMCD == "MACEMITM") %>% 
    select(USUBJID, time = AVAL, event)

  # Create a wide table that has one row per subject and merge outcomes
  surv_table <- df_out %>% 
    distinct(USUBJID) %>% 
    left_join(mace, by = "USUBJID") %>% 
    rename(time_mace = time, event_mace = event) %>%
    left_join(death, by = "USUBJID") %>% 
    rename(time_death = time, event_death = event) %>%
    left_join(mi, by = "USUBJID") %>% 
    rename(time_mi = time, event_mi = event)

  # Helper function to create the interval vector:
  # - time: the time to event (in months)
  # - event: indicator (1 = event occurred, 0 = censored/no event)
  # - num_intervals: total number of intervals
  # - interval_length: length of each interval (4 months here)
  create_interval_vector <- function(time, event, num_intervals, interval_length = 6) {
    res <- rep(0, num_intervals)
    if(event == 1) {
      # Determine which interval the event falls into
      event_interval <- ceiling(time / interval_length)
      if(event_interval > num_intervals) event_interval <- num_intervals
      res[event_interval] <- 1
      # Set subsequent intervals to NA once the event occurs
      if(event_interval < num_intervals) {
        res[(event_interval + 1):num_intervals] <- NA
      }
    }
    return(res)
  }

  # Apply the function to each outcome
  surv_table <- surv_table %>%
    mutate(
      Y = map2(time_mace, event_mace, ~create_interval_vector(.x, .y, num_intervals)),
      D = map2(time_death, event_death, ~create_interval_vector(.x, .y, num_intervals)),
      I = map2(time_mi, event_mi, ~create_interval_vector(.x, .y, num_intervals))
    )

  # Expand the list columns into separate columns for each interval:
  for (i in 1:num_intervals) {
    surv_table[[paste0("Y", i)]] <- map_dbl(surv_table$Y, ~.x[i])
    surv_table[[paste0("D", i)]] <- map_dbl(surv_table$D, ~.x[i])
    surv_table[[paste0("I", i)]] <- map_dbl(surv_table$I, ~.x[i])
  }

  # Select the final columns: one row per subject and columns for each interval and outcome
  final_table <- surv_table %>% 
    select(USUBJID, starts_with("Y"), starts_with("D"), starts_with("I"))
  
  final_table=select(final_table,-c("Y","Y11","D","D11","I","I11"))

  return(final_table)
}
# First, add a new column 'month' that extracts the month number.
# If "DAY 0" appears in AVISIT, mark month as "0"; otherwise, extract the digits after "MONTH ".
df_long <- df_long %>%
  mutate(month = if_else(str_detect(AVISIT, "DAY\\s*0"),
                        "0",
                        str_extract(AVISIT, "(?<=MONTH\\s)\\d+")))

# Optionally, filter to keep only rows that have a month value (if desired)
df_long_month <- df_long %>% 
  filter(!is.na(month))
generate_egfr<-function(){
  df_egfr <- df_long_month %>%
    filter(PARAMCD == "EGFRCKD") %>%
    select(USUBJID, month, AVAL) %>%
    group_by(USUBJID, month) %>% 
    summarise(AVAL = first(AVAL), .groups = "drop") %>%
    pivot_wider(id_cols = USUBJID, 
                names_from = month, 
                values_from = AVAL)
  return(df_egfr)
}
generate_hba1c<-function(){
  # HBA1C table
  df_hba1c <- df_long_month %>%
    filter(PARAMCD == "HBA1C") %>%
    select(USUBJID, month, AVAL) %>%
    group_by(USUBJID, month) %>% 
    summarise(AVAL = first(AVAL), .groups = "drop") %>%
    pivot_wider(id_cols = USUBJID, 
                names_from = month, 
                values_from = AVAL)
  return(df_hba1c)
}

generate_bmi<-function(){
    # BMI table
  df_bmi <- df_long_month %>%
    filter(PARAMCD == "BMI") %>%
    select(USUBJID, month, AVAL) %>%
    group_by(USUBJID, month) %>% 
    summarise(AVAL = first(AVAL), .groups = "drop") %>%
    pivot_wider(id_cols = USUBJID, 
                names_from = month, 
                values_from = AVAL)
return(df_bmi)
}