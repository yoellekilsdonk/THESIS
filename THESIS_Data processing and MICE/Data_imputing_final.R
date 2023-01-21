## This file contains the code for the Multiple Imputation via Chained Equations method 
## and some minor changes to input data sets to ensure compatibility with the method. 

library(dplyr)
library(mice)
library(modeest)

datasets = c(1,1.5,2) # Indicator for whether to use 1 million, 1.5 million, or 2 million simulated individuals

## PREPARING DATA SETS
for (i in 1:3){
  dataset = datasets[i]
  
  # Set working directory for storage purposes
  setwd("V:/UserData/079915")
  
  # Open fd_original (the RIVM data set)
  load(file="V:/UserData/079915/data_processing_3_data")
  fd_original <- fd
  length <- nrow(fd_original) # Number of rows in the original data set
  
  fd_original$id <- as.character(fd_original$id) # Due to compatibility issues with rbind of other two data sets later on
  
  # Combine female and male MISCAN simulation data sets into one
  if (dataset == 1) {
    female = read.csv(file="V:/UserData/079915/MISCAN_simulation_run_female_1mil")
    male = read.csv(file="V:/UserData/079915/MISCAN_simulation_run_male_1mil")
  } else if (dataset == 1.5) {
    female = read.csv(file="V:/UserData/079915/MISCAN_simulation_run_female_1.5mil")
    male = read.csv(file="V:/UserData/079915/MISCAN_simulation_run_male_1.5mil")
  } else if (dataset == 2) {
    female = read.csv(file="V:/UserData/079915/MISCAN_simulation_run_female_2mil")
    male = read.csv(file="V:/UserData/079915/MISCAN_simulation_run_male_2mil")
  }
  fd_MISCAN <- rbind(female,male)
  
  fd_MISCAN$hb <- NA_integer_ # Fill NANs for hb value (as these values are not available in MISCAN)
  
  # Open fd_15 (the RIVM data set with threshold 15 instead of 47)
  load(file="V:/UserData/079915/data_processing_1_data_15")
  fd_15 <- fd
  fd_15$hb <- round(with(fd_15, ifelse(fd_15$threshold %in% 275, fd_15$hb*0.170909,fd_15$hb )),1) # Correct for units of observations with threshold 275
  
  fd_15$id <- sub("$", "th15",as.character(fd_15$id)) # Create appendix to ID to distinguish between data sets

  # Remove unnecessary dataframes (for the sake of storage)
  rm(female)
  rm(male)
  rm(fd)
  
  # Include only relevant variables for each data set
  fd_original <- fd_original[,c("id", "participation_age","hb", "hb_stage_cat", "sex")]
  fd_MISCAN <- fd_MISCAN[,c("id", "participation_age", "hb", "hb_stage_cat", "sex")]
  fd_15 <- fd_15[,c("id", "participation_age","hb", "hb_stage_cat", "sex")]
  
  # Create data set of RIVM data, MISCAN simulation and RIVM data with threshold 15
  fd_mice <- rbind(fd_original, fd_MISCAN, fd_15)
  
  # Create prediction matrix where only participation_age, hb, and sex are used to impute hb_stage
  # Not necessary in principle as all variables other than "hb" and "hb_stage_cat" do not have missing variables
  # But for the sake of computing power we manually alter the prediction matrix
  pred_mat <- matrix(1,5,5)
  diag(pred_mat)<-0
  rownames(pred_mat) <- colnames(fd_15)
  colnames(pred_mat) <- colnames(fd_15)
  
  pred_mat[,"id"]<-0
  pred_mat["id",]<-0
  pred_mat["participation_age",]<-0
  pred_mat["sex",]<-0
  
  save(pred_mat, file="V:/UserData/079915/pred_mat")
  
  # Remove unnecessary dataframes (for the sake of storage)
  rm(fd_15)
  rm(fd_original)
  rm(fd_MISCAN)
  
  # Save files
  if (dataset == 1) {
    save(fd_mice, file="V:/UserData/079915/fd_mice_1mil")
  } else if (dataset == 1.5) {
    save(fd_mice, file="V:/UserData/079915/fd_mice_1.5mil")
  } else if (dataset == 2) {
    save(fd_mice, file="V:/UserData/079915/fd_mice_2mil")
  }
  
  rm(list = ls(all.names = TRUE)) # Clear environment
  gc() # Clear memory
  datasets = c(1,1.5,2)
} 

## PERFORMING MICE
for (i in 1:3){
  dataset = datasets[i]
  
  # Set working directory for storage purposes
  setwd("V:/UserData/079915")
  
  load(file="V:/UserData/079915/pred_mat")
  
  if (dataset == 1) {
    load(file="V:/UserData/079915/fd_mice_1mil")
  } else if (dataset == 1.5) {
    load(file="V:/UserData/079915/fd_mice_1.5mil")
  } else if (dataset == 2) {
    load(file="V:/UserData/079915/fd_mice_2mil")
  }
  
  set.seed(123)
  
  # Perform mice for 10 cycles, 5 times
  print(paste0("Starting MICE: ", Sys.time()))
  tempData <- mice(fd_mice, m=5, maxit=10, pred=pred_mat, method = c("","","sample","pmm",""),seed=123)
  
  rm(fd_mice)
  length <- 6796731 # nrow(fd_original)
  
  print(paste0("Starting dataframe creation: ", Sys.time()))
  # Create data frame consisting of the 10 newly imputed stage variable with length of original data set
  df<-complete(tempData,"repeated",include=TRUE)[1:length,]
  
  # Saving files
  print(paste0("Saving files: ", Sys.time()))
  if (dataset == 1) {
    save(df, file="V:/UserData/079915/df_1mil")
    save(tempData, file="V:/UserData/079915/temp_data_1mil")
  } else if (dataset == 1.5) {
    save(df, file="V:/UserData/079915/df_1.5mil")
    save(tempData, file="V:/UserData/079915/temp_data_1.5mil")
  } else if (dataset == 2) {
    save(df, file="V:/UserData/079915/df_2mil")
    save(tempData, file="V:/UserData/079915/temp_data_2mil")
  } 
  
  print(paste0("Done with ", dataset, ": ", Sys.time()))
  rm(list = ls(all.names = TRUE))
  gc()
  datasets = c(1,1.5,2)
}

## CALCULATING STAGE DISTRIBUTION AND CHOOSING OPTIMAL DATASET
datasets = c(1,1.5,2)
for (i in 1:3){
  ds = datasets[i]
  
  if (ds == 1) {
    load(file="V:/UserData/079915/df_1mil")
  } else if (ds == 1.5) {
    load(file="V:/UserData/079915/df_1.5mil")
  } else if (ds == 2) {
    load(file="V:/UserData/079915/df_2mil")
  } 
  
  for (j in 1:5) {
    if (j==1){
      assign((paste0("ds",ds,"mice",j,"percentage")), as.vector(table(df$hb_stage_cat.1))/nrow(df))
    } else if (j==2){
      assign((paste0("ds",ds,"mice",j,"percentage")), as.vector(table(df$hb_stage_cat.2))/nrow(df))
    } else if (j==3){
      assign((paste0("ds",ds,"mice",j,"percentage")), as.vector(table(df$hb_stage_cat.3))/nrow(df))
    } else if (j==4){
      assign((paste0("ds",ds,"mice",j,"percentage")), as.vector(table(df$hb_stage_cat.4))/nrow(df))
    } else if (j==5){
      assign((paste0("ds",ds,"mice",j,"percentage")), as.vector(table(df$hb_stage_cat.5))/nrow(df))
    }
  }
  rm(df)
}

# Create dataframe consisting of the percentages of each cancer per data set
df <- data.frame(ds1.5mice1percentage)
df$ds1.5mice2percentage <- ds1.5mice2percentage
df$ds1.5mice3percentage <- ds1.5mice3percentage # Third lowest absolute difference
df$ds1.5mice4percentage <- ds1.5mice4percentage
df$ds1.5mice5percentage <- ds1.5mice5percentage

df$ds1mice1percentage <- ds1mice1percentage # Second lowest absolute difference
df$ds1mice2percentage <- ds1mice2percentage
df$ds1mice3percentage <- ds1mice3percentage
df$ds1mice4percentage <- ds1mice4percentage
df$ds1mice5percentage <- ds1mice5percentage

df$ds2mice1percentage <- ds2mice1percentage
df$ds2mice2percentage <- ds2mice2percentage
df$ds2mice3percentage <- ds2mice3percentage # First lowest absolute difference
df$ds2mice4percentage <- ds2mice4percentage
df$ds2mice5percentage <- ds2mice5percentage

write.csv(df, "df_percentages") # To allow for manipulations in excel

## ADJUST ORIGINAL DATA SET
# Fill variable in data set
print(paste0("Filling hb in original data set: ", Sys.time()))
load(file="V:/UserData/079915/data_processing_3_data")
load(file="V:/UserData/079915/df_2mil")

fd$hb_stage_cat <- df$hb_stage_cat.3
save(fd, file="V:/UserData/079915/final_imputed_dataset")

load(file="V:/UserData/079915/temp_data_2mil")
# densityplot(tempData, ~hb_stage_cat, mayreplicate=FALSE)
# stripplot(tempData)

