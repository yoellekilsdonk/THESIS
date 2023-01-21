## This file performs all data pre-proicessing for the `15 threshold` data set. The procedures are similar to `data_processing_1.R` for the
## original data set

# install.packages("dplyr")
library("dplyr")
setwd("C:/Users/yoelle.kilsdonk")
load("Z:/inbox/Yoelle/20220624_fulldataSetGRevaluatie_15.Rdata")

# Calculate age := year of analysis - birthyear
tot.dataSet.15$ronde1_leeftijd_op_analysedatum_MJ <- as.integer(substr(tot.dataSet.15$ronde1_datum_ifobt_analyse_MJ,4,7))-tot.dataSet.15$geboortejaar
tot.dataSet.15$ronde2_leeftijd_op_analysedatum_MJ <- as.integer(substr(tot.dataSet.15$ronde2_datum_ifobt_analyse_MJ,4,7))-tot.dataSet.15$geboortejaar
tot.dataSet.15$ronde3_leeftijd_op_analysedatum_MJ <- as.integer(substr(tot.dataSet.15$ronde3_datum_ifobt_analyse_MJ,4,7))-tot.dataSet.15$geboortejaar
tot.dataSet.15$ronde4_leeftijd_op_analysedatum_MJ <- as.integer(substr(tot.dataSet.15$ronde4_datum_ifobt_analyse_MJ,4,7))-tot.dataSet.15$geboortejaar

# Change participation status to "Deelgenomen" if hb is present in said observation round
tot.dataSet.15[(tot.dataSet.15$ronde1_deelname_status!="Deelgenomen"&!is.na(tot.dataSet.15$ronde1_ifobt_uitslag_bloedwaarde)),c("ronde1_deelname_status")]<-"Deelgenomen" 
tot.dataSet.15[(tot.dataSet.15$ronde2_deelname_status!="Deelgenomen"&!is.na(tot.dataSet.15$ronde2_ifobt_uitslag_bloedwaarde)),c("ronde2_deelname_status")]<-"Deelgenomen"
tot.dataSet.15[(tot.dataSet.15$ronde3_deelname_status!="Deelgenomen"&!is.na(tot.dataSet.15$ronde3_ifobt_uitslag_bloedwaarde)),c("ronde3_deelname_status")]<-"Deelgenomen"
tot.dataSet.15[(tot.dataSet.15$ronde4_deelname_status!="Deelgenomen"&!is.na(tot.dataSet.15$ronde4_ifobt_uitslag_bloedwaarde)),c("ronde4_deelname_status")]<-"Deelgenomen"

generalinfo = subset(tot.dataSet.15, select= c(1,3,4)) # Select ID, geslacht, geboortejaar
generalinfo$geslacht <- with(generalinfo, ifelse(generalinfo$geslacht == "VROUW", 1,0)) # Convert VROUW to 1, and 0 otherwise

for (i in 1:4) {
  df = tot.dataSet.15[, grepl(sprintf("ronde%s",i),names(tot.dataSet.15))] # Only keep variables relevant for round 1
  names(df) = gsub(i,"", as.character(names(df))) # Remove 1,2,3,4 from variable names
  
  # Delete irrelevant variables
  df = df[, !grepl("datum_initiele_uitnodiging_MJ",names(df))]
  df = df[, !grepl("datum_ifobt_analyse",names(df))]
  df = df[, !grepl("ifobt_meeteenheid",names(df))]
  df = df[, !grepl("datum_eerste_mdl_verslag_MJ",names(df))]
  df = df[, !grepl("incidentie_MJ",names(df))]
  df = df[, !grepl("interval_of_screendetected",names(df))]
  df = df[, !grepl("tnm",names(df))]
  df = df[, !grepl("topografie",names(df))]
  df = df[, !grepl("morfologie",names(df))]
  df = df[, !grepl("_dagen_",names(df))]
  
  df = cbind(df,generalinfo) # Add general information to data set
  
  # Transform values of 88 threshold to 15 threshold (1/(275/47)) and round to one decimal
  df$ronde_ifobt_uitslag_bloedwaarde <- round(with(df, ifelse(df$ronde_ifobt_drempelwaarde %in% 88, df$ronde_ifobt_uitslag_bloedwaarde*0.170909,df$ronde_ifobt_uitslag_bloedwaarde )),1) 
  
  # Convert deelgenomen to 1, and 0 otherwise
  df$ronde_deelname_status <- as.integer(with(df, ifelse(df$ronde_deelname_status == "Deelgenomen", 1,0))) 
  
  # Convert Gunstig to 1 & Ongunstig, ongunstig (onbetrouwbaar) to 0 & NA, (nog) niet ontvangen, gunstig (onbetrouwbaar), onbeoordeelbaar to NA
  df$ronde_ifobt_conclusie <- recode(df$ronde_ifobt_conclusie, "Gunstig"=1, "Ongunstig"=0, "Ongunstig (onbetrouwbaar)"=0,"Gunstig (onbetrouwbaar)"=as.numeric(NA), "(Nog) niet Ontvangen"=as.numeric(NA), "Onbeoordeelbaar"= as.numeric(NA)) 
  
  if (i==1){
    data1 <- df
  }
  if (i==2){
    data2 <- df
    fd <- rbind(data1,data2) # Row bind first & second data set
    rm(data1) # Remove data sets (for sake of memory)
    rm(data2)
  }
  if (i==3){
    data3 <- df
    fd <-rbind(fd, data3)
    rm(data3)
  }
  if (i==4){
    data4 <- df
    fd <-rbind(fd, data4)
    rm(data4)
  }
  rm(df)  
}
rm(generalinfo)
rm(tot.dataSet.15)

# Rename final dataframe
names(fd) <- list("current_round", "participation", "participation_age", "threshold", "hb", "hb_conclusion", "hb_stage", "hb_stage_cat", "hb_stadium", "id", "sex", "birthyear")
fd$hb_stage_cat <- recode(fd$hb_stage_cat, "1"=4, "2"=3, "3"=2, "4"=as.numeric(NA),"5"=as.numeric(NA), "6"=as.numeric(NA), "7"=as.numeric(NA), "8"=1, "9"=as.numeric(NA)) 

fd <- fd[!is.na(fd$hb),] # Delete observations without hb value
fd <- fd[!is.na(fd$hb_stage_cat),] # Delete observations without known stage

save(fd, file="Z:/inbox/Yoelle/data_processing_1_data_15")