## This file transforms the original RIVVM data set from a horizontal structure (per id) to a vertical structure (per round)
## along with some (minor) data engineering

#install.packages("dplyr")
library("dplyr")
setwd("C:/Users/yoelle.kilsdonk")
load("Z:/inbox/Yoelle/20211115_fulldataSetGRevaluatie_47.Rdata")

tot.dataSet <- tot.dataSet[!tot.dataSet$ronde1_datum_initiele_uitnodiging_MJ=="12/2099",] # Remove individuals with invalid date for round 1 (other rounds have valid dates)

# Calculate age := year of analysis - birthyear
tot.dataSet$ronde1_leeftijd_op_analysedatum_MJ <- as.integer(substr(tot.dataSet$ronde1_datum_ifobt_analyse_MJ,4,7))-tot.dataSet$geboortejaar
tot.dataSet$ronde2_leeftijd_op_analysedatum_MJ <- as.integer(substr(tot.dataSet$ronde2_datum_ifobt_analyse_MJ,4,7))-tot.dataSet$geboortejaar
tot.dataSet$ronde3_leeftijd_op_analysedatum_MJ <- as.integer(substr(tot.dataSet$ronde3_datum_ifobt_analyse_MJ,4,7))-tot.dataSet$geboortejaar
tot.dataSet$ronde4_leeftijd_op_analysedatum_MJ <- as.integer(substr(tot.dataSet$ronde4_datum_ifobt_analyse_MJ,4,7))-tot.dataSet$geboortejaar

# Change participation status to "Deelgenomen" if hb is present in said observation round
tot.dataSet[(tot.dataSet$ronde1_deelname_status!="Deelgenomen"&!is.na(tot.dataSet$ronde1_ifobt_uitslag_bloedwaarde)),c("ronde1_deelname_status")]<-"Deelgenomen" 
tot.dataSet[(tot.dataSet$ronde2_deelname_status!="Deelgenomen"&!is.na(tot.dataSet$ronde2_ifobt_uitslag_bloedwaarde)),c("ronde2_deelname_status")]<-"Deelgenomen"
tot.dataSet[(tot.dataSet$ronde3_deelname_status!="Deelgenomen"&!is.na(tot.dataSet$ronde3_ifobt_uitslag_bloedwaarde)),c("ronde3_deelname_status")]<-"Deelgenomen"
tot.dataSet[(tot.dataSet$ronde4_deelname_status!="Deelgenomen"&!is.na(tot.dataSet$ronde4_ifobt_uitslag_bloedwaarde)),c("ronde4_deelname_status")]<-"Deelgenomen"

# Delete aberrant individuals with age <55 or >77 in round 1
tot.dataSet <- tot.dataSet[!(tot.dataSet$ronde1_leeftijd_op_analysedatum_MJ<=54|tot.dataSet$ronde1_leeftijd_op_analysedatum_MJ>=78),]

generalinfo = subset(tot.dataSet, select= c(1,3,4)) # Select ID, geslacht, geboortejaar
generalinfo$geslacht <- with(generalinfo, ifelse(generalinfo$geslacht == "VROUW", 1,0)) # Convert VROUW to 1, and 0 otherwise

for (i in 1:4) {
  df = tot.dataSet[, grepl(sprintf("ronde%s",i),names(tot.dataSet))] # Only keep variables relevant for round i
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
  
  # Transform values of 275 threshold to 47 threshold (1/(275/47)) and round to one decimal
  df$ronde_ifobt_uitslag_bloedwaarde <- round(with(df, ifelse(df$ronde_ifobt_drempelwaarde %in% 275, df$ronde_ifobt_uitslag_bloedwaarde*0.170909,df$ronde_ifobt_uitslag_bloedwaarde )),1) 
  
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
rm(tot.dataSet)

# Rename final data frame
names(fd) <- list("current_round", "participation", "participation_age", "threshold", "hb", "hb_conclusion", "hb_stage", "hb_stage_cat", "hb_stadium", "id", "sex", "birthyear")
fd$hb_stage_cat <- recode(fd$hb_stage_cat, "1"=4, "2"=3, "3"=2, "4"=as.numeric(NA),"5"=as.numeric(NA), "6"=as.numeric(NA), "7"=as.numeric(NA), "8"=1, "9"=as.numeric(NA)) 

save(fd, file="Z:/inbox/Yoelle/data_processing_1_data")
