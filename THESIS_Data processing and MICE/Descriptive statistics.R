#load(file="V:/UserData/079915/data_processing_3_data")
# 
# # Find number of individuals / obeservations per FIT
# test1 <- length(unique(fd[fd$max_rounds==1,]$id))
# test11 <- fd[fd$max_rounds==1,]
# test2 <- length(unique(fd[fd$max_rounds==2,]$id))
# test12 <- fd[fd$max_rounds==2,]
# test3 <- length(unique(fd[fd$max_rounds==3,]$id))
# test13 <- fd[fd$max_rounds==3,]
# test4 <- length(unique(fd[fd$max_rounds==4,]$id))
# test14 <- fd[fd$max_rounds==4,]
# length(unique(fd$id))
# 
# unfavourable <- table(fd$hb_conclusion)[1]
# favourable <- table(fd$hb_conclusion)[2]
# 
# unfavourable_perc <- unfavourable / (unfavourable+favourable)
# favourable_perc <- 1-unfavourable_perc
# 
# stage1 <- table(fd$hb_stage_cat)[1]
# stage2 <- table(fd$hb_stage_cat)[2]
# stage3 <- table(fd$hb_stage_cat)[3]
# stage4 <- table(fd$hb_stage_cat)[4]
# stageNA <- nrow(fd)-(stage1+stage2+stage3+stage4)
# 
# stage1_perc <- stage1 / (stage1+stage2+stage3+stage4+stageNA)
# stage2_perc <- stage2 / (stage1+stage2+stage3+stage4+stageNA)
# stage3_perc <- stage3 / (stage1+stage2+stage3+stage4+stageNA)
# stage4_perc <- stage4 / (stage1+stage2+stage3+stage4+stageNA)
# stageNA_perc <- 1- (stage1_perc+stage2_perc+stage3_perc+stage4_perc)
# 
# summary(fd$participation_age)
# summary(fd$hb)
# summary(fd$hb_difference)
# summary(fd$hb_max)
# summary(fd$hb_min)
# summary(fd$hb_previous)
# summary(fd$hb_conclusion)
# summary(fd$birthyear)
# summary(fd$id)
# 
# # Unique ids for female/male in the data set
# fd2 <- fd %>%
#   group_by(id) %>%
#   slice(1)
# 
# fem <- table(fd2$sex)[2]
# mal <- table(fd2$sex)[1]
# feml_perc <- fem / (fem+mal)
# mal_perc <- 1-feml_perc
# 
# redish <- rgb(187,95,105, maxColorValue = 255)
# hist(fd$hb, # histogram
#      col=rgb(187,95,105,75, maxColorValue = 255), # column color #peachpuff
#      border="black",
#      prob = TRUE, # show densities instead of frequencies
#      xlab = "Haemoglobin concentration",
#      main = "",
#      xlim = c(0,400))
# lines(density(fd$hb), # density plot
#       lwd = 2, # thickness of line
#       col = rgb(123,53,52, maxColorValue = 255)) #"chocolate3"
# 
# test<-fd[fd$hb>47,]
# hist(test$hb, # histogram
#      col=rgb(187,95,105,75, maxColorValue = 255), # column color
#      border="black",
#      prob = TRUE, # show densities instead of frequencies
#      xlab = "Haemoglobin concentration",
#      main = "",
#      ylim = c(0,0.01), 
#      xlim = c(0,400))
# lines(density(test$hb), # density plot
#       lwd = 2, # thickness of line
#       col = rgb(123,53,52, maxColorValue = 255))

datasets = c(1,2,3)
totalobs_imp = c(0,0,0)

stage1_imp = c(0,0,0)
stage2_imp = c(0,0,0)
stage3_imp = c(0,0,0)
stage4_imp = c(0,0,0)

perc1_imp = c(0,0,0)
perc2_imp = c(0,0,0)
perc3_imp = c(0,0,0)
perc4_imp = c(0,0,0)

result0_imp = c(0,0,0)
result1_imp = c(0,0,0)

percres0_imp = c(0,0,0)
percres1_imp = c(0,0,0)

for (i in 1:3){
  if (datasets[i]==1){
  load(file="V:/UserData/079915/data_imputed_final_1mil")
} else if (datasets[i]==2){
  female = read.csv(file="V:/UserData/079915/MISCAN_simulation_run_female_2mil")
  male = read.csv(file="V:/UserData/079915/MISCAN_simulation_run_male_2mil")
  fd <- rbind(female,male)
  load(file="V:/UserData/079915/data_imputed_final_1.5mil")
} else if (datasets[i]==3){
  load(file="V:/UserData/079915/data_processing_1_data_15")
  load(file="V:/UserData/079915/data_imputed_final_2mil")
}

  stage1_imp[i] <- table(fd$hb_stage_cat)[1]
  stage2_imp[i] <- table(fd$hb_stage_cat)[2]
  stage3_imp[i] <- table(fd$hb_stage_cat)[3]
  stage4_imp[i] <- table(fd$hb_stage_cat)[4]
  perc1_imp[i] <- stage1_imp[i] / (stage1_imp[i]+stage2_imp[i]+stage3_imp[i]+stage4_imp[i])
  perc2_imp[i] <- stage2_imp[i] / (stage1_imp[i]+stage2_imp[i]+stage3_imp[i]+stage4_imp[i])
  perc3_imp[i] <- stage3_imp[i] / (stage1_imp[i]+stage2_imp[i]+stage3_imp[i]+stage4_imp[i])
  perc4_imp[i] <- stage4_imp[i] / (stage1_imp[i]+stage2_imp[i]+stage3_imp[i]+stage4_imp[i])
  totalobs_imp[i] <- (stage1_imp[i]+stage2_imp[i]+stage3_imp[i]+stage4_imp[i])
  
  result0_imp[i] <- table(fd$hb_conclusion)[1]
  result1_imp[i] <- table(fd$hb_conclusion)[2]

  percres0_imp[i] <- result0_imp[i] / (result0_imp[i]+result1_imp[i])
  percres1_imp[i] <- result1_imp[i] / (result1_imp[i]+result0_imp[i])
}