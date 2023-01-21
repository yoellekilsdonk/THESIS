## This file creates the minimum and maximum blood value variables.

library("dplyr")
setwd("C:/Users/yoelle.kilsdonk")
load("Z:/inbox/Yoelle/data_processing_2_data")

# One hot encoding current round (FIT)
fd$FIT1 <- with(fd, ifelse(fd$FIT == 1, 1,0) )
fd$FIT2 <- with(fd, ifelse(fd$FIT == 2, 1,0) )
fd$FIT3 <- with(fd, ifelse(fd$FIT == 3, 1,0) )
fd$FIT4 <- with(fd, ifelse(fd$FIT == 4, 1,0) )

# Create FITHB* := hb value per FIT, to obtain maximum and minimum value per id
fd$FITHB1 <- fd$hb*fd$FIT1 # Hb value in FIT1
fd$FITHB2 <- fd$hb*fd$FIT2 # Hb value in FIT2
fd$FITHB3 <- fd$hb*fd$FIT3 # Hb value in FIT3
fd$FITHB4 <- fd$hb*fd$FIT4 # Hb value in FIT4

fd <- fd %>%
  group_by(id)%>%
  mutate(max1=max(FITHB1),
         max2=max(FITHB1,FITHB2), # Maximum of round 1 and 2
         max3=max(max2, FITHB3), # Maximum of round 1,2 and 3
         max4=max(hb), # Maximum of round 1,2,3 and 4 <=> maximum over all hb values 
         min1=max(FITHB1), 
         min2=min(max(FITHB1),max(FITHB2)), # Minimum of round 1 and 2 *NOTE: IF INDIVIDUAL DID NOT PARTICIPATE IN ROUND 2 THIS IS AUTOMATICALLY EQUAL TO ZERO, MAY BE UNTRUE
         min3=min(min2, max(FITHB3)), # Minimum of round 1,2 and 3 *
         min4=min(min3, max(FITHB4))) # Minimum of round 1,2,3 and 4 !<=> minimum over all hb values *

# Create max := maximum attained value up until current round within individual
fd$hb_max <- with(fd, ifelse(fd$FIT == 1, 0, # Default if round == 1, maximum == 0
                          ifelse(fd$FIT == 2, fd$max1, # If round == 2, the maximum value up until now equals the previous value
                                 ifelse(fd$FIT == 3, fd$max2,fd$max3)))) # If round == 3, maximum value equals maximum over all previous values

# Obtain total number of rounds in which an individual participated (necessary for data_processing_3)
max_per_id <- fd %>%
   group_by(id)%>%
   summarize(max_rounds=max(FIT)) # Max rounds is found through the maximum value of FIT (1,2,3,4) per id 
fd <- merge(fd, max_per_id,  by="id")

# Create min := minimum attained value up until current round within individual
# Given the issue described in *NOTE, assignment of this value is subdivided into four possible categories
# max_rounds == 4 to max_rounds == 1
fd$hb_min <- 0 
fd$hb_min[fd$max_rounds == 4]<-  with(fd, ifelse(fd$FIT == 4, fd$min3, 
                                              ifelse(fd$FIT3 == 1, fd$min2,
                                                     ifelse(fd$FIT2 == 1, fd$min1, 0))))[fd$max_rounds == 4] 

fd$hb_min[fd$max_rounds == 3] <-  with(fd, ifelse(fd$FIT3 == 1, fd$min2, 
                                               ifelse(fd$FIT2 == 1, fd$min1, 0)))[fd$max_rounds == 3] 

fd$hb_min[fd$max_rounds == 2] <- with(fd, ifelse(fd$FIT2 == 1, fd$min1, 0))[fd$max_rounds == 2]

# Prepare data frame to only contain relevant variables
fd <- fd[, !grepl("FITHB",names(fd))]

save(fd, file="Z:/inbox/Yoelle/data_processing_3_data")
