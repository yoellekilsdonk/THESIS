## This file creates a lagged dependent variable, and the FIT variable which denotes the sequence number of the current FIT.
## It also deletes aberrant observations. Finally, this file ensures that only individuals who participate in consecutive rounds 
## (or one round maximum) are included. 

library("dplyr")
setwd("C:/Users/yoelle.kilsdonk")
load(file="Z:/inbox/Yoelle/data_processing_1_data")

## Construct FIT variable
# Create participation_FIT := Sequence number of actual FITs taken
# Create participation_year := Years between FITs where Round 1 = 0, Round 2 = 2, Round 3 = 4, Round 4 = 6, to track years between tests (not used in analysis now)
fd$FIT <- NA_integer_
fd$participation_year =NA_integer_

for (i in 1:4){
  index <- fd %>% 
    mutate(rn = row_number()) %>%
    group_by(id) %>%
    summarize (n=rn[current_round==nth(na.omit(current_round),i)]) # n := row number where (current_round == ith smallest value in current_round)
    # e.g., when i == 1, this chunck of code finds the row index of the 1st smallest value (i.e., the first round in which an individual participates)
    # when i == 4, this finds the row index of the 4th smallest value (i.e., the last round in which an individual participates if max rounds == 4, otherwise this equals NA)
  
  fd$FIT <- replace(fd$FIT, as.vector(index$n), i) # replace fd$FIT by i (:= 1,2,3,4 depending on which iteration you're in) on index 'index$n' where NA is skipped (described above)
  fd$participation_year[(fd$current_round==i)]=2*(i-1) # if i=1 -> 0, if i=2 -> 2, if i=3 -> 4, if i=4 -> 6
}

## Construct lagged dependent variable
# Create hb_difference := difference between current hb value and previous hb value (0 for FIT==1)
# Create hb_previous := hb value attained during previous fit (0 for FIT==1)
fd$hb_difference <- ave(fd$hb, fd$id, FUN=function(x) c(0,diff(x))) # will not be used itself
fd$hb_previous <- with(fd, ifelse(fd$FIT == 1, 0,fd$hb-fd$hb_difference))

## Delete aberrant observations 
fd <- fd[!fd$hb_previous>47,] # Delete observations of individuals who receive another FIT after unfavorable conclusion, 
fd <- fd[!is.na(fd$hb),] # Delete observations without hb value
fd <- fd[!is.na(fd$hb_conclusion),] # Delete observations without hb_conclusion
fd <- fd[!((fd$hb_conclusion==0) & is.na(fd$hb_stage_cat)),] # Delete observations with positive (unfavorable) test but unknown stage

## Only include individuals who participate in consecutive rounds
# Check if current_round is sequential (not true if difference between values within id >1)
consecutive_index <- fd %>%
  mutate(rn = row_number()) %>%
  group_by(id)%>%
  summarize(n=rn[diff(current_round,1)>1]) # returns the id and row numbers where the current_round is not sequential

# Create consecutive := dummy variable which is equal to one if individual participated in consecutive rounds
fd$consecutive <- with(fd, ifelse(is.element(fd$id, unique(consecutive_index$id)), 0, 1)) # set consecutive to 0 if fd$id occurs in consecturive_index$id
fd <- fd[fd$consecutive==1,] # Only keep individuals with exclusively consecutive participation observations

save(fd, file="Z:/inbox/Yoelle/data_processing_2_data")