"""This branch simulates cohort 1 through 98 from The Netherlands and imitates the Dutch CRC
screening program from 2014 to 2019. The onset of adenomas and preclinical cancers is logged and
saved, and events are logged individually.
"""

# Import packages
import pandas as pd

from datetime import datetime
from panmodel import Universe, processes
from panmodel.tools.cohort import Cohort, PopulationModel

import THESIS_cohort_strategy
import THESIS_crc_screening2
import THESIS_cohort_birth_year #Birth year of individuals in each cohort. Python file Danica.
import THESIS_cohort_size_females #Fraction of the population that should be in each cohort.
import THESIS_cohort_size_males
import THESIS_crc_data
import THESIS_crc_screening_data_males
import THESIS_crc_screening_data_females

cohort_strategies = pd.read_csv('/Users/yoellekilsdonk/Documents/GitHub/thesis-erasmus-mc/MISCAN-Simulation/THESIS_strategies.csv', sep=';')

# Set population size
N = 1000000
N_female = 0.5184 # Percentage of females in RIVM data set
N_male = 1-N_female

# Create birth table, with all individuals born in one year (0.0 percent of the individuals are born at birthyear and all individuals are born by the time its birthyear + 1)
def create_birth_table(birth_year):
    return [(float(0), birth_year), (float(1), birth_year + 1)] #E.g. [(0.0, 1938), (1.0, 1939)] ]

# A function that creates the cohorts
def create_cohort_female(cohort_name):
    # Birth process :: This process generates births for each individual in the simulated population.
    birth = processes.Birth(
        name="birth",
        birth_table=create_birth_table(
            THESIS_cohort_birth_year.birth_year[cohort_name] #Returns birth year per cohort
        )
    )


    # OC process :: Death by other causes
    oc = processes.OC(
        name="oc",
        life_table=processes.oc.data.nl_2019.life_table
        # The life table to draw random death ages from.
        # The leading statistic to calculate the cumulative mortality is the per-year mortality rate q(x),
        # which reports the probability that an individual alive at age x will die before reaching age (x+1).
    )

    # CRC process :: Development of cancer
    crc = processes.CRC(
        name = "crc",

        # Non sex specific variables (inherited from panmodel)
        max_lesions = processes.crc.data.nl.max_lesions,
        localization=processes.crc.data.nl.localization,
        adenoma_small=processes.crc.data.nl.adenoma_small,
        dwell=processes.crc.data.nl.dwell,
        dwell_cancer_shape=processes.crc.data.nl.dwell_cancer_shape,
        survival_group=processes.crc.data.nl.survival_group,
        non_cure=processes.crc.data.nl.non_cure,
        time_to_death=processes.crc.data.nl.time_to_death,

        # All variables below are sex specific, female in this case
        hazard = THESIS_crc_data.hazard_females,
        age = THESIS_crc_data.age_females,
        non_progressive = THESIS_crc_data.non_progressive_females,
        transition_clinical_loc = THESIS_crc_data.transition_clinical_loc_females,
        transition_clinical_age = THESIS_crc_data.transition_clinical_age_females,
    )

    # Specify cutoff to use with crc_screening_data_females tests ***EDITED***
    cutoff = 47

    # CRC screening process :: Dutch screening processes
    crc_screening_process = THESIS_crc_screening2.CRC_screening( # This process simulates screening for colorectal cancer.
        name="crc_screening",
        tests=THESIS_crc_screening_data_females.tests_per_cutoff(cutoff), # Data of the CRC_screening process female ***EDITED***
        #strategy=processes.crc_screening.data.nl.strategy,
        strategy=THESIS_cohort_strategy.create_screening_strategy(cohort_strategies, cohort_name), # User specified strategy process ***EDITED***
        surveillance=processes.crc_screening.data.nl.surveillance # Surveillance process (inherited from panmodel)
    )

    # Create universe
    screening = Universe(
        name="crc_screening",
        processes=[birth, oc, crc, crc_screening_process]
    )

    # Returns cohort with screening universe
    return Cohort(
        name=cohort_name,
        universes=[screening]
    )

def create_cohort_male(cohort_name):
    # Birth process :: This process generates births for each individual in the simulated population.
    birth = processes.Birth(
        name="birth",
        birth_table=create_birth_table(
            THESIS_cohort_birth_year.birth_year[cohort_name] # Returns birth year per cohort
        )
    )

    # OC process :: Death by other causes
    oc = processes.OC(
        name="oc",
        life_table=processes.oc.data.nl_2019.life_table
        # The life table to draw random death ages from.
        # The leading statistic to calculate the cumulative mortality is the per-year mortality rate q(x),
        # which reports the probability that an individual alive at age x will die before reaching age (x+1).
    )

    # CRC process :: Development of cancer
    crc = processes.CRC(
        name = "crc",

        # Non sex specific variables (inherited from panmodel)
        max_lesions = processes.crc.data.nl.max_lesions,
        localization=processes.crc.data.nl.localization,
        adenoma_small=processes.crc.data.nl.adenoma_small,
        dwell=processes.crc.data.nl.dwell,
        dwell_cancer_shape=processes.crc.data.nl.dwell_cancer_shape,
        survival_group=processes.crc.data.nl.survival_group,
        non_cure=processes.crc.data.nl.non_cure,
        time_to_death=processes.crc.data.nl.time_to_death,

        # All variables below are sex specific, male in this case ***EDITED***
        hazard = THESIS_crc_data.hazard_males,
        age = THESIS_crc_data.age_males,
        non_progressive = THESIS_crc_data.non_progressive_males,
        transition_clinical_loc = THESIS_crc_data.transition_clinical_loc_males,
        transition_clinical_age = THESIS_crc_data.transition_clinical_age_males,
    )

    # Specify cutoff to use with crc_screening_data_females tests ***EDITED***
    cutoff = 47

    # CRC screening process :: Dutch screening processes
    crc_screening_process = THESIS_crc_screening2.CRC_screening( # This process simulates screening for colorectal cancer.
        name="crc_screening",
        tests=THESIS_crc_screening_data_males.tests_per_cutoff(cutoff), # Data of the CRC_screening process males ***EDITED***
        #strategy=processes.crc_screening.data.nl.strategy,
        strategy=THESIS_cohort_strategy.create_screening_strategy(cohort_strategies, cohort_name), # User specified strategy process ***EDITED***
        surveillance=processes.crc_screening.data.nl.surveillance # Surveillance process (inherited from panmodel)
    )

    # Create universe
    screening = Universe(
        name="crc_screening",
        processes=[birth, oc, crc, crc_screening_process]
    )

    return Cohort(
        name=cohort_name,
        universes=[screening]
    )

def main(sex): # Sex is a binary indicator for running the process for females (sex==1) or males (o.w.)
    if sex == 1:
        cohort_names = [
            'cohort' + str(int(nr)) for nr in list(range(1, 51)) # List of 'cohort1', ..., 'cohort50'
        ]

        # Create cohorts for all 50 cohorts
        cohorts = [create_cohort_female(x) for x in cohort_names]

        # Model
        model = PopulationModel(cohorts)

        # Run model
        result = model.run(
            n=THESIS_cohort_size_females.create_cohort_size(N*N_female),
            seeds_properties={
                "birth": 1234,
                "oc": 1234,
                "crc": 1234,
                "crc_screening": 1234
            },
            seeds_properties_tmp={
                "crc": 1234
            },
            seeds_random={
                "crc_screening": 1234 # The seeds to use for the random number generators accessible during the simulation.
            },
            log_events_individual=True,
            event_ages=range(101), # The lower bounds of the age groups that should be used in the event log. E.g. [40, 60] would group the events in the following three groups: [0, 40), [40, 60), [60, inf).
            event_years=range(1938, 2115), # The lower bounds of the year groups that should be used in the event log. E.g. [1990, 2000] would group the events in the following three groups: [0, 1990), [1990, 2000), [2000, inf).
            duration_ages=range(101), # The lower bounds of the age groups that should be used in the duration log.
            duration_years=range(1938, 2115), # The lower bounds of the year groups that should be used in the duration log
            verbose=True, # Show steps
            cores="all",
            event_tags=['state_1',
                        'state_2',
                        'state_3',
                        'state_4',
                        'crc_screening_15FIT75_negative',
                        'crc_screening_15FIT75_positive',
                        'crc_screening_15FIT55_negative',
                        'crc_screening_15FIT55_positive',
                        'crc_screening_47FIT75_negative',
                        'crc_screening_47FIT75_positive',
                        'crc_screening_47FIT70_negative',
                        'crc_screening_47FIT70_positive',
                        'crc_screening_47FIT65_negative',
                        'crc_screening_47FIT65_positive',
                        'crc_screening_47FIT60_negative',
                        'crc_screening_47FIT60_positive',
                        'crc_screening_47FIT55_negative',
                        'crc_screening_47FIT55_positive']
        )
        print("right here! (female)", datetime.now())
        result.individual.to_csv("individual_results_2014_2020_female.csv")
        print("still here... (female)", datetime.now())
    else :
        cohort_names = [
            'cohort' + str(int(nr)) for nr in list(range(1, 51))  # List of 'cohort1', ..., 'cohort50'
        ]

        # Create cohorts
        cohorts = [create_cohort_male(x) for x in cohort_names]

        # Model
        model = PopulationModel(cohorts)

        # Run model
        result = model.run(
            n=THESIS_cohort_size_males.create_cohort_size(N*N_male),
            seeds_properties={
                "birth": 1234,
                "oc": 1234,
                "crc": 1234,
                "crc_screening": 1234
            },
            seeds_properties_tmp={
                "crc": 1234
            },
            seeds_random={
                "crc_screening": 1234 # The seeds to use for the random number generators accessible during the simulation.
            },
            log_events_individual=True,
            return_properties= True,
            event_ages=range(101), # The lower bounds of the age groups that should be used in the event log. E.g. [40, 60] would group the events in the following three groups: [0, 40), [40, 60), [60, inf).
            event_years=range(1938, 2115), # The lower bounds of the year groups that should be used in the event log. E.g. [1990, 2000] would group the events in the following three groups: [0, 1990), [1990, 2000), [2000, inf).
            duration_ages=range(101),  # The lower bounds of the age groups that should be used in the duration log.
            duration_years=range(1938, 2115), # The lower bounds of the year groups that should be used in the duration log
            verbose=True,  # Show steps
            cores="all",
            event_tags = [  'state_1',
                            'state_2',
                            'state_3',
                            'state_4',
                            'crc_screening_15FIT75_negative',
                            'crc_screening_15FIT75_positive',
                            'crc_screening_15FIT55_negative',
                            'crc_screening_15FIT55_positive',
                            'crc_screening_47FIT75_negative',
                            'crc_screening_47FIT75_positive',
                            'crc_screening_47FIT70_negative',
                            'crc_screening_47FIT70_positive',
                            'crc_screening_47FIT65_negative',
                            'crc_screening_47FIT65_positive',
                            'crc_screening_47FIT60_negative',
                            'crc_screening_47FIT60_positive',
                            'crc_screening_47FIT55_negative',
                            'crc_screening_47FIT55_positive']
        )
        print("right here! (male)", datetime.now())
        result.individual.to_csv("individual_results_2014_2020_male.csv")
        print("still here... (male)", datetime.now())

    # Export
    # print(result.individual)
    print("should be done by now, right?", datetime.now())

if __name__ == "__main__":
    main(sex=1)
    print("female: done")
    main(sex=0)
    print("male: done")
