"""Functions to create screening strategies
"""

import numpy as np


# Function to create a screening strategy based on cohort information
def create_screening_strategy(df, cohort_name):
    indices = np.where(df['name'] == cohort_name)
    cohort_df = df.iloc[indices]

    strategy = []
    for index, intervention in cohort_df.iterrows():
        strategy.append(fill_screening_strategy(intervention['participation_first'], intervention['participation_participator'],
                                                intervention['participation_non_participator'],
                                                intervention['participation_diag'],
                                                intervention['test'], intervention['age']))

    return strategy


# Function to fill a screening strategy based on cohort information
def fill_screening_strategy(participation_first, participator, non_participator, participation_surveillance, test, age):
    strategy = {
        "age": float(age),
        "test": test,
        "participation_first": participation_first,
        "participation_participator": participator,
        "participation_non_participator": non_participator,
        "participation_diag": participation_surveillance,
        "surveillance": {
            "positive": "diagnostic_colonoscopy"
        }
    }

    return strategy
