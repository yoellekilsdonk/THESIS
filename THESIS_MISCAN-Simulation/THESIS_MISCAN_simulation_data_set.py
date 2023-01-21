import numpy as np
import pandas as pd

def main(sex):
    # Load female data frame from MISCAN population run
    if sex == 1:
        df = pd.read_csv('/Users/yoellekilsdonk/Documents/GitHub/thesis-erasmus-mc/MISCAN-Simulation/individual_results_2014_2020_female.csv', sep=',')

        df.rename(columns={'Unnamed: 1': 'sex'}, inplace=True)
        df['sex'] = 1

    # Load male data frame from MISCAN population run
    elif sex == 0:
        df = pd.read_csv('/Users/yoellekilsdonk/Documents/GitHub/thesis-erasmus-mc/MISCAN-Simulation/individual_results_2014_2020_male.csv', sep=',')

        df.rename(columns={'Unnamed: 1': 'sex'}, inplace=True)
        df['sex'] = 0

    # Create state variable and FIT result
    df["state"] = [1 if ele  == "state_1" else (2 if ele ==  "state_2"  else ( 3 if ele == "state_3" else (4 if ele == "state_4" else 0 ))) for ele in df["tag"]]
    df["result"] = df["tag"].apply(lambda x: x.rsplit('_', 1)[1])

    print("num of ids:")
    print(sex)
    print(len(df.individual.unique()))

    df = df.reset_index()

    # Round age variables
    df["age"] = df["age"].astype(int)

    # Create unique ID
    df["cohort"] = df["cohort"].apply(lambda x: x.lstrip("cohort"))
    df["id"]=df["cohort"].astype(int)*10
    df["id"]=(df["id"].astype(str)+df["individual"].astype(str)).astype(int)

    df["id_specific"] = df.groupby(['id']).cumcount()+1
    df["id_specific"] = [1 if (ele  == 2) else (3 if (ele == 4) else (5 if (ele == 6) else (7 if (ele == 8) else ele))) for ele in df["id_specific"]]
    df["id_specific"] = df["id"].astype(str)+"_"+df["id_specific"].astype(str)

    # Create state dataset (all variables except state)
    result = df.drop('state', axis=1)
    result["result"] = [1 if ele  == "positive" else (2 if ele == "negative" else 0) for ele in result["result"]]   # Convert positive FIT result to 1, and negative result to 2 (0 otherwise, never happens)

    result = result.loc[~(result["result"] == 0)] # Delete all 0 observations (these hold no information)
    result["result"] = [0 if ele  == 2 else 1 for ele in result["result"]] # Recode result to match RIVM data 0 favourable 1 unfavourable

    # Create state dataset (only id (for merging) and state)
    state = df.loc[:,("id_specific","state")]
    state = state.loc[~(state["state"] == 0)] # Delete all 0 observations (these hold no information)

    # Merge data set state and result
    merged = pd.merge(result, state, on="id_specific")
    fd = merged.loc[:,("id_specific","age", "result","state", "sex")]
    fd.rename(columns = {"id_specific":"id", "age":"participation_age", "result":"hb_conclusion","state":"hb_stage_cat"}, inplace = True) # Rename to match RIVM data set

    print(fd)

    if sex == 1:
        fd.to_csv('MISCAN_simulation_run_female')

    if sex == 0:
        fd.to_csv('MISCAN_simulation_run_male')

if __name__ == "__main__":
    main(sex=1)
    print("done female")
    main(sex=0)
    print("done male")