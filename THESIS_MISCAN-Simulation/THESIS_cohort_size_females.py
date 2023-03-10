

"""Fraction of the population that should be in each cohort.
"""
def create_cohort_size(n):
    size = {
        "cohort1" : 0.00538 * n,
        "cohort2" : 0.00217 * n,
        "cohort3" : 0.00267 * n,
        "cohort4" : 0.00538 * n,
        "cohort5" : 0.00142 * n,
        "cohort6" : 0.01083 * n,
        "cohort7" : 0.00239 * n,
        "cohort8" : 0.00769 * n,
        "cohort9" : 0.00022 * n,
        "cohort10" : 0.00749 * n,
        "cohort11" : 0.00245 * n,
        "cohort12" : 0.00007 * n,
        "cohort13" : 0.00033 * n,
        "cohort14" : 0.0074 * n,
        "cohort15" : 0.0128 * n,
        "cohort16" : 0.00165 * n,
        "cohort17" : 0.01184 * n,
        "cohort18" : 0.00299 * n,
        "cohort19" : 0.01139 * n,
        "cohort20" : 0.00525 * n,
        "cohort21" : 0.0119 * n,
        "cohort22" : 0.00967 * n,
        "cohort23" : 0.00133 * n,
        "cohort24" : 0.00799 * n,
        "cohort25" : 0.00956 * n,
        "cohort26" : 0.00158 * n,
        "cohort27" : 0.00157 * n,
        "cohort28" : 0.00157 * n,
        "cohort29" : 0.00156 * n,
        "cohort30" : 0.01286 * n,
        "cohort31" : 0.00153 * n,
        "cohort32" : 0.01331 * n,
        "cohort33" : 0.014 * n,
        "cohort34" : 0.00045 * n,
        "cohort35" : 0.00901 * n,
        "cohort36" : 0.01049 * n,
        "cohort37" : 0.00053 * n,
        "cohort38" : 0.00057 * n,
        "cohort39" : 0.00059 * n,
        "cohort40" : 0.0144 * n,
        "cohort41" : 0.00058 * n,
        "cohort42" : 0.01495 * n,
        "cohort43" : 0.01552 * n,
        "cohort44" : 0.00986 * n,
        "cohort45" : 0.01538 * n,
        "cohort46" : 0.01594 * n,
        "cohort47" : 0.01642 * n,
        "cohort48" : 0.01615 * n,
        "cohort49" : 0.01665 * n,
        "cohort50" : 0.01635 * n
    }

    return {k: size[k]*(1/0.36408) for k in size}                         

