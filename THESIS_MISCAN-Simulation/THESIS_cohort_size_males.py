

"""Fraction of the population that should be in each cohort.
"""
def create_cohort_size(n):
    size = {
        "cohort1" : 0.00501 * n,
        "cohort2" : 0.00182 * n,
        "cohort3" : 0.00232 * n,
        "cohort4" : 0.00467 * n,
        "cohort5" : 0.00139 * n,
        "cohort6" : 0.01061 * n,
        "cohort7" : 0.00237 * n,
        "cohort8" : 0.00762 * n,
        "cohort9" : 0.00022 * n,
        "cohort10" : 0.00742 * n,
        "cohort11" : 0.00243 * n,
        "cohort12" : 0.00005 * n,
        "cohort13" : 0.00029 * n,
        "cohort14" : 0.00648 * n,
        "cohort15" : 0.01233 * n,
        "cohort16" : 0.00161 * n,
        "cohort17" : 0.01155 * n,
        "cohort18" : 0.00297 * n,
        "cohort19" : 0.01129 * n,
        "cohort20" : 0.00521 * n,
        "cohort21" : 0.01184 * n,
        "cohort22" : 0.00959 * n,
        "cohort23" : 0.00117 * n,
        "cohort24" : 0.00722 * n,
        "cohort25" : 0.00914 * n,
        "cohort26" : 0.00152 * n,
        "cohort27" : 0.00153 * n,
        "cohort28" : 0.00155 * n,
        "cohort29" : 0.00155 * n,
        "cohort30" : 0.01275 * n,
        "cohort31" : 0.00151 * n,
        "cohort32" : 0.01324 * n,
        "cohort33" : 0.01393 * n,
        "cohort34" : 0.00041 * n,
        "cohort35" : 0.00834 * n,
        "cohort36" : 0.00998 * n,
        "cohort37" : 0.0005 * n,
        "cohort38" : 0.00057 * n,
        "cohort39" : 0.00058 * n,
        "cohort40" : 0.01416 * n,
        "cohort41" : 0.00058 * n,
        "cohort42" : 0.01482 * n,
        "cohort43" : 0.01544 * n,
        "cohort44" : 0.00924 * n,
        "cohort45" : 0.0153 * n,
        "cohort46" : 0.01593 * n,
        "cohort47" : 0.01641 * n,
        "cohort48" : 0.01607 * n,
        "cohort49" : 0.01677 * n,
        "cohort50" : 0.01647 * n
    }

    return {k: size[k]*(1/0.35575) for k in size}                         
