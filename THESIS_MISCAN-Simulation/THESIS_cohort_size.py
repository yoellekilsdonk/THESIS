

"""Fraction of the population that should be in each cohort.
"""
def create_cohort_size(n):
    size = {
        "cohort1" : 0.00548 * n,
        "cohort2" : 0.002 * n,
        "cohort3" : 0.00249 * n,
        "cohort4" : 0.00502 * n,
        "cohort5" : 0.0014 * n,
        "cohort6" : 0.01072 * n,
        "cohort7" : 0.00238 * n,
        "cohort8" : 0.00765 * n,
        "cohort9" : 0.00022 * n,
        "cohort10" : 0.00746 * n,
        "cohort11" : 0.00244 * n,
        "cohort12" : 0.00006 * n,
        "cohort13" : 0.00031 * n,
        "cohort14" : 0.00694 * n,
        "cohort15" : 0.01256 * n,
        "cohort16" : 0.00163 * n,
        "cohort17" : 0.0117 * n,
        "cohort18" : 0.00298 * n,
        "cohort19" : 0.01134 * n,
        "cohort20" : 0.00523 * n,
        "cohort21" : 0.01187 * n,
        "cohort22" : 0.00963 * n,
        "cohort23" : 0.00125 * n,
        "cohort24" : 0.0076 * n,
        "cohort25" : 0.00935 * n,
        "cohort26" : 0.00155 * n,
        "cohort27" : 0.00155 * n,
        "cohort28" : 0.00156 * n,
        "cohort29" : 0.00156 * n,
        "cohort30" : 0.0128 * n,
        "cohort31" : 0.00152 * n,
        "cohort32" : 0.01327 * n,
        "cohort33" : 0.01397 * n,
        "cohort34" : 0.00043 * n,
        "cohort35" : 0.00868 * n,
        "cohort36" : 0.01023 * n,
        "cohort37" : 0.00051 * n,
        "cohort38" : 0.00057 * n,
        "cohort39" : 0.00058 * n,
        "cohort40" : 0.01428 * n,
        "cohort41" : 0.00058 * n,
        "cohort42" : 0.01489 * n,
        "cohort43" : 0.01548 * n,
        "cohort44" : 0.00955 * n,
        "cohort45" : 0.01534 * n,
        "cohort46" : 0.01594 * n,
        "cohort47" : 0.01642 * n,
        "cohort48" : 0.01611 * n,
        "cohort49" : 0.01671 * n,
        "cohort50" : 0.01641 * n        
    }

    return {k: size[k]*(1/0.36019) for k in size}                         
