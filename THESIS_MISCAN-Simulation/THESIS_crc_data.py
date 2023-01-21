"""Data for the CRC process calibrated on NL observations.

References:

- Arminski TC, McLean DW. Incidence and Distribution of Adenomatous Polyps of the Colon and Rectum
  Based on 1,000 Autopsy Examinations. Dis Colon Rectum. 1964;7:249-61.[1]
- Blatt L. Polyps of the Colon and Rectum: Incidence and Distribution. Dis Colon Rectum.
  1961;4:277-82.[2]
- Bombi JA. Polyps of the colon in Barcelona, Spain. An autopsy study. Cancer.
  1988;61(7):1472-6.[3]
- Chapman I. Adenomatous polypi of large intestine: incidence and distribution. Ann Surg.
  1963;157:223-6.[4]
- Clark JC, Collan Y, Eide TJ, Esteve J, Ewen S, Gibbs NM, et al. Prevalence of polyps in an
  autopsy series from areas with varying incidence of large-bowel cancer. Int J Cancer.
  1985;36(2):179-86.[5]
- Jass JR, Young PJ, Robinson EM. Predictors of presence, multiplicity, size and dysplasia of
  colorectal adenomas. A necropsy study in New Zealand. Gut. 1992;33(11):1508-14.[6]
- Johannsen LG, Momsen O, Jacobsen NO. Polyps of the large intestine in Aarhus, Denmark.
  An autopsy study. Scand J Gastroenterol. 1989;24(7):799-806.[7]
- Rickert RR, Auerbach O, Garfinkel L, Hammond EC, Frasca JM. Adenomatous lesions of the large
  bowel: an autopsy survey. Cancer. 1979;43(5):1847-57.[8]
- Vatn MH, Stalsberg H. The prevalence of polyps of the large intestine in Oslo: an autopsy study.
  Cancer. 1982;49(4):819-25.[9]
- Williams AR, Balasooriya BA, Day DW. Polyps and cancer of the large bowel: a necropsy study in
  Liverpool. Gut. 1982;23(10):835-42.[10]
- Stoop EM, de Haan MC, de Wijkerslooth TR, Bossuyt PM, van Ballegooijen M, Nio CY, et al.
  Participation and yield of colonoscopy versus non-cathartic CT colonography in population-based
  screening for colorectal cancer: a randomised controlled trial. Lancet Oncol. 2011;13(1):55-64.
  [11]
- Atkin WS, Edwards R, Kralj-Hans I, Wooldrage K, Hart AR, Northover JM, et al. Once-only flexible
  sigmoidoscopy screening in prevention of colorectal cancer: a multicentre randomised controlled
  trial. Lancet. 2010;375(9726):1624-33.[12]
"""

max_lesions = 50
"""The maximum number of lesions in an individual is an assumption, loosely based on the fact that
we talk about syndromes rather than average risk population in the case of over 50 adenomas.
"""

hazard = {
    "mean": 1.,
    "variance": 2.66755
}

hazard_females = {
    'mean': 1.0,
    'variance': 3.813755128207963
}

hazard_males = {
    'mean': 1.0,
    'variance': 2.745006633512538
}
"""The individual hazards are calibrated based on overall CRC incidence between 2009 and 2013
provided by the Netherlands Cancer Registry (IKNL) and adenoma prevalence (based on international
literature [1]-[10]).
"""

age = [
    (0, 0.000000000),
    (20, 0.011139400),
    (25, 0.039039050),
    (30, 0.068335150),
    (35, 0.099628550),
    (40, 0.172521550),
    (45, 0.296247050),
    (50, 0.478152050),
    (55, 0.690815550),
    (60, 0.903479050),
    (65, 1.116143550),
    (70, 1.401645050),
    (75, 1.751546050),
    (80, 1.751737712),
    (85, 1.751773149),
    (100, 1.751846890)
]

age_females = [
    (0, 0.0),
    (20, 0.033906227396442654),
    (25, 0.05921586074841467),
    (30, 0.09075084283534136),
    (35, 0.11855234689730763),
    (40, 0.20578029906778997),
    (45, 0.3702500285333413),
    (50, 0.5642499079280354),
    (55, 0.7214228526859956),
    (60, 1.0125241819365813),
    (65, 1.2703603537758876),
    (70, 1.4948623146607132),
    (75, 1.9866430496000995),
    (80, 2.0592352998938837),
    (85, 2.071496236261969),
    (100, 2.074760285814102)
]

age_males = [
    (0, 0.0),
    (20, 0.002634509451170698),
    (25, 0.026545051786762648),
    (30, 0.06422582892317527),
    (35, 0.08473359050970314),
    (40, 0.09805954639376122),
    (45, 0.2963654873063879),
    (50, 0.3834261145162611),
    (55, 0.6118119122616826),
    (60, 0.8182306619549826),
    (65, 1.1378663592281073),
    (70, 1.4616000177248747),
    (75, 1.7103379122936029),
    (80, 2.102533207591088),
    (85, 2.132567810653789),
    (100, 2.2758501941851836)
]
"""The age factors which determine the age of each onset are derived from the calibration on CRC
incidence between 2009 and 2013 provided by the Netherlands Cancer Registry (IKNL) and adenoma
prevalence (based on international literature [1]-[10]).
"""

localization = [
    (0.0000, 0),
    (0.3020, 1),
    (0.3319, 2),
    (0.6028, 3),
    (0.6332, 4),
    (0.7544, 5),
    (0.8607, 6),
    (1.0000, 7)
]
"""The localization of lesions is based on CRC incidence between 2009 and 2013 provided by the
Netherlands Cancer Registry (IKNL).
"""

non_progressive = [
    (0, 0.999001135),
    (45, 0.801220306),
    (65, 0.32037),
    (100, 0.07261)
]

non_progressive_females = [
    (0, 0.9992899822256381),
    (45, 0.8390899543849797),
    (65, 0.518768803609899),
    (100, 0.31477324980118415)
]

non_progressive_males = [
    (0, 0.9990413713437812),
    (45, 0.7463871450993881),
    (65, 0.21330482585781751),
    (100, 0.0)
]
"""The probability that an adenoma is non-progressive is calibrated to CRC incidence between
2009 and 2013 (provided by the Netherlands Cancer Registry, IKNL) and adenoma prevalence (based on
international literature [1]-[10]), after calibration of adenoma dwell time based on
screening trials [12].
"""

adenoma_small = {
    "progressive": .30,
    "non_progressive": .33
}
"""The estimates of the probability that an adenoma grows until it is 6-9mm, were fit to the size
distribution of adenomas in the COCOS trial (corrected for colonoscopy sensitivity [11]).
"""

dwell = {
    "ad51": 34.,
    "ad52": 34.,
    "npl_ad6_9": 34.,
    "ad53": 98.9009,
    "ad54": 68.7396,
    "pl_ad6_9": 28.7584,
    "p_ad10": 42.7869,
    "p_ad6_9": 41.3841,
    "pcl1a": 2.0738,
    "pcl1b": 2.0738,
    "pcl2a": 0.9101,
    "pcl2b": 0.9101,
    "pcl3a": 2.3891,
    "pcl3b": 2.3891,
    "pcl4": 0.8110
}
"""Dwell times of non-progressive adenomas were based on the COCOS trial [11]. The average
duration between the onset of a progressive adenoma and the transition to pre-clinical cancer was
based on the interval cancer rates after a once-only sigmoidoscopy in an RCT in the UK [12]. Dwell
times in preclinical cancer stages were calibrated to screen-detected and interval cancer rates
in the Dutch CRC screening program between 2014 and 2017.
"""

dwell_cancer_shape = 2.4413
"""The shape parameter of the Weibull distribution used to determine individual dwell times is
calibrated to screen-detected and interval cancer rates in the Dutch CRC screening program
between 2014 and 2017.
"""

transition_clinical_loc = {
    1: [
        (2, 0.319332),
        (2, 0.350276),
        (4, 0.350276),
        (4, 0.236436)
    ],
    2: [
        (2, 0.364663),
        (2, 0.490188),
        (4, 0.490188),
        (4, 0.531819)
    ],
    3: [
        (2, 0.644464),
        (2, 0.512545),
        (4, 0.512545),
        (4, 0.496934)
    ]
}

transition_clinical_loc_females = {
    1: [
        (2, 0.29056770302925233),
        (2, 0.3326284440121514),
        (4, 0.3326284440121514),
        (4, 0.20833500683989947)
    ],
    2: [
        (2, 0.37147107224730047),
        (2, 0.4842151658457414),
        (4, 0.4842151658457414),
        (4, 0.5308361664971509)
    ],
    3: [
        (2, 0.694425594410414),
        (2, 0.5821983747850524),
        (4, 0.5821983747850524),
        (4, 0.5911540034459355)
    ]
}

transition_clinical_loc_males = {
    1: [
        (2, 0.25025454119906754),
        (2, 0.2956774011200143),
        (4, 0.2956774011200143),
        (4, 0.1971282130823439)
    ],
    2: [
        (2, 0.31296928866457496),
        (2, 0.43477640295502207),
        (4, 0.43477640295502207),
        (4, 0.4784503611142632)
    ],
    3: [
        (2, 0.6819190622815409),
        (2, 0.5356780838658767),
        (4, 0.5356780838658767),
        (4, 0.5199021839013239)
    ]
}
"""Transition probabilities based on location were based on survival data obtained by the
Netherlands Cancer Registry (IKNL) between 2010 and 2014, stratified by age group, location
and stage distribution.
"""

transition_clinical_age = {
    1: [
        (0, 0.002956),
        (35, 0.171849),
        (40, 0.193023),
        (45, 0.213459),
        (50, 0.233156),
        (55, 0.252115),
        (60, 0.270335),
        (65, 0.287817),
        (70, 0.304560),
        (75, 0.320565),
        (80, 0.335832),
        (85, 0.35036),
        (100, 0.389513)
    ], 2: [
        (0, 0.000163),
        (35, 0.168605),
        (40, 0.192668),
        (45, 0.216731),
        (50, 0.240794),
        (55, 0.264857),
        (60, 0.288920),
        (65, 0.312983),
        (70, 0.337046),
        (75, 0.36111),
        (80, 0.385173),
        (85, 0.409236),
        (100, 0.481425)
    ], 3: [
        (0, 0.287838),
        (35, 0.450995),
        (40, 0.467047),
        (45, 0.481284),
        (50, 0.493707),
        (55, 0.504316),
        (60, 0.513111),
        (65, 0.520092),
        (70, 0.525259),
        (75, 0.528611),
        (80, 0.53015),
        (85, 0.529874),
        (100, 0.518162)
    ]
}

transition_clinical_age_females = {
    1: [
        (0, 0.04593922846896444),
        (35, 0.2123179045567085),
        (40, 0.23342743018895948),
        (45, 0.2538722416547109),
        (50, 0.27365233895396274),
        (55, 0.2927677220867151),
        (60, 0.31121839105296795),
        (65, 0.3290043458527212),
        (70, 0.34612558648597486),
        (75, 0.362582112952729),
        (80, 0.37837392525298363),
        (85, 0.39350102338673865),
        (100, 0.4348940327890066)
    ],
    2: [
        (0, 0.004190755739658242),
        (35, 0.17775906271386127),
        (40, 0.20255453513874744),
        (45, 0.2273500075636336),
        (50, 0.2521454799885197),
        (55, 0.2769409524134059),
        (60, 0.301736424838292),
        (65, 0.3265318972631782),
        (70, 0.35132736968806433),
        (75, 0.3761228421129505),
        (80, 0.40091831453783666),
        (85, 0.4257137869627227),
        (100, 0.5001002042373812)
    ],
    3: [
        (0, 0.33416191552200414),
        (35, 0.4071423733020547),
        (40, 0.4151540197113574),
        (45, 0.42256213280226945),
        (50, 0.4293667125747912),
        (55, 0.4355677590289225),
        (60, 0.44116527216466334),
        (65, 0.4461592519820138),
        (70, 0.4505496984809738),
        (75, 0.4543366116615433),
        (80, 0.4575199915237223),
        (85, 0.46009983806751104),
        (100, 0.4642181777885343)
    ]
}

transition_clinical_age_males = {
    1: [
        (0, 0.002437332219560597),
        (35, 0.21313022735632028),
        (40, 0.23865253249744092),
        (45, 0.26303066766895744),
        (50, 0.2862646328708698),
        (55, 0.3083544281031781),
        (60, 0.32930005336588225),
        (65, 0.3491015086589823),
        (70, 0.3677587939824782),
        (75, 0.38527190933637),
        (80, 0.40164085472065764),
        (85, 0.4168656301353412),
        (100, 0.4556749365617673)
    ],
    2: [
        (0, 0.0040047748009999495),
        (35, 0.19211927322688693),
        (40, 0.21899277300201364),
        (45, 0.24586627277714038),
        (50, 0.27273977255226706),
        (55, 0.2996132723273938),
        (60, 0.3264867721025205),
        (65, 0.35336027187764724),
        (70, 0.3802337716527739),
        (75, 0.4071072714279006),
        (80, 0.43398077120302736),
        (85, 0.46085427097815407),
        (100, 0.5414747703035342)
    ],
    3: [
        (0, 0.3424118795347559),
        (35, 0.4464474494074705),
        (40, 0.4558744058741257),
        (45, 0.4639425453905622),
        (50, 0.4706518679567798),
        (55, 0.47600237357277864),
        (60, 0.47999406223855856),
        (65, 0.48262693395411965),
        (70, 0.4839009887194619),
        (75, 0.4838162265345852),
        (80, 0.4823726473994899),
        (85, 0.4795702513141755),
        (100, 0.46301016135691964)
    ]
}
"""Transition probabilities based on age were based on survival data obtained by the Netherlands
Cancer Registry (IKNL) between 2010 and 2014, stratified by age group, location and stage
distribution.
"""

survival_group = [
    (2., "rectum"),
    (7., "colon")
]
"""The survival groups were based on survival data obtained by the Netherlands Cancer Registry
(IKNL) between 2010 and 2014, stratified by age group, location and stage distribution.
"""

non_cure = {
    0: {
        "rectum": [
            (0, 0.),
            (100, 0.)
        ],
        "colon": [
            (0, 0.),
            (100, 0.)
        ]
    },
    1: {
        "rectum": [
            (0, 0.1184),
            (45, 0.1184),
            (45, 0.0780),
            (55, 0.0780),
            (55, 0.0628),
            (65, 0.0628),
            (65, 0.1034),
            (75, 0.1034),
            (75, 0.0438),
            (100, 1.0000)
        ],
        "colon": [
            (0, 0.1658),
            (45, 0.1658),
            (45, 0.1162),
            (55, 0.1162),
            (55, 0.1198),
            (65, 0.1198),
            (65, 0.1566),
            (75, 0.1566),
            (75, 0.1698),
            (100, 1.0000)
        ]
    },
    2: {
        "rectum": [
            (0, 0.1056),
            (45, 0.1056),
            (45, 0.1898),
            (55, 0.1898),
            (55, 0.2068),
            (65, 0.2068),
            (65, 0.2526),
            (75, 0.2526),
            (75, 0.1648),
            (100, 1.0000)
        ],
        "colon": [
            (0, 0.2328),
            (45, 0.2328),
            (45, 0.2914),
            (55, 0.2914),
            (55, 0.3164),
            (65, 0.3164),
            (65, 0.3266),
            (75, 0.3266),
            (75, 0.4202),
            (100, 1.0000)

        ]
    },
    3: {
        "rectum": [
            (0, 0.3076),
            (45, 0.3076),
            (45, 0.3552),
            (55, 0.3552),
            (55, 0.3676),
            (65, 0.3676),
            (65, 0.3890),
            (75, 0.3890),
            (75, 0.4698),
            (100, 1.0000)
        ],
        "colon": [
            (0, 0.3832),
            (45, 0.3832),
            (45, 0.3524),
            (55, 0.3524),
            (55, 0.3648),
            (65, 0.3648),
            (65, 0.4508),
            (75, 0.4508),
            (75, 0.7520),
            (100, 1.0000)
        ]
    },
    4: {
        "rectum": [
            (0, 0.8682),
            (45, 0.8682),
            (45, 0.9476),
            (55, 0.9476),
            (55, 0.9524),
            (65, 0.9524),
            (65, 0.9548),
            (75, 0.9548),
            (75, 0.9622),
            (100, 1.0000)
        ],
        "colon": [
            (0, 0.8460),
            (45, 0.8460),
            (45, 0.8998),
            (55, 0.8998),
            (55, 0.9206),
            (65, 0.9206),
            (65, 0.9370),
            (75, 0.9370),
            (75, 0.9422),
            (100, 1.0000)
        ]
    }
}
"""The age-dependent probability that a lesion is not cured was based on survival data obtained by
the Netherlands Cancer Registry (IKNL) between 2010 and 2014, stratified by age group, location
and stage distribution.
"""

time_to_death = {
    1: {
        "rectum": [
            (0.000000, 0),
            (0.215364, 1),
            (0.316335, 2),
            (0.434762, 3),
            (0.549364, 4),
            (0.603027, 5),
            (0.623328, 6),
            (0.721797, 7),
            (0.774188, 8),
            (0.874856, 9),
            (0.906829, 10),
            (0.906829, 11),
            (0.906829, 12),
            (1.000000, 13)
        ],
        "colon": [
            (0.0000, 0),
            (0.1174, 1),
            (0.1569, 2),
            (0.2386, 3),
            (0.3546, 4),
            (0.4485, 5),
            (0.5376, 6),
            (0.6485, 7),
            (0.6936, 8),
            (0.7163, 9),
            (0.7889, 10),
            (0.8255, 11),
            (0.8260, 12),
            (0.8260, 13),
            (1.0000, 14)
        ]
    },
    2: {
        "rectum": [
            (0.0000, 0),
            (0.1942, 1),
            (0.2871, 2),
            (0.4003, 3),
            (0.4793, 4),
            (0.5716, 5),
            (0.6582, 6),
            (0.7257, 7),
            (0.7655, 8),
            (0.8043, 9),
            (0.8733, 10),
            (0.9157, 11),
            (0.9157, 12),
            (0.9157, 13),
            (0.9157, 14),
            (1.0000, 15)
        ],
        "colon": [
            (0.0000, 0),
            (0.1286, 1),
            (0.2128, 2),
            (0.2901, 3),
            (0.3978, 4),
            (0.4516, 5),
            (0.5547, 6),
            (0.6150, 7),
            (0.7144, 8),
            (0.7353, 9),
            (0.8047, 10),
            (0.8544, 11),
            (0.9234, 12),
            (0.9797, 13),
            (0.9797, 14),
            (1.0000, 15)

        ]
    },
    3: {
        "rectum": [
            (0.0000, 0),
            (0.2380, 1),
            (0.4253, 2),
            (0.5881, 3),
            (0.7078, 4),
            (0.7908, 5),
            (0.8642, 6),
            (0.8846, 7),
            (0.9353, 8),
            (0.9555, 9),
            (0.9749, 10),
            (0.9982, 11),
            (1.0000, 12)
        ],
        "colon": [
            (0.0000, 0),
            (0.1198, 1),
            (0.2419, 2),
            (0.3335, 3),
            (0.4658, 4),
            (0.5725, 5),
            (0.6666, 6),
            (0.7442, 7),
            (0.7792, 8),
            (0.8205, 9),
            (0.8621, 10),
            (0.8621, 11),
            (0.8938, 12),
            (0.9821, 13),
            (1.0000, 14)
        ]
    },
    4: {
        "rectum": [
            (0.0000, 0),
            (0.5284, 1),
            (0.7474, 2),
            (0.8562, 3),
            (0.9188, 4),
            (0.9491, 5),
            (0.9607, 6),
            (0.9745, 7),
            (0.9755, 8),
            (0.9776, 9),
            (0.9797, 10),
            (0.9853, 11),
            (0.9871, 12),
            (0.9876, 13),
            (1.0000, 14)
        ],
        "colon": [
            (0.0000, 0),
            (0.4333, 1),
            (0.6656, 2),
            (0.8145, 3),
            (0.8872, 4),
            (0.9258, 5),
            (0.9469, 6),
            (0.9649, 7),
            (0.9822, 8),
            (0.9832, 9),
            (0.9908, 10),
            (0.9950, 11),
            (1.0000, 12)
        ]
    }
}
"""The time to death when a lesion is not cured was based on survival data obtained by the
Netherlands Cancer Registry (IKNL) between 2010 and 2014, stratified by age group, location
and stage distribution.
"""
