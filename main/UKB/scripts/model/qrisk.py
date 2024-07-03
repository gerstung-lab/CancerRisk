
import math
import numpy as np

def blood_cancer_female(age, bmi, c_hb, new_abdopain, new_haematuria, new_necklump,new_nightsweats, new_pmb, new_vte, new_weightloss, s1_bowelchange, s1_bruising):
    dage = age
    dage=dage/10
    age_1 = pow(dage,-2)
    age_2 = pow(dage,-2)*math.log(dage)
    dbmi = bmi
    dbmi=dbmi/10
    bmi_1 = pow(dbmi,-2)
    bmi_2 = pow(dbmi,-2)*math.log(dbmi)

    age_1 = age_1 - 0.039541322737932
    age_2 = age_2 - 0.063867323100567
    bmi_1 = bmi_1 - 0.151021569967270
    bmi_2 = bmi_2 - 0.142740502953529
           
    a=0
    
    a += age_1 * 35.9405666896283120000000000
    a += age_2 * -68.8496375977904480000000000
    a += bmi_1 * 0.0785171223057501980000000
    a += bmi_2 * -5.3730627788681424000000000

    a += c_hb * 1.7035866502297630000000000
    a += new_abdopain * 0.3779206239385797800000000
    a += new_haematuria * 0.4086662974598894700000000
    a += new_necklump * 2.9539029476671903000000000
    a += new_nightsweats * 1.3792892192392403000000000
    a += new_pmb * 0.4689216313440992500000000
    a += new_vte * 0.6036630662990674100000000
    a += new_weightloss * 0.8963398932306315700000000
    a += s1_bowelchange * 0.7291379612468620300000000
    a += s1_bruising * 1.0255003552753392000000000

    score = a + -7.4207849482565749000000000
    return score

def breast_cancer_female(age, alcohol_cat4, bmi, fh_breastcancer, new_breastlump, new_breastpain, new_breastskin, new_pmb, new_vte):

    Ialcohol =  np.asarray([0, 0.0543813075945134560000000,0.1245709972983817800000000, 0.1855198679261514700000000])

    dage = age
    dage=dage/10
    age_1 = pow(dage,-2)
    age_2 = pow(dage,-2)*math.log(dage)
    dbmi = bmi
    dbmi=dbmi/10
    bmi_1 = pow(dbmi,-2)
    bmi_2 = pow(dbmi,-2)*math.log(dbmi)

    age_1 = age_1 - 0.039541322737932
    age_2 = age_2 - 0.063867323100567
    bmi_1 = bmi_1 - 0.151021569967270
    bmi_2 = bmi_2 - 0.142740502953529

    a=0
    a += np.matmul(Ialcohol, alcohol_cat4)
    
    a += age_1 * -14.3029484067898500000000000
    a += age_2 * -25.9301811377364260000000000
    a += bmi_1 * -1.7540983825680900000000000
    a += bmi_2 * 2.0601979121740364000000000


    a += fh_breastcancer * 0.3863899675953914000000000
    a += new_breastlump * 3.9278533274888368000000000
    a += new_breastpain * 0.8779616078329102200000000
    a += new_breastskin * 2.2320296233987880000000000
    a += new_pmb * 0.4465053002248299800000000
    a += new_vte * 0.2728610297213165400000000

    score = a + -6.1261694200869234000000000
    return score

def cervical_cancer_female(age, bmi, c_hb, new_abdopain, new_haematuria, new_imb, new_pmb, new_postcoital, new_vte, smoke_cat):

    Ismoke =  np.asarray([0, 0.3247875277095715300000000,0.7541211259076738800000000, 0.7448343035139659600000000, 0.6328348533913806800000000])

    dage = age
    dage=dage/10
    age_1 = pow(dage,-2)
    age_2 = pow(dage,-2)*math.log(dage)
    dbmi = bmi
    dbmi=dbmi/10
    bmi_1 = pow(dbmi,-2)
    bmi_2 = pow(dbmi,-2)*math.log(dbmi)

    age_1 = age_1 - 0.039541322737932
    age_2 = age_2 - 0.063867323100567
    bmi_1 = bmi_1 - 0.151021569967270
    bmi_2 = bmi_2 - 0.142740502953529

    a=0
    a += np.matmul(Ismoke, smoke_cat)
    a += age_1 * 10.1663393107505800000000000
    a += age_2 * -16.911890249110002000000000
    a += bmi_1 * -0.5675143308052614800000000
    a += bmi_2 * -2.6377586334504044000000000
    a += c_hb * 1.2205973555195053000000000
    a += new_abdopain * 0.7229870191773574200000000
    a += new_haematuria * 1.6126499968790107000000000
    a += new_imb * 1.9527008812518938000000000
    a += new_pmb * 3.3618997560756485000000000
    a += new_postcoital * 3.1391568551730864000000000
    a += new_vte * 1.1276327958138455000000000
    score = a + -8.8309098444401926000000000
    return score

def colorectal_cancer_female(age, alcohol_cat4, bmi, c_hb, fh_gicancer, new_abdodist, new_abdopain, new_appetiteloss, new_rectalbleed, new_vte, new_weightloss, s1_bowelchange, s1_constipation):

    Ialcohol =  np.asarray([0, 0.2429014262884695900000000,0.2359224520197608100000000, 0.4606605934539446100000000])

    dage = age
    dage=dage/10
    age_1 = pow(dage,-2)
    age_2 = pow(dage,-2)*math.log(dage)
    dbmi = bmi
    dbmi=dbmi/10
    bmi_1 = pow(dbmi,-2)
    bmi_2 = pow(dbmi,-2)*math.log(dbmi)

    age_1 = age_1 - 0.039541322737932
    age_2 = age_2 - 0.063867323100567
    bmi_1 = bmi_1 - 0.151021569967270
    bmi_2 = bmi_2 - 0.142740502953529

    a=0;
    a += np.matmul(Ialcohol, alcohol_cat4)
    a += age_1 * -11.6175606616390770000000000
    a += age_2 * -42.9098057686870220000000000
    a += bmi_1 * -0.5344237822753052900000000
    a += bmi_2 * 2.6900552265408226000000000

    a += c_hb * 1.4759238359186861000000000
    a += fh_gicancer * 0.4044501048847998200000000
    a += new_abdodist * 0.6630074287856559900000000
    a += new_abdopain * 1.4990872468711913000000000
    a += new_appetiteloss * 0.5068020107261922400000000
    a += new_rectalbleed * 2.7491673095810105000000000
    a += new_vte * 0.7072816884002932600000000
    a += new_weightloss * 1.0288860866585736000000000
    a += s1_bowelchange * 0.7664414123199643200000000
    a += s1_constipation * 0.3375158123121173600000000

    score = a + -7.5466948789670942000000000
    return score

def gastro_oesophageal_cancer_female(age, bmi, c_hb, new_abdopain, new_appetiteloss, new_dysphagia, new_gibleed, new_heartburn, new_indigestion, new_vte, new_weightloss, smoke_cat):

    Ismoke =  np.asarray([0, 0.2108835385994093400000000,0.4020914846651602000000000, 0.8497119766959212500000000, 1.1020585469724540000000000])

    dage = age
    dage=dage/10
    age_1 = pow(dage,-2)
    age_2 = pow(dage,-2)*math.log(dage)
    dbmi = bmi
    dbmi=dbmi/10
    bmi_1 = pow(dbmi,-2)
    bmi_2 = pow(dbmi,-2)*math.log(dbmi)


    age_1 = age_1 - 0.039541322737932
    age_2 = age_2 - 0.063867323100567
    bmi_1 = bmi_1 - 0.151021569967270
    bmi_2 = bmi_2 - 0.142740502953529

    a=0
    a += np.matmul(Ismoke, smoke_cat)
    a += age_1 * 5.5127932958160830000000000
    a += age_2 * -70.2734062916161830000000000
    a += bmi_1 * 2.6063377632938987000000000
    a += bmi_2 * -1.2389834515079798000000000
    a += c_hb * 1.2479756970482034000000000
    a += new_abdopain * 0.7825304005124729100000000
    a += new_appetiteloss * 0.6514592236889243900000000
    a += new_dysphagia * 3.7751714910656862000000000
    a += new_gibleed * 1.4264472204617833000000000
    a += new_heartburn * 0.8178746069193373300000000
    a += new_indigestion * 1.4998439683677578000000000
    a += new_vte * 0.7199894658172598700000000
    a += new_weightloss * 1.2287925630053846000000000
    score = a + -8.8746031610250764000000000
    return score

def lung_cancer_female(age, b_copd, bmi, c_hb, new_appetiteloss, new_dysphagia, new_haemoptysis, new_indigestion, new_necklump, new_vte, new_weightloss, s1_cough, smoke_cat):
    # The conditional arrays
    Ismoke = np.asarray([0, 1.3397416191950409, 1.9500839456663224, 2.1881694694325233, 2.4828660433307768])

    # Applying the fractional polynomial transforms (which includes scaling)
    dage = age / 10
    age_1 = pow(dage, -2)
    age_2 = pow(dage, -2) * math.log(dage)
    dbmi = bmi / 10
    bmi_1 = pow(dbmi, -2)
    bmi_2 = pow(dbmi, -2) * math.log(dbmi)

    # Centring the continuous variables
    age_1 = age_1 - 0.039541322737932
    age_2 = age_2 - 0.063867323100567
    bmi_1 = bmi_1 - 0.151021569967270
    bmi_2 = bmi_2 - 0.142740502953529

    # Start of Sum
    a = 0

    # The conditional sums
    a += np.matmul(Ismoke, smoke_cat)

    # Sum from continuous values
    a += age_1 * -117.24057375029625
    a += age_2 * 25.17022547412681
    a += bmi_1 * 2.584548813392435
    a += bmi_2 * -0.60835239667627994

    # Sum from boolean values
    a += b_copd * 0.79429019626713648
    a += c_hb * 0.86279803244016284
    a += new_appetiteloss * 0.71702321213794462
    a += new_dysphagia * 0.67184268060773233
    a += new_haemoptysis * 2.9286439157734474
    a += new_indigestion * 0.36348937301142736
    a += new_necklump * 1.209724038009159
    a += new_vte * 0.8907072670032341
    a += new_weightloss * 1.1384524885073082
    a += s1_cough * 0.64399170532756023

    # Sum from interaction terms

    # Calculate the score itself
    score = a + -8.6449002971789692
    return score

def ovarian_cancer_female(age, bmi, c_hb, fh_ovariancancer, new_abdodist, new_abdopain, new_appetiteloss, new_haematuria, new_indigestion, new_pmb, new_vte, new_weightloss, s1_bowelchange):
    # The conditional arrays

    # Applying the fractional polynomial transforms (which includes scaling)
    dage = age / 10
    age_1 = pow(dage, -2)
    age_2 = pow(dage, -2) * math.log(dage)
    dbmi = bmi / 10
    bmi_1 = pow(dbmi, -2)
    bmi_2 = pow(dbmi, -2) * math.log(dbmi)

    # Centring the continuous variables
    age_1 = age_1 - 0.039541322737932
    age_2 = age_2 - 0.063867323100567
    bmi_1 = bmi_1 - 0.151021569967270
    bmi_2 = bmi_2 - 0.142740502953529

    # Start of Sum
    a = 0

    # The conditional sums

    # Sum from continuous values
    a += age_1 * -61.083181446256894
    a += age_2 * 20.302861270110689
    a += bmi_1 * -2.1261135335028407
    a += bmi_2 * 3.2168200408772472

    # Sum from boolean values
    a += c_hb * 1.3625636791018674
    a += fh_ovariancancer * 1.995177480995183
    a += new_abdodist * 2.9381020883363806
    a += new_abdopain * 1.7307824546132513
    a += new_appetiteloss * 1.0606947909647773
    a += new_haematuria * 0.49588359974681079
    a += new_indigestion * 0.38437310274939984
    a += new_pmb * 1.5869592940878865
    a += new_vte * 1.6839747529852673
    a += new_weightloss * 0.47743323938217208
    a += s1_bowelchange * 0.68498500071823143

    # Sum from interaction terms

    # Calculate the score itself
    score = a + -7.5609929644491318
    return score

def pancreatic_cancer_female(age, b_chronicpan, b_type2, bmi, new_abdopain, new_appetiteloss, new_dysphagia, new_gibleed,
                                 new_indigestion, new_vte, new_weightloss, s1_bowelchange, smoke_cat):
    # The conditional arrays
    Ismoke = np.asarray([0, -0.063130184815204424, 0.35236959505289345, 0.71460036703271568, 0.80732074103354412])

    # Applying the fractional polynomial transforms (which includes scaling)
    dage = age / 10
    age_1 = pow(dage, -2)
    age_2 = pow(dage, -2) * math.log(dage)
    dbmi = bmi / 10
    bmi_1 = pow(dbmi, -2)
    bmi_2 = pow(dbmi, -2) * math.log(dbmi)

    # Centring the continuous variables
    age_1 = age_1 - 0.039541322737932
    age_2 = age_2 - 0.063867323100567
    bmi_1 = bmi_1 - 0.151021569967270
    bmi_2 = bmi_2 - 0.142740502953529

    # Start of Sum
    a = 0

    # The conditional sums
    a += np.matmul(Ismoke, smoke_cat)

    # Sum from continuous values
    a += age_1 * -6.8219654517231225
    a += age_2 * -65.640489730518865
    a += bmi_1 * 3.9715559458995728
    a += bmi_2 * -3.11611079991305

    # Sum from boolean values
    a += b_chronicpan * 1.1948138830441282
    a += b_type2 * 0.7951745325664703
    a += new_abdopain * 1.9230379689782926
    a += new_appetiteloss * 1.5209568259888571
    a += new_dysphagia * 1.0107551560302726
    a += new_gibleed * 0.93240591532542594
    a += new_indigestion * 1.1134012616631439
    a += new_vte * 1.4485586969016084
    a += new_weightloss * 1.5791912580663912
    a += s1_bowelchange * 0.93617386119414447

    # Sum from interaction terms

    # Calculate the score itself
    score = a + -9.2782129678657608
    return score

def renal_tract_cancer_female(age, bmi, c_hb, new_abdopain, new_appetiteloss, new_haematuria, new_indigestion, new_pmb, new_weightloss, smoke_cat):
    # The conditional arrays
    Ismoke = np.asarray([0, 0.27521757277393727, 0.54986566314758611, 0.65362421821366801, 0.90537636617858797])

    # Applying the fractional polynomial transforms (which includes scaling)
    dage = age / 10
    age_1 = pow(dage, -2)
    age_2 = pow(dage, -2) * math.log(dage)
    dbmi = bmi / 10
    bmi_1 = pow(dbmi, -2)
    bmi_2 = pow(dbmi, -2) * math.log(dbmi)

    # Centring the continuous variables
    age_1 = age_1 - 0.039541322737932
    age_2 = age_2 - 0.063867323100567
    bmi_1 = bmi_1 - 0.151021569967270
    bmi_2 = bmi_2 - 0.142740502953529

    # Start of Sum
    a = 0

    # The conditional sums
    a += np.matmul(Ismoke, smoke_cat)

    # Sum from continuous values
    a += age_1 * -0.032322656962661747
    a += age_2 * -56.355141078663578
    a += bmi_1 * 1.210391053577933
    a += bmi_2 * -4.7221299079939785

    # Sum from boolean values
    a += c_hb * 1.2666531852544143
    a += new_abdopain * 0.61559549847075945
    a += new_appetiteloss * 0.68421845946760196
    a += new_haematuria * 4.1791444537241542
    a += new_indigestion * 0.56943292248218746
    a += new_pmb * 1.2541097882792864
    a += new_weightloss * 0.77116105602905183

    # Sum from interaction terms

    # Calculate the score itself
    score = a + -8.9440775553776248
    return score


def uterine_cancer_female(age, b_endometrial, b_type2, bmi, new_abdopain, new_haematuria, new_imb, new_pmb, new_vte):
    # The conditional arrays

    # Applying the fractional polynomial transforms (which includes scaling)
    dage = age / 10
    age_1 = pow(dage, -2)
    age_2 = pow(dage, -2) * math.log(dage)
    dbmi = bmi / 10
    bmi_1 = pow(dbmi, -2)
    bmi_2 = pow(dbmi, -2) * math.log(dbmi)

    # Centring the continuous variables
    age_1 = age_1 - 0.039541322737932
    age_2 = age_2 - 0.063867323100567
    bmi_1 = bmi_1 - 0.151021569967270
    bmi_2 = bmi_2 - 0.142740502953529

    # Start of Sum
    a = 0

    # The conditional sums

    # Sum from continuous values
    a += age_1 * 2.7778124257317254
    a += age_2 * -59.533351456663333
    a += bmi_1 * 3.7623897936404322
    a += bmi_2 * -26.804545007465432

    # Sum from boolean values
    a += b_endometrial * 0.8742311851235286
    a += b_type2 * 0.26551810240635559
    a += new_abdopain * 0.68919538367355804
    a += new_haematuria * 1.6798617740998527
    a += new_imb * 1.7853122923827887
    a += new_pmb * 4.4770199876067398
    a += new_vte * 1.0362058616761669

    # Sum from interaction terms

    # Calculate the score itself
    score = a + -8.9931390822564037
    return score

def blood_cancer_male(age, bmi, c_hb, new_abdodist, new_abdopain, new_appetiteloss, new_dysphagia, new_haematuria,
                          new_haemoptysis, new_indigestion, new_necklump, new_nightsweats, new_testicularlump, new_vte,
                          new_weightloss):
    # The conditional arrays

    # Applying the fractional polynomial transforms (which includes scaling)
    dage = age / 10
    age_1 = dage
    age_2 = dage * math.log(dage)
    dbmi = bmi / 10
    bmi_1 = pow(dbmi, -2)
    bmi_2 = dbmi

    # Centring the continuous variables
    age_1 = age_1 - 4.800777912139893
    age_2 = age_2 - 7.531354427337647
    bmi_1 = bmi_1 - 0.146067067980766
    bmi_2 = bmi_2 - 2.616518735885620

    # Start of Sum
    a = 0

    # The conditional sums

    # Sum from continuous values
    a += age_1 * 3.497017935455661
    a += age_2 * -1.0806801421562633
    a += bmi_1 * 0.95192594795117924
    a += bmi_2 * 0.17146693584100858

    # Sum from boolean values
    a += c_hb * 1.8905802113004144
    a += new_abdodist * 0.84304321972113938
    a += new_abdopain * 0.62264732882949925
    a += new_appetiteloss * 1.067215038075376
    a += new_dysphagia * 0.5419443056595199
    a += new_haematuria * 0.46075380853635217
    a += new_haemoptysis * 0.95014468992418366
    a += new_indigestion * 0.56356865693313374
    a += new_necklump * 3.1567783466839603
    a += new_nightsweats * 1.5201300180753576
    a += new_testicularlump * 0.99575249282451073
    a += new_vte * 0.61425897261328666
    a += new_weightloss * 1.2233663263194712

    # Sum from interaction terms

    # Calculate the score itself
    score = a + -7.2591289466850277
    return score

def colorectal_cancer_male(age, alcohol_cat4, bmi, c_hb, fh_gicancer, new_abdodist, new_abdopain, new_appetiteloss,
                                new_rectalbleed, new_weightloss, s1_bowelchange, s1_constipation):
    # The conditional arrays
    Ialcohol = np.asarray([0, 0.067443170026859178, 0.2894952197787854, 0.44195399849740974])

    # Applying the fractional polynomial transforms (which includes scaling)
    dage = age / 10
    age_1 = dage
    age_2 = dage * math.log(dage)
    dbmi = bmi / 10
    bmi_1 = pow(dbmi, -2)
    bmi_2 = dbmi

    # Centring the continuous variables
    age_1 = age_1 - 4.800777912139893
    age_2 = age_2 - 7.531354427337647
    bmi_1 = bmi_1 - 0.146067067980766
    bmi_2 = bmi_2 - 2.616518735885620

    # Start of Sum
    a = 0

    # The conditional sums

    a += np.matmul(Ialcohol, alcohol_cat4)

    # Sum from continuous values
    a += age_1 * 7.2652842514036369
    a += age_2 * -2.3119103657424414
    a += bmi_1 * 0.4591530847132721
    a += bmi_2 * 0.14026516690905994

    # Sum from boolean values
    a += c_hb * 1.4066322376473517
    a += fh_gicancer * 0.40572853210100446
    a += new_abdodist * 1.3572627165452165
    a += new_abdopain * 1.5179997924486877
    a += new_appetiteloss * 0.54213354577521133
    a += new_rectalbleed * 2.8846500840638964
    a += new_weightloss * 1.1082218896963933
    a += s1_bowelchange * 1.2962496832506105
    a += s1_constipation * 0.22842561154989671

    # Sum from interaction terms

    # Calculate the score itself
    score = a + -7.6876342765226262
    return score

def gastro_oesophageal_cancer_male(age, bmi, c_hb, new_abdopain, new_appetiteloss, new_dysphagia, new_gibleed,
                                       new_heartburn, new_indigestion, new_necklump, new_weightloss, smoke_cat):
    # The conditional arrays
    Ismoke = np.asarray([0, 0.35326859222399482, 0.63432015577122913, 0.65008197369041587, 0.62734130105599528])

    # Applying the fractional polynomial transforms (which includes scaling)
    dage = age / 10
    age_1 = dage
    age_2 = dage * math.log(dage)
    dbmi = bmi / 10
    bmi_1 = pow(dbmi, -2)
    bmi_2 = dbmi

    # Centring the continuous variables
    age_1 = age_1 - 4.800777912139893
    age_2 = age_2 - 7.531354427337647
    bmi_1 = bmi_1 - 0.146067067980766
    bmi_2 = bmi_2 - 2.616518735885620

    # Start of Sum
    a = 0

    # The conditional sums

    a += np.matmul(Ismoke, smoke_cat)

    # Sum from continuous values
    a += age_1 * 8.5841509312915623
    a += age_2 * -2.765040945011636
    a += bmi_1 * 4.1816752831070323
    a += bmi_2 * 0.624710628895496

    # Sum from boolean values
    a += c_hb * 1.1065543049459461
    a += new_abdopain * 1.0280133043080188
    a += new_appetiteloss * 1.1868017500634926
    a += new_dysphagia * 3.8253199428642568
    a += new_gibleed * 1.8454733322333583
    a += new_heartburn * 1.1727679169313121
    a += new_indigestion * 1.8843639195644077
    a += new_necklump * 0.84146963853933576
    a += new_weightloss * 1.4698638306735652

    # Sum from interaction terms

    # Calculate the score itself
    score = a + -8.4208700270300625
    return score

def lung_cancer_male(age, b_copd, bmi, c_hb, new_abdopain, new_appetiteloss, new_dysphagia, new_haemoptysis,
                         new_indigestion, new_necklump, new_nightsweats, new_vte, new_weightloss, s1_cough, smoke_cat):
    # The conditional arrays
    Ismoke = np.asarray([0, 0.84085747375244646, 1.4966499028172435, 1.7072509513243501, 1.8882615411851338])

    # Applying the fractional polynomial transforms (which includes scaling)
    dage = age / 10
    age_1 = dage
    age_2 = dage * math.log(dage)
    dbmi = bmi / 10
    bmi_1 = pow(dbmi, -2)
    bmi_2 = dbmi

    # Centring the continuous variables
    age_1 = age_1 - 4.800777912139893
    age_2 = age_2 - 7.531354427337647
    bmi_1 = bmi_1 - 0.146067067980766
    bmi_2 = bmi_2 - 2.616518735885620

    # Start of Sum
    a = 0

    # The conditional sums
    a += np.matmul(Ismoke, smoke_cat)

    # Sum from continuous values
    a += age_1 * 11.917808960225496
    a += age_2 * -3.8503786390624457
    a += bmi_1 * 1.860558422294992
    a += bmi_2 * -0.11327500388008699

    # Sum from boolean values
    a += b_copd * 0.55261276296940742
    a += c_hb * 0.82437891170693112
    a += new_abdopain * 0.39964248791030577
    a += new_appetiteloss * 0.7487413720163385
    a += new_dysphagia * 1.0410482089004374
    a += new_haemoptysis * 2.8241680746676243
    a += new_indigestion * 0.2689673675929089
    a += new_necklump * 1.1065323833644807
    a += new_nightsweats * 0.78906965838459642
    a += new_vte * 0.79911502960387548
    a += new_weightloss * 1.3738119234931856
    a += s1_cough * 0.51541790034374857

    # Sum from interaction terms

    # Calculate the score itself
    score = a + -8.7166918098019277
    return score


def pancreatic_cancer_male(age, b_chronicpan, b_type2, bmi, new_abdopain, new_appetiteloss, new_dysphagia, new_gibleed,
                               new_indigestion, new_vte, new_weightloss, s1_constipation, smoke_cat):
    # The conditional arrays
    Ismoke = np.asarray([0, 0.27832981720899735, 0.30794189289176033, 0.56473593949911283, 0.77651254271268666])

    # Applying the fractional polynomial transforms (which includes scaling)
    dage = age / 10
    age_1 = dage
    age_2 = dage * math.log(dage)
    dbmi = bmi / 10
    bmi_1 = pow(dbmi, -2)
    bmi_2 = dbmi

    # Centring the continuous variables
    age_1 = age_1 - 4.800777912139893
    age_2 = age_2 - 7.531354427337647
    bmi_1 = bmi_1 - 0.146067067980766
    bmi_2 = bmi_2 - 2.616518735885620

    # Start of Sum
    a = 0

    # The conditional sums
    a += np.matmul(Ismoke, smoke_cat)

    # Sum from continuous values
    a += age_1 * 8.0275778709105907
    a += age_2 * -2.6082429130982798
    a += bmi_1 * 1.781957499473682
    a += bmi_2 * -0.024960006489569975

    # Sum from boolean values
    a += b_chronicpan * 0.99132463479918231
    a += b_type2 * 0.73969050982025408
    a += new_abdopain * 2.1506984011721579
    a += new_appetiteloss * 1.4272326009960661
    a += new_dysphagia * 0.91686892075260662
    a += new_gibleed * 0.98810610330811499
    a += new_indigestion * 1.2837402377092237
    a += new_vte * 1.1741805346104719
    a += new_weightloss * 2.0466064239967046
    a += s1_constipation * 0.62405480330482144

    # Sum from interaction terms

    # Calculate the score itself
    score = a + -9.2275729512009956
    return score

def prostate_cancer_male(age, bmi, fh_prostatecancer, new_abdopain, new_appetiteloss, new_haematuria, new_rectalbleed,
                              new_testespain, new_testicularlump, new_vte, new_weightloss, s1_impotence, s1_nocturia,
                              s1_urinaryfreq, s1_urinaryretention):
    # The conditional arrays

    # Applying the fractional polynomial transforms (which includes scaling)
    dage = age / 10
    age_1 = dage
    age_2 = dage * math.log(dage)
    dbmi = bmi / 10
    bmi_1 = pow(dbmi, -2)
    bmi_2 = dbmi

    # Centring the continuous variables
    age_1 = age_1 - 4.800777912139893
    age_2 = age_2 - 7.531354427337647
    bmi_1 = bmi_1 - 0.146067067980766
    bmi_2 = bmi_2 - 2.616518735885620

    # Start of Sum
    a = 0

    # The conditional sums

    # Sum from continuous values
    a += age_1 * 14.839101042656692
    a += age_2 * -4.8051341054408843
    a += bmi_1 * -2.8369035324107057
    a += bmi_2 * -0.36349842659000514

    # Sum from boolean values
    a += fh_prostatecancer * 1.2892957682128878
    a += new_abdopain * 0.44455883728607742
    a += new_appetiteloss * 0.34255819715349151
    a += new_haematuria * 1.4890866073593347
    a += new_rectalbleed * 0.34786129520339637
    a += new_testespain * 0.63876093500764075
    a += new_testicularlump * 0.6338177436853567
    a += new_vte * 0.57581908041962615
    a += new_weightloss * 0.75287362266658731
    a += s1_impotence * 0.36921800415342415
    a += s1_nocturia * 1.0381560026453696
    a += s1_urinaryfreq * 0.70364102530803652
    a += s1_urinaryretention * 0.85257033994355869

    # Sum from interaction terms

    # Calculate the score itself
    score = a + -7.8871012697298699
    return score

def renal_tract_cancer_male(age, bmi, new_abdopain, new_haematuria, new_nightsweats, new_weightloss, smoke_cat):
    # The conditional arrays
    Ismoke = np.asarray([0, 0.4183007995792849, 0.63351623682787428, 0.78472308793222056, 0.96310914112952117])

    # Applying the fractional polynomial transforms (which includes scaling)
    dage = age / 10
    age_1 = dage
    age_2 = dage * math.log(dage)
    dbmi = bmi / 10
    bmi_1 = pow(dbmi, -2)
    bmi_2 = dbmi

    # Centring the continuous variables
    age_1 = age_1 - 4.800777912139893
    age_2 = age_2 - 7.531354427337647
    bmi_1 = bmi_1 - 0.146067067980766
    bmi_2 = bmi_2 - 2.616518735885620

    # Start of Sum
    a = 0

    # The conditional sums
    a += np.matmul(Ismoke, smoke_cat)

    # Sum from continuous values
    a += age_1 * 6.2113803461111061
    a += age_2 * -1.983566150695387
    a += bmi_1 * -1.5995682550089132
    a += bmi_2 * -0.077769683693075312

    # Sum from boolean values
    a += new_abdopain * 0.60894656789095847
    a += new_haematuria * 4.1596453389556789
    a += new_nightsweats * 1.0520790556587876
    a += new_weightloss * 0.6824635274408537

    # Sum from interaction terms

    # Calculate the score itself
    score = a + -8.300655539894251
    return score

def testicular_cancer_male_raw(age, bmi, new_testespain, new_testicularlump, new_vte):
    # The conditional arrays

    # Applying the fractional polynomial transforms (which includes scaling)
    dage = age / 10
    age_1 = dage
    age_2 = dage * math.log(dage)
    dbmi = bmi / 10
    bmi_1 = pow(dbmi, -2)
    bmi_2 = dbmi

    # Centring the continuous variables
    age_1 = age_1 - 4.800777912139893
    age_2 = age_2 - 7.531354427337647
    bmi_1 = bmi_1 - 0.146067067980766
    bmi_2 = bmi_2 - 2.616518735885620

    # Start of Sum
    a = 0

    # The conditional sums

    # Sum from continuous values
    a += age_1 * 3.9854184482476338
    a += age_2 * -1.7426970576325218
    a += bmi_1 * 2.0160796798276812
    a += bmi_2 * -0.042734043745477374

    # Sum from boolean values
    a += new_testespain * 2.7411880902787775
    a += new_testicularlump * 5.2200886149323269
    a += new_vte * 2.2416746922896493

    # Sum from interaction terms

    # Calculate the score itself
    score = a + -8.7592209887895898
    return score


def qrisk(sex, age, bmi, alcohol_cat4, smoke_cat, b_chronicpan, b_copd, b_endometrial, b_type2, c_hb, fh_breastcancer, fh_gicancer, fh_ovariancancer, fh_prostatecancer, new_abdodist, new_abdopain, new_appetiteloss, new_breastlump, new_breastpain, new_breastskin, new_dysphagia, new_gibleed,
new_haematuria, new_haemoptysis, new_heartburn, new_imb, new_indigestion, new_necklump, new_nightsweats, new_pmb, new_postcoital, new_rectalbleed, new_vte, new_weightloss, new_testespain, new_testicularlump, s1_bowelchange, s1_bruising, s1_constipation, s1_cough, 
s1_impotence, s1_nocturia, s1_urinaryfreq, s1_urinaryretention):

    pred = np.zeros((1, 22))

    if sex==0:
        pred[0, 0] = gastro_oesophageal_cancer_female(age, bmi, c_hb, new_abdopain, new_appetiteloss,new_dysphagia, new_gibleed, new_heartburn, new_indigestion, new_vte, new_weightloss, smoke_cat)
        pred[0, 1] = gastro_oesophageal_cancer_female(age, bmi, c_hb, new_abdopain, new_appetiteloss,new_dysphagia, new_gibleed, new_heartburn, new_indigestion, new_vte, new_weightloss, smoke_cat)
        pred[0, 2] = colorectal_cancer_female(age, alcohol_cat4, bmi, c_hb, fh_gicancer, new_abdodist, new_abdopain, new_appetiteloss, new_rectalbleed, new_vte, new_weightloss, s1_bowelchange, s1_constipation)
        pred[0, 4] = pancreatic_cancer_female(age, b_chronicpan, b_type2, bmi, new_abdopain, new_appetiteloss, new_dysphagia, new_gibleed,new_indigestion, new_vte, new_weightloss, s1_bowelchange, smoke_cat)
        pred[0, 5] = lung_cancer_female(age, b_copd, bmi, c_hb, new_appetiteloss, new_dysphagia, new_haemoptysis, new_indigestion, new_necklump, new_vte, new_weightloss, s1_cough, smoke_cat)
        pred[0, 7] = breast_cancer_female(age, alcohol_cat4, bmi, fh_breastcancer, new_breastlump, new_breastpain, new_breastskin, new_pmb, new_vte)
        pred[0, 8] = cervical_cancer_female(age, bmi, c_hb, new_abdopain, new_haematuria, new_imb, new_pmb, new_postcoital, new_vte, smoke_cat)
        pred[0, 9] = uterine_cancer_female(age, b_endometrial, b_type2, bmi, new_abdopain, new_haematuria, new_imb, new_pmb, new_vte)
        pred[0, 10] = ovarian_cancer_female(age, bmi, c_hb, fh_ovariancancer, new_abdodist, new_abdopain, new_appetiteloss, new_haematuria, new_indigestion, new_pmb, new_vte, new_weightloss, s1_bowelchange)
        pred[0, 13] = renal_tract_cancer_female(age, bmi, c_hb, new_abdopain, new_appetiteloss, new_haematuria, new_indigestion, new_pmb, new_weightloss, smoke_cat)
        pred[0, 19] = blood_cancer_female(age, bmi, c_hb, new_abdopain, new_haematuria, new_necklump,new_nightsweats, new_pmb, new_vte, new_weightloss, s1_bowelchange, s1_bruising)

    else:

        pred[0, 0] = gastro_oesophageal_cancer_male(age, bmi, c_hb, new_abdopain, new_appetiteloss, new_dysphagia, new_gibleed, new_heartburn, new_indigestion, new_necklump, new_weightloss, smoke_cat)
        pred[0, 1] = gastro_oesophageal_cancer_male(age, bmi, c_hb, new_abdopain, new_appetiteloss, new_dysphagia, new_gibleed, new_heartburn, new_indigestion, new_necklump, new_weightloss, smoke_cat)
        pred[0, 2] = colorectal_cancer_male(age, alcohol_cat4, bmi, c_hb, fh_gicancer, new_abdodist, new_abdopain, new_appetiteloss,new_rectalbleed, new_weightloss, s1_bowelchange, s1_constipation)
        pred[0, 4] = pancreatic_cancer_male(age, b_chronicpan, b_type2, bmi, new_abdopain, new_appetiteloss, new_dysphagia, new_gibleed, new_indigestion, new_vte, new_weightloss, s1_constipation, smoke_cat)

        pred[0, 5] = lung_cancer_male(age, b_copd, bmi, c_hb, new_abdopain, new_appetiteloss, new_dysphagia, new_haemoptysis, new_indigestion, new_necklump, new_nightsweats, new_vte, new_weightloss, s1_cough, smoke_cat)

        pred[0, 11] = prostate_cancer_male(age, bmi, fh_prostatecancer, new_abdopain, new_appetiteloss, new_haematuria, new_rectalbleed,new_testespain, new_testicularlump, new_vte, new_weightloss, s1_impotence, s1_nocturia, s1_urinaryfreq, s1_urinaryretention)
        pred[0, 12] = testicular_cancer_male_raw(age, bmi, new_testespain, new_testicularlump, new_vte)
        pred[0, 13] = renal_tract_cancer_male(age, bmi, new_abdopain, new_haematuria, new_nightsweats, new_weightloss, smoke_cat)
        pred[0, 19] = blood_cancer_male(age, bmi, c_hb, new_abdodist, new_abdopain, new_appetiteloss, new_dysphagia, new_haematuria,
                                  new_haemoptysis, new_indigestion, new_necklump, new_nightsweats, new_testicularlump, new_vte,
                                  new_weightloss)

    return pred


    
    
    