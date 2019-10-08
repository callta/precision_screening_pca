import pandas as pd
import numpy as np
import os

# Generate the risk distribution parameters from the risk_distribution.py script
from risk_distribution import *

# Import parameters from parameters.py script
from parameters import *

# Set path for saving dataframes
base_path = '...'
sims = 10000

# Functions to return probabilistic variables in suitable format
def gamma(alpha, beta):
    alpha = np.array([alpha] * sims)
    beta = np.array([beta] * sims)
    samples = np.random.gamma(alpha, beta)
    return samples

def gamma_specified(min, multiplier, alpha, beta):
    min = np.array([min] * sims).T
    alpha = np.array([alpha] * sims)
    beta = np.array([beta] * sims)
    samples = min + np.random.gamma(alpha, beta) * multiplier
    samples = samples.T
    return samples

def normal(parameter, sd):
    samples = np.random.normal(parameter, sd, sims)
    samples = np.array([samples] * 45).T
    return samples

def lognormal(parameter, sd):
    samples = np.random.lognormal(parameter, sd, sims)
    samples = np.array([samples] * 45).T
    return samples

def beta(parameter, se):
    alpha = np.array([parameter * ((parameter*(1-parameter))/(se**2)-1)] * sims)
    beta = (alpha/parameter) - alpha
    samples = np.random.beta(alpha, beta)
    samples = samples.T
    return samples

# Function to deliver PSA simulation matrix for variables not being varied
def psa_function(var):
    return np.array([var] * sims)

# Function to generate outcomes
def outcomes(parameter):
    # Simulations - one total value per simulation
    sims = np.sum(parameter, axis=1)
    # Mean value across all simulations
    mean = np.mean(parameter, axis=0)
    # Total value (mean and sum across all simulations)
    total = np.sum(mean)
    return sims, mean, total

##############
# Parameters #
##############

# Costs
cost_psa = gamma(33.9,0.3)
cost_psa = np.tile(cost_psa, (45,1)).T   # Extend cost_psa to be a matrix of length 45 x sims
cost_prs = gamma(33.9,0.7)
cost_biopsy = gamma(33.9,11.5)
cost_biopsy = np.tile(cost_biopsy, (45,1)).T
cost_refuse_biopsy = gamma(33.9,3.1)
cost_refuse_biopsy = np.tile(cost_refuse_biopsy, (45,1)).T

cost_assessment = gamma(33.9,22.7)
cost_as = gamma(33.9,128.1)
cost_rp = gamma(33.9,241.2)
cost_rt = gamma(33.9,158.9)
cost_brachytherapy = gamma(33.9,45.1)
cost_adt = gamma(33.9,16.5)
cost_chemo = gamma(33.9,219.2)

cost_rt_chemo = cost_rt + cost_chemo
cost_rp_rt = cost_rp + cost_rt
cost_rp_chemo = cost_rp + cost_chemo
cost_rp_rt_chemo = cost_rp + cost_rt + cost_chemo

costs_local = np.stack((cost_chemo, cost_rp,
                        cost_rt, cost_rt_chemo,
                        cost_rp_chemo, cost_rp_rt,
                        cost_rp_rt_chemo, cost_as,
                        cost_adt, cost_brachytherapy), axis=-1)

costs_adv = np.array(costs_local, copy=True)

# Incident costs / treatment dataframe
tx_costs_local = costs_local * tx_local
tx_costs_adv = costs_adv * tx_adv

pca_death_costs = gamma(1.8,3854.9)

# Utilities
pca_incidence_utility_psa = gamma_specified((pca_incidence_utility-0.05), 0.2, 5, 0.05)
utility_background_psa = gamma_specified((utility_background-0.03), 0.167, 4, 0.06)

# Relative risk of death in screened cohort
rr_death_screening = lognormal(-0.2357, 0.0724)

# Proportion of cancers at risk of overdiagnosis
p_overdiagnosis_psa = beta(p_overdiagnosis, 0.001)
additional_years = psa_function(np.repeat(0,20))
p_overdiagnosis_psa = np.concatenate((p_overdiagnosis_psa, additional_years.T))
p_overdiagnosis_psa[0:10,:] = 0

# Relative risk incidence of advanced cancer (stages III and IV)
rr_adv_screening = lognormal(-0.1625, 0.0829)
rr_adv_screening[:,0:10] = 0
rr_adv_screening[:,25:] = 0

# The relative increase in cancers detected if screened
p_increase_df = pd.read_csv('data/p_increase_df.csv', index_col='age')

[RR_INCIDENCE_SC_55, RR_INCIDENCE_SC_56,
 RR_INCIDENCE_SC_57, RR_INCIDENCE_SC_58,
 RR_INCIDENCE_SC_59, RR_INCIDENCE_SC_60,
 RR_INCIDENCE_SC_61, RR_INCIDENCE_SC_62,
 RR_INCIDENCE_SC_63, RR_INCIDENCE_SC_64,
 RR_INCIDENCE_SC_65, RR_INCIDENCE_SC_66,
 RR_INCIDENCE_SC_67, RR_INCIDENCE_SC_68,
 RR_INCIDENCE_SC_69] = [np.random.lognormal(p_increase_df.loc[i, '1.23_log'],
                                            p_increase_df.loc[i, 'se'],
                                            sims)
                        for i in np.arange(55,70,1)]

rr_incidence = np.vstack((np.array([np.repeat(1,sims)]*10),
                          RR_INCIDENCE_SC_55, RR_INCIDENCE_SC_56, RR_INCIDENCE_SC_57,
                          RR_INCIDENCE_SC_58, RR_INCIDENCE_SC_59, RR_INCIDENCE_SC_60,
                          RR_INCIDENCE_SC_61, RR_INCIDENCE_SC_62, RR_INCIDENCE_SC_63,
                          RR_INCIDENCE_SC_64, RR_INCIDENCE_SC_65, RR_INCIDENCE_SC_66,
                          RR_INCIDENCE_SC_67, RR_INCIDENCE_SC_68, RR_INCIDENCE_SC_69))

rr_incidence[rr_incidence < 1] = 1.03 # truncate

# Drop in incidence in the year after screening stops
post_sc_incidence_drop = 0.9

# Number of biopsies per cancer detected
# Proportion having biopsy (screened arms)
p_suspected = normal(0.24,0.05)
p_suspected_refuse_biopsy = normal(0.24,0.05)

# Proportion having biopsy (non-screened arms)
# (201/567) - Ahmed et al. 2017, Table S6 (doi: 10.1016/S0140-6736(16)32401-1)
p_suspected_ns = normal((201/567),0.05)
p_suspected_refuse_biopsy_ns = normal((201/567),0.05)

n_psa_tests = normal(1.2,0.05)

# Relative cost increase if clinically detected
# Source: Pharoah et al. 2013
relative_cost_clinically_detected = normal(1.1,0.04)

# Create a function to append the results to the relevant lists
def gen_list_outcomes(parameter_list, parameter):
    parameter_list.append(parameter)
    return parameter_list

# Run through each AR threshold in turn:
reference_absolute_risk = np.round(np.arange(0.02,0.105,0.005),3)
for reference_value in reference_absolute_risk:

    a_risk = pd.read_csv(base_path+(str(np.round(reference_value*100,2)))+'/a_risk_'+(str(np.round(reference_value*100,2)))+'.csv').set_index('age')

    # Generate lists to store the variables
    (s_qalys_discount_ns_list, s_cost_discount_ns_list, s_pca_deaths_ns_list,
     ns_cohort_list, outcomes_ns_psa_list,

     s_qalys_discount_age_list, s_cost_discount_age_list,
     s_pca_deaths_age_list, s_overdiagnosis_age_list,
     age_cohort_list, outcomes_age_psa_list,

     s_qalys_discount_prs_list, s_cost_discount_prs_list,
     s_pca_deaths_prs_list, s_overdiagnosis_prs_list,
     prs_cohort_list, outcomes_prs_psa_list) = [[] for _ in range(17)]

    parameter_list_ns = [s_qalys_discount_ns_list, s_cost_discount_ns_list, s_pca_deaths_ns_list,
                         ns_cohort_list, outcomes_ns_psa_list]

    parameter_list_age = [s_qalys_discount_age_list, s_cost_discount_age_list,
                          s_pca_deaths_age_list, s_overdiagnosis_age_list,
                          age_cohort_list, outcomes_age_psa_list]

    parameter_list_prs = [s_qalys_discount_prs_list, s_cost_discount_prs_list,
                          s_pca_deaths_prs_list, s_overdiagnosis_prs_list,
                          prs_cohort_list, outcomes_prs_psa_list]

    # Loop through years 45-69 to build cohorts
    for year in (a_risk.index[0:25]):

                                    ################################################
                                    #              Non-screening Cohort            #
                                    ################################################


        #################################
        # Transition rates - no screening
        #################################

        tr_incidence = psa_function(pca_incidence[year-45:])
        tr_pca_death_baseline = psa_function(pca_death_baseline[year-45:])
        tr_death_other_causes = psa_function(death_other_causes[year-45:])
        psa_stage_local = psa_function(stage_local[year-45:])
        psa_stage_adv = psa_function(stage_adv[year-45:])

        # Year 1 in the model
        #####################

        age = np.arange(year,90)
        length_df = len(age)

        # Cohorts, numbers 'healthy', and incident cases
        cohort = np.array([np.repeat(pop[year], length_df)] * sims)
        pca_alive = np.array([np.zeros(length_df)] * sims)
        healthy = cohort - pca_alive
        pca_incidence_ns_cohort = healthy * tr_incidence

        # Deaths
        pca_death = ((pca_alive * tr_pca_death_baseline)
                     + (healthy * tr_pca_death_baseline))

        pca_death_other = ((pca_incidence_ns_cohort
                            + pca_alive
                            - pca_death)
                           * tr_death_other_causes)

        healthy_death_other = ((healthy - pca_incidence_ns_cohort)
                               * tr_death_other_causes)

        total_death = (pca_death
                       + pca_death_other
                       + healthy_death_other)

        # Prevalent cases & life-years
        pca_prevalence_ns = (pca_incidence_ns_cohort
                             - pca_death
                             - pca_death_other)

        lyrs_pca_nodiscount = pca_prevalence_ns * 0.5

        # Treatment costs
        costs_tx = np.array([np.zeros(length_df)] * sims)

        costs_tx[:,0] = ((pca_incidence_ns_cohort[:,0]
                          * psa_stage_local[:,0].T
                          * tx_costs_local.T).sum(axis=0)

                         + (pca_incidence_ns_cohort[:,0]
                            * psa_stage_adv[:,0].T
                            * tx_costs_adv.T).sum(axis=0)

                        * relative_cost_clinically_detected[:,0]) # this variable is tiled to reach 45 - each level is the same

        # Year 2 onwards
        ################
        total_cycles = length_df
        for i in range(1, total_cycles):

           # Cohorts, numbers 'healthy', and incident cases
            cohort[:,i] = cohort[:,i-1] - total_death[:,i-1]

            pca_alive[:,i] = (pca_alive[:,i-1]
                              + pca_incidence_ns_cohort[:,i-1]
                              - pca_death[:,i-1]
                              - pca_death_other[:,i-1]) # PCa alive at the beginning of the year

            healthy[:,i] = (cohort[:,i] - pca_alive[:,i])

            pca_incidence_ns_cohort[:,i] = healthy[:,i] * tr_incidence[:,i]

            # Deaths
            pca_death[:,i] = ((pca_alive[:,i] * tr_pca_death_baseline[:,i])
                              + (healthy[:,i] * tr_pca_death_baseline[:,i]))

            pca_death_other[:,i] = ((pca_incidence_ns_cohort[:,i]
                                     + pca_alive[:,i]
                                     - pca_death[:,i])
                                    * tr_death_other_causes[:,i])

            healthy_death_other[:,i] = ((healthy[:,i] - pca_incidence_ns_cohort[:,i])
                                        * tr_death_other_causes[:,i])

            total_death[:,i] = (pca_death[:,i]
                                + pca_death_other[:,i]
                                + healthy_death_other[:,i])

            # Prevalent cases & life-years
            pca_prevalence_ns[:,i] = (pca_incidence_ns_cohort[:,i]
                                      + pca_alive[:,i]
                                      - pca_death[:,i]
                                      - pca_death_other[:,i])

            lyrs_pca_nodiscount[:,i] = ((pca_prevalence_ns[:,i-1]
                                           + pca_prevalence_ns[:,i])
                                          * 0.5)

            # Costs
            costs_tx[:,i] = ((pca_incidence_ns_cohort[:,i]
                              * psa_stage_local[:,i].T
                              * tx_costs_local.T).sum(axis=0)

                             + (pca_incidence_ns_cohort[:,i]
                                * psa_stage_adv[:,i].T
                                * tx_costs_adv.T).sum(axis=0)

                            * relative_cost_clinically_detected[:,i])

        ############
        # Outcomes #
        ############

        # INDEX:
        # s_ = sim (this is the sum across the simulations i.e. one total value per simulation)
        # m_ = mean (this is the mean across the simulations i.e. one value for each year of the model)
        # t_ = total
        # nodiscount = not discounted
        # discount = discounted
        # _ns = outcomes for the no screening cohort

        # Total incident cases
        ######################
        s_cases_ns, m_cases_ns, t_cases_ns = outcomes(pca_incidence_ns_cohort)

        # PCa alive
        s_pca_alive_ns, m_pca_alive_ns, t_pca_alive_ns = outcomes(pca_alive)

        # Healthy
        s_healthy_ns, m_healthy_ns, t_healthy_ns = outcomes(healthy)

        # Deaths from other causes amongst prostate cancer cases
        s_pca_deaths_other_ns, m_pca_deaths_other_ns, t_pca_deaths_other_ns = outcomes(pca_death_other)

        # Deaths from other causes amongst the healthy
        (s_healthy_deaths_other_ns,
         m_healthy_deaths_other_ns,
         t_healthy_deaths_other_ns) = outcomes(healthy_death_other)

        # Total deaths from other causes
        ################################
        deaths_other_ns = pca_death_other + healthy_death_other
        s_deaths_other_ns, m_deaths_other_ns, t_deaths_other_ns = outcomes(deaths_other_ns)

        # Total deaths from prostate cancer
        ###################################
        s_deaths_pca_ns, m_deaths_pca_ns, t_deaths_pca_ns = outcomes(pca_death)

        # Life-years ('healthy')
        lyrs_healthy_nodiscount_ns = healthy-(0.5 * (healthy_death_other + pca_incidence_ns_cohort))

        (s_lyrs_healthy_nodiscount_ns,
         m_lyrs_healthy_nodiscount_ns,
         t_lyrs_healthy_nodiscount_ns) = outcomes(lyrs_healthy_nodiscount_ns)

        lyrs_healthy_discount_ns = lyrs_healthy_nodiscount_ns * discount_factor[:total_cycles]

        (s_lyrs_healthy_discount_ns,
         m_lyrs_healthy_discount_ns,
         t_lyrs_healthy_discount_ns) = outcomes(lyrs_healthy_discount_ns)

        # Life-years with prostate cancer
        lyrs_pca_discount_ns = lyrs_pca_nodiscount * discount_factor[:total_cycles]

        (s_lyrs_pca_discount_ns,
         m_lyrs_pca_discount_ns,
         t_lyrs_pca_discount_ns) = outcomes(lyrs_pca_discount_ns)

        # Total life-years
        ##################
        lyrs_nodiscount_ns = lyrs_healthy_nodiscount_ns + lyrs_pca_nodiscount

        (s_lyrs_nodiscount_ns,
         m_lyrs_nodiscount_ns,
         t_lyrs_nodiscount_ns) = outcomes(lyrs_nodiscount_ns)

        lyrs_discount_ns = lyrs_healthy_discount_ns + lyrs_pca_discount_ns

        (s_lyrs_discount_ns,
         m_lyrs_discount_ns,
         t_lyrs_discount_ns) = outcomes(lyrs_discount_ns)

        # QALYs in the healthy
        qalys_healthy_nodiscount_ns = lyrs_healthy_nodiscount_ns * utility_background_psa[:,year-45:]
        qalys_healthy_discount_ns = lyrs_healthy_discount_ns * utility_background_psa[:,year-45:]

        (s_qalys_healthy_discount_ns,
         m_qalys_healthy_discount_ns,
         t_qalys_healthy_discount_ns) = outcomes(qalys_healthy_discount_ns)

        # QALYs with prostate cancer
        qalys_pca_nodiscount_ns = lyrs_pca_nodiscount * pca_incidence_utility_psa[:,year-45:]
        qalys_pca_discount_ns = lyrs_pca_discount_ns * pca_incidence_utility_psa[:,year-45:]

        (s_qalys_pca_discount_ns,
         m_qalys_pca_discount_ns,
         t_qalys_pca_discount_ns) = outcomes(qalys_pca_discount_ns)

        # Total QALYs
        #############
        qalys_nodiscount_ns = qalys_healthy_nodiscount_ns + qalys_pca_nodiscount_ns

        (s_qalys_nodiscount_ns,
         m_qalys_nodiscount_ns,
         t_qalys_nodiscount_ns) = outcomes(qalys_nodiscount_ns)

        qalys_discount_ns = qalys_healthy_discount_ns + qalys_pca_discount_ns

        (s_qalys_discount_ns,
         m_qalys_discount_ns,
         t_qalys_discount_ns) = outcomes(qalys_discount_ns)

        # Cost of PSA testing
        n_psa_tests_ns = ((pca_incidence_ns_cohort / p_suspected_ns[:,year-45:])
                           + ((pca_incidence_ns_cohort * (1-uptake_biopsy[year-45:]))
                              / p_suspected_refuse_biopsy_ns[:,year-45:])) * n_psa_tests[:,year-45:]

        (s_n_psa_tests_ns,
         m_n_psa_tests_ns,
         total_n_psa_tests_ns) = outcomes(n_psa_tests_ns)

        cost_psa_testing_nodiscount_ns = n_psa_tests_ns * cost_psa[:,year-45:] * relative_cost_clinically_detected[:,year-45:]

        (s_cost_psa_testing_nodiscount_ns,
         m_cost_psa_testing_nodiscount_ns,
         t_cost_psa_testing_nodiscount_ns) = outcomes(cost_psa_testing_nodiscount_ns)

        cost_psa_testing_discount_ns = cost_psa_testing_nodiscount_ns * discount_factor[:total_cycles]

        (s_cost_psa_testing_discount_ns,
         m_cost_psa_testing_discount_ns,
         t_cost_psa_testing_discount_ns) = outcomes(cost_psa_testing_discount_ns)

        # Cost of suspected cancer - biopsies
        n_biopsies_ns = pca_incidence_ns_cohort / p_suspected_ns[:,year-45:]

        (s_n_biopsies_ns,
         m_n_biopsies_ns,
         total_n_biopsies_ns) = outcomes(n_biopsies_ns)

        cost_biopsy_nodiscount_ns = (((pca_incidence_ns_cohort / p_suspected_ns[:,year-45:])
                                      * cost_biopsy[:,year-45:])

                                     + (((pca_incidence_ns_cohort * (1-uptake_biopsy[year-45:]))
                                          / p_suspected_refuse_biopsy_ns[:,year-45:])
                                         * cost_refuse_biopsy[:,year-45:])

                                    * relative_cost_clinically_detected[:,year-45:])

        (s_cost_biopsy_nodiscount_ns,
         m_cost_biopsy_nodiscount_ns,
         t_cost_biopsy_nodiscount_ns) = outcomes(cost_biopsy_nodiscount_ns)

        cost_biopsy_discount_ns = cost_biopsy_nodiscount_ns * discount_factor[:total_cycles]

        (s_cost_biopsy_discount_ns,
         m_cost_biopsy_discount_ns,
         t_cost_biopsy_discount_ns) = outcomes(cost_biopsy_discount_ns)

        # Cost of staging
        cost_staging_nodiscount_ns = (cost_assessment
                                      * psa_stage_adv.T
                                      * pca_incidence_ns_cohort.T
                                      * relative_cost_clinically_detected[:,year-45:].T).T

        (s_cost_staging_nodiscount_ns,
         m_cost_staging_nodiscount_ns,
         t_cost_staging_nodiscount_ns) = outcomes(cost_staging_nodiscount_ns)

        cost_staging_discount_ns = cost_staging_nodiscount_ns * discount_factor[:total_cycles]

        (s_cost_staging_discount_ns,
         m_cost_staging_discount_ns,
         t_cost_staging_discount_ns) = outcomes(cost_staging_discount_ns)

        # Cost in last 12 months of life
        cost_eol_nodiscount_ns = (pca_death_costs * pca_death.T).T

        (s_cost_eol_nodiscount_ns,
         m_cost_eol_nodiscount_ns,
         t_cost_eol_nodiscount_ns) = outcomes(cost_eol_nodiscount_ns)

        cost_eol_discount_ns = cost_eol_nodiscount_ns * discount_factor[:total_cycles]

        (s_cost_eol_discount_ns,
         m_cost_eol_discount_ns,
         t_cost_eol_discount_ns) = outcomes(cost_eol_discount_ns)

        # Costs of treatment
        (s_cost_tx_nodiscount_ns,
         m_cost_tx_nodiscount_ns,
         t_cost_tx_nodiscount_ns) = outcomes(costs_tx)

        cost_tx_discount_ns = costs_tx * discount_factor[:total_cycles]

        (s_cost_tx_discount_ns,
         m_cost_tx_discount_ns,
         t_cost_tx_discount_ns) = outcomes(cost_tx_discount_ns)

        # Amalgamated costs
        cost_nodiscount_ns = (cost_psa_testing_nodiscount_ns
                              + cost_biopsy_nodiscount_ns
                              + cost_staging_nodiscount_ns
                              + costs_tx
                              + cost_eol_nodiscount_ns)

        (s_cost_nodiscount_ns,
         m_cost_nodiscount_ns,
         t_cost_nodiscount_ns) = outcomes(cost_nodiscount_ns)

        cost_discount_ns = (cost_psa_testing_discount_ns
                            + cost_biopsy_discount_ns
                            + cost_staging_discount_ns
                            + cost_tx_discount_ns
                            + cost_eol_discount_ns)

        (s_cost_discount_ns,
         m_cost_discount_ns,
         t_cost_discount_ns) = outcomes(cost_discount_ns)

        # Generate a mean dataframe
        ns_matrix = [age, m_cases_ns, m_deaths_other_ns, m_deaths_pca_ns,
                     m_pca_alive_ns, m_healthy_ns, m_lyrs_healthy_nodiscount_ns,
                     m_lyrs_healthy_discount_ns, m_lyrs_pca_discount_ns, m_lyrs_discount_ns,
                     m_qalys_healthy_discount_ns, m_qalys_pca_discount_ns, m_qalys_discount_ns,
                     m_cost_psa_testing_discount_ns, m_cost_biopsy_discount_ns, m_cost_staging_discount_ns,
                     m_cost_tx_discount_ns, m_cost_eol_discount_ns, m_cost_discount_ns]

        ns_columns = ['age', 'pca_cases', 'deaths_other', 'deaths_pca',
                      'pca_alive', 'healthy', 'lyrs_healthy_nodiscount', 'lyrs_healthy_discount',
                      'lyrs_pca_discount', 'total_lyrs_discount',
                      'qalys_healthy_discount', 'qalys_pca_discount', 'total_qalys_discount',
                      'cost_psa_testing_discount', 'cost_biopsy_discount', 'cost_staging_discount',
                      'cost_treatment_discount', 'costs_eol_discount', 'total_cost_discount']

        ns_cohort = pd.DataFrame(ns_matrix, index = ns_columns).T

        t_parameters_ns = [year, t_cases_ns, t_deaths_pca_ns,
                           t_deaths_other_ns,
                           t_lyrs_healthy_discount_ns, t_lyrs_pca_discount_ns,
                           t_lyrs_nodiscount_ns, t_lyrs_discount_ns,
                           t_qalys_healthy_discount_ns, t_qalys_pca_discount_ns,
                           t_qalys_nodiscount_ns, t_qalys_discount_ns,
                           t_cost_psa_testing_nodiscount_ns, t_cost_psa_testing_discount_ns,
                           t_cost_biopsy_nodiscount_ns, t_cost_biopsy_discount_ns,
                           t_cost_staging_nodiscount_ns, t_cost_staging_discount_ns,
                           t_cost_eol_nodiscount_ns, t_cost_eol_discount_ns,
                           t_cost_tx_nodiscount_ns, t_cost_tx_discount_ns,
                           t_cost_nodiscount_ns, t_cost_discount_ns,
                           total_n_psa_tests_ns, total_n_biopsies_ns]

        columns_ns = ['cohort_age_at_start', 'pca_cases',
                      'pca_deaths', 'deaths_other_causes', 'lyrs_healthy_discounted',
                      'lyrs_pca_discounted', 'lyrs_undiscounted', 'lyrs_discounted',
                      'qalys_healthy_discounted', 'qalys_pca_discounted',
                      'qalys_undiscounted', 'qalys_discounted',
                      'cost_psa_testing_undiscounted', 'cost_psa_testing_discounted',
                      'cost_biopsy_undiscounted', 'cost_biopsy_discounted',
                      'cost_staging_undiscounted', 'cost_staging_discounted',
                      'cost_eol_undiscounted', 'cost_eol_discounted',
                      'cost_treatment_undiscounted', 'cost_treatment_discounted',
                      'costs_undiscounted', 'costs_discounted', 'n_psa_tests', 'n_biopsies']

        outcomes_ns_psa = pd.DataFrame(t_parameters_ns, index = columns_ns).T
        outcomes_ns_psa['overdiagnosis'] = 0

        parameters_ns = [s_qalys_discount_ns, s_cost_discount_ns, s_deaths_pca_ns,
                         ns_cohort, outcomes_ns_psa]

        for index, parameter in enumerate(parameter_list_ns):
            parameter = gen_list_outcomes(parameter_list_ns[index], parameters_ns[index])

                                                #######################
                                                # Age-based screening #
                                                #######################

        ###################################
        # Specific transition probabilities
        ###################################

        if year < 55:
            # Yearly probability of PCa incidence
            smoothed_pca_incidence_age = psa_function(pca_incidence[year-45:])

            # Yearly probability of death from PCa - smoothed entry and exit
            smoothed_pca_mortality_age = psa_function(pca_death_baseline[year-45:])

            # Proportion of cancers detected by screening at an advanced stage
            stage_screened_adv = psa_function(stage_adv)
            psa_stage_screened_adv = stage_screened_adv[:,year-45:]

            # Proportion of cancers detected by screening at a localised stage
            stage_screened_local = 1-stage_screened_adv
            psa_stage_screened_local = stage_screened_local[:,year-45:]

        if year > 54:
            # Yearly probability of PCa incidence
            smoothed_pca_incidence = psa_function(pca_incidence)
            smoothed_pca_incidence[:,10:25] = (smoothed_pca_incidence[:,10:25].T * rr_incidence[year-45,:]).T
            smoothed_pca_incidence[:,25:35] = (smoothed_pca_incidence[:,25:35] * np.linspace(post_sc_incidence_drop,1,10))
            smoothed_pca_incidence_age = smoothed_pca_incidence[:,year-45:]

            # Yearly probability of death from PCa - smoothed entry and exit
            smoothed_pca_mortality = psa_function(pca_death_baseline)
            smoothed_pca_mortality[:,10:15] = smoothed_pca_mortality[:,10:15] * np.linspace(1,0.79,5)
            smoothed_pca_mortality[:,15:] = smoothed_pca_mortality[:,15:] * rr_death_screening[:,15:]
            smoothed_pca_mortality_age = smoothed_pca_mortality[:,year-45:]

            # Proportion of cancers detected by screening at a localised / advanced stage
            stage_screened_adv = stage_adv * rr_adv_screening
            stage_screened_local = 1-stage_screened_adv
            psa_stage_screened_local = stage_screened_local[:,year-45:]
            psa_stage_screened_adv = stage_screened_adv[:,year-45:]

        #######################
        # Year 1 in the model #
        #######################

        age = np.arange(year,90)
        length_df = len(age)
        length_screen = len(np.arange(year,70)) # number of screening years depending on age cohort starting

        # Cohorts, numbers healthy, and incident cases
        cohort_sc = np.array([np.repeat(pop[year], length_df)] * sims) * uptake_psa
        cohort_ns = np.array([np.repeat(pop[year], length_df)] * sims) * (1-uptake_psa)

        pca_alive_sc = np.array([np.zeros(length_df)] * sims)
        pca_alive_ns = np.array([np.zeros(length_df)] * sims)

        healthy_sc = cohort_sc - pca_alive_sc
        healthy_ns = cohort_ns - pca_alive_ns

        pca_incidence_sc = healthy_sc * smoothed_pca_incidence_age # Total incidence in screened arm

        if year > 54:
            pca_incidence_screened = pca_incidence_sc.copy()
            pca_incidence_post_screening = np.array([np.zeros(length_df)] * sims) # Post-screening cancers - 0 until model reaches age 70.

        elif year < 55:
            pca_incidence_screened = np.array([np.zeros(length_df)] * sims)
            pca_incidence_post_screening = np.array([np.zeros(length_df)] * sims) # post-screening cancers 0 as no screening (needed for later code to run smoothly)

        pca_incidence_ns = healthy_ns * tr_incidence # Incidence in non-screened

        # Deaths
        pca_death_sc = ((pca_alive_sc * smoothed_pca_mortality_age)
                        + (healthy_sc * smoothed_pca_mortality_age))

        pca_death_ns = ((pca_alive_ns * tr_pca_death_baseline)
                        + (healthy_ns * tr_pca_death_baseline))

        pca_death_other_sc = ((pca_incidence_sc
                               + pca_alive_sc
                               - pca_death_sc)
                              * tr_death_other_causes)

        pca_death_other_ns = ((pca_incidence_ns
                               + pca_alive_ns
                               - pca_death_ns)
                              * tr_death_other_causes)

        healthy_death_other_sc = ((healthy_sc - pca_incidence_sc)
                                  * tr_death_other_causes)

        healthy_death_other_ns = ((healthy_ns - pca_incidence_ns)
                                  * tr_death_other_causes)

        t_death_sc = (pca_death_sc
                      + pca_death_other_sc
                      + healthy_death_other_sc) # Total deaths screened arm

        t_death_ns = (pca_death_ns
                      + pca_death_other_ns
                      + healthy_death_other_ns) # Total deaths non-screened arm

        t_death = t_death_sc + t_death_ns # Total deaths

        # Prevalent cases & life-years
        pca_prevalence_sc = (pca_incidence_sc
                             - pca_death_sc
                             - pca_death_other_sc)

        pca_prevalence_ns = (pca_incidence_ns
                             - pca_death_ns
                             - pca_death_other_ns)

        lyrs_pca_sc_nodiscount = pca_prevalence_sc * 0.5
        lyrs_pca_ns_nodiscount = pca_prevalence_ns * 0.5

        # Costs
        if year > 54:
            costs_tx_screened = np.array([np.zeros(length_df)] * sims)
            costs_tx_post_screening = np.array([np.zeros(length_df)] * sims)

            costs_tx_screened[:,0] = ((pca_incidence_screened[:,0]
                                       * psa_stage_screened_local[:,0].T
                                       * tx_costs_local.T).sum(axis=0)

                                      + (pca_incidence_screened[:,0]
                                        * psa_stage_screened_adv[:,0].T
                                        * tx_costs_adv.T).sum(axis=0)) # cost of screen-detected cancers

            costs_tx_post_screening[:,0] = ((pca_incidence_post_screening[:,0]
                                             * psa_stage_local[:,0].T
                                             * tx_costs_local.T).sum(axis=0)

                                            + (pca_incidence_post_screening[:,0]
                                               * psa_stage_adv[:,0].T
                                               * tx_costs_adv.T).sum(axis=0)

                                           * relative_cost_clinically_detected[:,0])

            costs_tx_sc = np.array([np.zeros(length_df)] * sims)
            costs_tx_sc[:,0] = (costs_tx_screened[:,0] + costs_tx_post_screening[:,0]) # total cost in screened arms

        elif year < 55:
            costs_tx_sc = np.array([np.zeros(length_df)] * sims)
            costs_tx_sc[:,0] =  ((pca_incidence_sc[:,0]
                                 * psa_stage_local[:,0].T
                                 * tx_costs_local.T).sum(axis=0)

                                + (pca_incidence_sc[:,0]
                                   * psa_stage_adv[:,0].T
                                   * tx_costs_adv.T).sum(axis=0)

                                * relative_cost_clinically_detected[:,0])

        costs_tx_ns = np.array([np.zeros(length_df)] * sims)
        costs_tx_ns[:,0] =  ((pca_incidence_ns[:,0]
                             * psa_stage_local[:,0].T
                             * tx_costs_local.T).sum(axis=0)

                            + (pca_incidence_ns[:,0]
                               * psa_stage_adv[:,0].T
                               * tx_costs_adv.T).sum(axis=0)

                            * relative_cost_clinically_detected[:,0])


        ##################
        # Year 2 onwards #
        ##################
        total_cycles = length_df
        for i in range(1, total_cycles):

           # Cohorts, numbers healthy, incident & prevalent cases
            cohort_sc[:,i] = cohort_sc[:,i-1] - t_death_sc[:,i-1]
            cohort_ns[:,i] = cohort_ns[:,i-1] - t_death_ns[:,i-1]

            pca_alive_sc[:,i] = (pca_alive_sc[:,i-1]
                                 + pca_incidence_sc[:,i-1]
                                 - pca_death_sc[:,i-1]
                                 - pca_death_other_sc[:,i-1])

            pca_alive_ns[:,i] = (pca_alive_ns[:,i-1]
                                 + pca_incidence_ns[:,i-1]
                                 - pca_death_ns[:,i-1]
                                 - pca_death_other_ns[:,i-1])

            healthy_sc[:,i] = (cohort_sc[:,i] - pca_alive_sc[:,i])
            healthy_ns[:,i] = (cohort_ns[:,i] - pca_alive_ns[:,i])

            pca_incidence_sc[:,i] = healthy_sc[:,i] * smoothed_pca_incidence_age[:,i]

            if year > 54:
                if i < length_screen:
                    pca_incidence_screened[:,i] = pca_incidence_sc[:,i].copy() # Screen-detected cancers
                    pca_incidence_post_screening[:,i] = 0

                else:
                    pca_incidence_screened[:,i] = 0 # Screen-detected cancers
                    pca_incidence_post_screening[:,i] = pca_incidence_sc[:,i].copy()

            elif year < 55:
                pca_incidence_screened[:,i] = 0 # Screen-detected cancers
                pca_incidence_post_screening[:,i] = 0 # post-screening cancers 0 as no screening (needed for later code to run smoothly)

            pca_incidence_ns[:,i] = healthy_ns[:,i] * tr_incidence[:,i]

            # Deaths
            pca_death_sc[:,i] = ((pca_alive_sc[:,i] * smoothed_pca_mortality_age[:,i])
                                 + (healthy_sc[:,i] * smoothed_pca_mortality_age[:,i]))

            pca_death_ns[:,i] = ((pca_alive_ns[:,i] * tr_pca_death_baseline[:,i])
                                 + (healthy_ns[:,i] * tr_pca_death_baseline[:,i]))

            pca_death_other_sc[:,i] = ((pca_incidence_sc[:,i]
                                        + pca_alive_sc[:,i]
                                        - pca_death_sc[:,i])
                                       * tr_death_other_causes[:,i])

            pca_death_other_ns[:,i] = ((pca_incidence_ns[:,i]
                                        + pca_alive_ns[:,i]
                                        - pca_death_ns[:,i])
                                       * tr_death_other_causes[:,i])

            healthy_death_other_sc[:,i] = ((healthy_sc[:,i] - pca_incidence_sc[:,i])
                                           * tr_death_other_causes[:,i])

            healthy_death_other_ns[:,i] = ((healthy_ns[:,i] - pca_incidence_ns[:,i])
                                           * tr_death_other_causes[:,i])

            t_death_sc[:,i] = (pca_death_sc[:,i]
                               + pca_death_other_sc[:,i]
                               + healthy_death_other_sc[:,i])

            t_death_ns[:,i] = (pca_death_ns[:,i]
                               + pca_death_other_ns[:,i]
                               + healthy_death_other_ns[:,i])

            t_death[:,i] = t_death_sc[:,i] + t_death_ns[:,i]

            # Prevalent cases & life-years
            pca_prevalence_sc[:,i] = (pca_incidence_sc[:,i]
                                      + pca_alive_sc[:,i]
                                      - pca_death_sc[:,i]
                                      - pca_death_other_sc[:,i])

            pca_prevalence_ns[:,i] = (pca_incidence_ns [:,i]
                                      + pca_alive_ns[:,i]
                                      - pca_death_ns[:,i]
                                      - pca_death_other_ns[:,i])

            lyrs_pca_sc_nodiscount[:,i] = ((pca_prevalence_sc[:,i-1]
                                              + pca_prevalence_sc[:,i])
                                             * 0.5)

            lyrs_pca_ns_nodiscount[:,i] = ((pca_prevalence_ns[:,i-1]
                                              + pca_prevalence_ns[:,i])
                                             * 0.5)

            # Costs
            if year > 54:
                costs_tx_screened[:,i] = ((pca_incidence_screened[:,i]
                                            * psa_stage_screened_local[:,i].T
                                            * tx_costs_local.T).sum(axis=0)

                                          + (pca_incidence_screened[:,i]
                                            * psa_stage_screened_adv[:,i].T
                                            * tx_costs_adv.T).sum(axis=0)) # cost of screen-detected cancers

                costs_tx_post_screening[:,i] = ((pca_incidence_post_screening[:,i]
                                                 * psa_stage_local[:,i].T
                                                 * tx_costs_local.T).sum(axis=0)

                                                + (pca_incidence_post_screening[:,i]
                                                   * psa_stage_adv[:,i].T
                                                   * tx_costs_adv.T).sum(axis=0)

                                               * relative_cost_clinically_detected[:,i])

                costs_tx_sc[:,i] = (costs_tx_screened[:,i]
                                    + costs_tx_post_screening[:,i]) # total cost in screened arms

            elif year < 55:
                costs_tx_sc[:,i] =  ((pca_incidence_sc[:,i]
                                     * psa_stage_local[:,i].T
                                     * tx_costs_local.T).sum(axis=0)

                                    + (pca_incidence_sc[:,i]
                                       * psa_stage_adv[:,i].T
                                       * tx_costs_adv.T).sum(axis=0)

                                    * relative_cost_clinically_detected[:,i])

            costs_tx_ns[:,i] =  ((pca_incidence_ns[:,i]
                                 * psa_stage_local[:,i].T
                                 * tx_costs_local.T).sum(axis=0)

                                + (pca_incidence_ns[:,i]
                                   * psa_stage_adv[:,i].T
                                   * tx_costs_adv.T).sum(axis=0)

                                * relative_cost_clinically_detected[:,i])

        ############
        # Outcomes #
        ############

        # INDEX:
        # s_ = sim (this is the sum across the simulations i.e. one total value per simulation)
        # m_ = mean (this is the mean across the simulations i.e. one value for each year of the model)
        # t_ = total
        # nodiscount = not discounted
        # discount = discounted
        # _age = outcomes for the age-based screening cohort

        # Total incident cases (screened arm)
        s_cases_sc_age, m_cases_sc_age, t_cases_sc_age = outcomes(pca_incidence_sc)

        # Total screen-detected cancers (screened arm)
        s_cases_sc_detected_age, m_cases_sc_detected_age, t_cases_sc_detected_age = outcomes(pca_incidence_screened)

        # Total cancers detected after screening stops (screened arm)
        s_cases_post_screening_age, m_cases_post_screening_age, t_cases_post_screening_age = outcomes(pca_incidence_post_screening)

        # Incident cases (non-screened arm)
        s_cases_ns_age, m_cases_ns_age, t_cases_ns_age = outcomes(pca_incidence_ns)

        # Incident cases (total)
        ########################
        s_cases_age = s_cases_sc_age + s_cases_ns_age
        m_cases_age = m_cases_sc_age + m_cases_ns_age
        t_cases_age = t_cases_sc_age + t_cases_ns_age

        # PCa alive
        s_pca_alive_age, m_pca_alive_age, t_pca_alive_age = outcomes((pca_alive_sc + pca_alive_ns))

        # Healthy
        s_healthy_age, m_healthy_age, t_healthy_age = outcomes((healthy_sc + healthy_ns))

        # Overdiagnosed cases
        overdiagnosis_age = pca_incidence_screened * p_overdiagnosis_psa.T[:,year-45:]
        s_overdiagnosis_age, m_overdiagnosis_age, t_overdiagnosis_age = outcomes(overdiagnosis_age)

        # Deaths from other causes (screened arm)
        deaths_sc_other_age = pca_death_other_sc + healthy_death_other_sc
        s_deaths_sc_other_age, m_deaths_sc_other_age, t_deaths_sc_other_age = outcomes(deaths_sc_other_age)

        # Deaths from other causes (non-screened arm)
        deaths_ns_other_age = pca_death_other_ns + healthy_death_other_ns
        s_deaths_ns_other_age, m_deaths_ns_other_age, t_deaths_ns_other_age = outcomes(deaths_ns_other_age)

        # Deaths from other causes (total)
        s_deaths_other_age = s_deaths_sc_other_age + s_deaths_ns_other_age
        m_deaths_other_age = m_deaths_sc_other_age + m_deaths_ns_other_age
        t_deaths_other_age = t_deaths_sc_other_age + t_deaths_ns_other_age

        # Deaths from prosate cancer (screened arm)
        s_deaths_sc_pca_age, m_deaths_sc_pca_age, t_deaths_sc_pca_age = outcomes(pca_death_sc)

        # Deaths from prosate cancer (non-screened arm)
        s_deaths_ns_pca_age, m_deaths_ns_pca_age, t_deaths_ns_pca_age = outcomes(pca_death_ns)

        # Deaths from prosate cancer (total)
        ####################################
        s_deaths_pca_age = s_deaths_sc_pca_age + s_deaths_ns_pca_age
        m_deaths_pca_age = m_deaths_sc_pca_age + m_deaths_ns_pca_age
        t_deaths_pca_age = t_deaths_sc_pca_age + t_deaths_ns_pca_age

        # Healthy life-years (screened arm)
        lyrs_healthy_sc_nodiscount_age = (healthy_sc
                                          - (0.5 * (healthy_death_other_sc+pca_incidence_sc)))

        lyrs_healthy_sc_discount_age = lyrs_healthy_sc_nodiscount_age * discount_factor[:total_cycles]

        (s_lyrs_healthy_sc_discount_age,
         m_lyrs_healthy_sc_discount_age,
         t_lyrs_healthy_sc_discount_age) = outcomes(lyrs_healthy_sc_discount_age)

        # Healthy life-years (non-screened arm)
        lyrs_healthy_ns_nodiscount_age = (healthy_ns
                                          - (0.5 * (healthy_death_other_ns+pca_incidence_ns)))

        lyrs_healthy_ns_discount_age = lyrs_healthy_ns_nodiscount_age * discount_factor[:total_cycles]

        (s_lyrs_healthy_ns_discount_age,
         m_lyrs_healthy_ns_discount_age,
         t_lyrs_healthy_ns_discount_age) = outcomes(lyrs_healthy_ns_discount_age)

        # Total healthy life-years
        lyrs_healthy_nodiscount_age = lyrs_healthy_sc_nodiscount_age + lyrs_healthy_ns_nodiscount_age

        (s_lyrs_healthy_nodiscount_age,
         m_lyrs_healthy_nodiscount_age,
         t_lyrs_healthy_nodiscount_age) = outcomes(lyrs_healthy_nodiscount_age)

        lyrs_healthy_discount_age = lyrs_healthy_nodiscount_age * discount_factor[:total_cycles]

        (s_lyrs_healthy_discount_age,
         m_lyrs_healthy_discount_age,
         t_lyrs_healthy_discount_age) = outcomes(lyrs_healthy_discount_age)

        # Life-years with prostate cancer in screened arm
        lyrs_pca_sc_discount = lyrs_pca_sc_nodiscount * discount_factor[:total_cycles]

        (s_lyrs_pca_sc_discount_age,
         m_lyrs_pca_sc_discount_age,
         t_lyrs_pca_sc_discount_age) = outcomes(lyrs_pca_sc_discount)

        # Life-years with prostate cancer in non-screened arm
        lyrs_pca_ns_discount = lyrs_pca_ns_nodiscount * discount_factor[:total_cycles]

        (s_lyrs_pca_ns_discount_age,
         m_lyrs_pca_ns_discount_age,
         t_lyrs_pca_ns_age) = outcomes(lyrs_pca_ns_discount)

        #  Life-years with prostate cancer in both arms
        lyrs_pca_nodiscount_age = lyrs_pca_sc_nodiscount + lyrs_pca_ns_nodiscount
        lyrs_pca_discount_age = lyrs_pca_sc_discount + lyrs_pca_ns_discount

        (s_lyrs_pca_discount_age,
         m_lyrs_pca_discount_age,
         t_lyrs_pca_discount_age) = outcomes(lyrs_pca_discount_age)

        # Total life-years
        ##################
        lyrs_nodiscount_age = lyrs_healthy_nodiscount_age + lyrs_pca_nodiscount_age

        (s_lyrs_nodiscount_age,
         m_lyrs_nodiscount_age,
         t_lyrs_nodiscount_age) = outcomes(lyrs_nodiscount_age)

        lyrs_discount_age = lyrs_healthy_discount_age + lyrs_pca_discount_age

        (s_lyrs_discount_age,
         m_lyrs_discount_age,
         t_lyrs_discount_age) = outcomes(lyrs_discount_age)

        # QALYs (healthy life) - screened arm
        qalys_healthy_sc_nodiscount_age = lyrs_healthy_sc_nodiscount_age * utility_background_psa[:,year-45:]
        qalys_healthy_sc_discount_age = lyrs_healthy_sc_discount_age * utility_background_psa[:,year-45:]

        (s_qalys_healthy_sc_discount_age,
         m_qalys_healthy_sc_discount_age,
         t_qalys_healthy_sc_discount_age) = outcomes(qalys_healthy_sc_discount_age)

        # QALYs (healthy life) - non-screened arm
        qalys_healthy_ns_nodiscount_age = lyrs_healthy_ns_nodiscount_age * utility_background_psa[:,year-45:]
        qalys_healthy_ns_discount_age = lyrs_healthy_ns_discount_age * utility_background_psa[:,year-45:]

        (s_qalys_healthy_ns_discount_age,
         m_qalys_healthy_ns_discount_age,
         t_qalys_healthy_ns_discount_age) = outcomes(qalys_healthy_ns_discount_age)

        # Total QALYs (healthy life)
        qalys_healthy_nodiscount_age = lyrs_healthy_nodiscount_age * utility_background_psa[:,year-45:]
        qalys_healthy_discount_age = lyrs_healthy_discount_age * utility_background_psa[:,year-45:]

        (s_qalys_healthy_discount_age,
         m_qalys_healthy_discount_age,
         t_qalys_healthy_discount_age) = outcomes(qalys_healthy_discount_age)

        # QALYS with prostate cancer - screened arm
        qalys_pca_sc_nodiscount_age = lyrs_pca_sc_nodiscount * pca_incidence_utility_psa[:,year-45:]
        qalys_pca_sc_discount_age = lyrs_pca_sc_discount * pca_incidence_utility_psa[:,year-45:]

        (s_qalys_pca_sc_discount_age,
         m_qalys_pca_sc_discount_age,
         t_qalys_pca_sc_discount_age) = outcomes(qalys_pca_sc_discount_age)

        # QALYS with prostate cancer - non-screened arm
        qalys_pca_ns_nodiscount_age = lyrs_pca_ns_nodiscount * pca_incidence_utility_psa[:,year-45:]
        qalys_pca_ns_discount_age = lyrs_pca_ns_discount * pca_incidence_utility_psa[:,year-45:]

        (s_qalys_pca_ns_discount_age,
         m_qalys_pca_ns_discount_age,
         t_qalys_pca_ns_discount_age) = outcomes(qalys_pca_ns_discount_age)

        # Total QALYS with prostate cancer
        qalys_pca_nodiscount_age = lyrs_pca_nodiscount_age * pca_incidence_utility_psa[:,year-45:]
        qalys_pca_discount_age = lyrs_pca_discount_age * pca_incidence_utility_psa[:,year-45:]

        (s_qalys_pca_discount_age,
         m_qalys_pca_discount_age,
         t_qalys_pca_discount_age) = outcomes(qalys_pca_discount_age)

        # Total QALYs
        #############
        qalys_nodiscount_age = qalys_healthy_nodiscount_age + qalys_pca_nodiscount_age

        (s_qalys_nodiscount_age,
         m_qalys_nodiscount_age,
         t_qalys_nodiscount_age) = outcomes(qalys_nodiscount_age)

        qalys_discount_age = qalys_healthy_discount_age + qalys_pca_discount_age

        (s_qalys_discount_age,
         m_qalys_discount_age,
         t_qalys_discount_age) = outcomes(qalys_discount_age)

        # Costs of PSA testing in non-screened arm
        n_psa_tests_ns_age = ((pca_incidence_ns / p_suspected_ns[:,year-45:])
                              + ((pca_incidence_ns * (1-uptake_biopsy[year-45:]))
                                 / p_suspected_refuse_biopsy_ns[:,year-45:])) * n_psa_tests[:,year-45:]

        cost_psa_testing_ns_nodiscount_age = n_psa_tests_ns_age * cost_psa[:,year-45:] * relative_cost_clinically_detected[:,year-45:]

        (s_cost_psa_testing_ns_nodiscount_age,
         m_cost_psa_testing_ns_nodiscount_age,
         t_cost_psa_testing_ns_nodiscount_age) = outcomes(cost_psa_testing_ns_nodiscount_age)

        cost_psa_testing_ns_discount_age = cost_psa_testing_ns_nodiscount_age * discount_factor[:total_cycles]

        (s_cost_psa_testing_ns_discount_age,
         m_cost_psa_testing_ns_discount_age,
         t_cost_psa_testing_ns_discount_age) = outcomes(cost_psa_testing_ns_discount_age)

        # Costs of PSA testing in screened arm (PSA screening every four years)
        # PSA tests during screened and non-screened period
        if year < 55:

            # Assuming all cancers are clinically detected as these cohorts
            # are not eligible for screening (hence p_suspected_ns)
            # This uses 1-uptake biopsy as the original part of the equation works out
            # the number of biopsies which is then multiplied by n_psa_tests to get the number of PSA tests

            n_psa_tests_sc_age = (((pca_incidence_sc / p_suspected_ns[:,year-45:])
                                  + ((pca_incidence_sc * (1-uptake_biopsy[year-45:]))
                                     / p_suspected_refuse_biopsy_ns[:,year-45:]))
                                  * uptake_psa
                                  * n_psa_tests[:,year-45:])

            cost_psa_testing_sc_nodiscount_age = (n_psa_tests_sc_age
                                                  * cost_psa[:,year-45:]
                                                  * relative_cost_clinically_detected[:,year-45:])

        if year > 54:

            # Get the screened years
            lyrs_healthy_screened_nodiscount_age = np.array([np.zeros(length_df)] * sims)
            lyrs_healthy_screened_nodiscount_age[:,:length_screen] = lyrs_healthy_sc_nodiscount_age[:,:length_screen].copy()
            lyrs_healthy_screened_nodiscount_age[:,length_screen:] = 0

            # Population-level PSA testing during screening phase
            n_psa_tests_screened_age = lyrs_healthy_screened_nodiscount_age * uptake_psa / 4

            # Assuming all cancers are clinically detected in the post-screening phase
            n_psa_tests_post_screening_age = (((pca_incidence_post_screening / p_suspected_ns[:,year-45:])
                                              + ((pca_incidence_post_screening * (1-uptake_biopsy[year-45:]))
                                                 / p_suspected_refuse_biopsy_ns[:,year-45:]))
                                              * uptake_psa
                                              * n_psa_tests[:,year-45:])

            # Total PSA tests
            n_psa_tests_sc_age = (n_psa_tests_screened_age + n_psa_tests_post_screening_age)

            cost_psa_testing_screened_age = n_psa_tests_screened_age * cost_psa[:,year-45:]

            cost_psa_testing_post_screening_age = (n_psa_tests_post_screening_age
                                                   * cost_psa[:,year-45:]
                                                   * relative_cost_clinically_detected[:,year-45:])

            cost_psa_testing_sc_nodiscount_age = (cost_psa_testing_screened_age
                                                  + cost_psa_testing_post_screening_age)

        (s_cost_psa_testing_sc_nodiscount_age,
         m_cost_psa_testing_sc_nodiscount_age,
         t_cost_psa_testing_sc_nodiscount_age) = outcomes(cost_psa_testing_sc_nodiscount_age)

        cost_psa_testing_sc_discount_age = cost_psa_testing_sc_nodiscount_age * discount_factor[:total_cycles]

        (s_cost_psa_testing_sc_discount_age,
         m_cost_psa_testing_sc_discount_age,
         t_cost_psa_testing_sc_discount_age) = outcomes(cost_psa_testing_sc_discount_age)

        # Total costs of PSA testing
        ############################
        n_psa_tests_age = n_psa_tests_ns_age + n_psa_tests_sc_age

        (s_n_psa_tests_age,
         m_n_psa_tests_age,
         total_n_psa_tests_age) = outcomes(n_psa_tests_age)

        cost_psa_testing_nodiscount_age = cost_psa_testing_ns_nodiscount_age + cost_psa_testing_sc_nodiscount_age

        (s_cost_psa_testing_nodiscount_age,
         m_cost_psa_testing_nodiscount_age,
         t_cost_psa_testing_nodiscount_age) = outcomes(cost_psa_testing_nodiscount_age)

        cost_psa_testing_discount_age = cost_psa_testing_ns_discount_age + cost_psa_testing_sc_discount_age

        (s_cost_psa_testing_discount_age,
         m_cost_psa_testing_discount_age,
         t_cost_psa_testing_discount_age) = outcomes(cost_psa_testing_discount_age)

        # Costs of biopsy - screened arm
        if year < 55:

            # Assuming all cancers are clinically detected as these cohorts
            # are not eligible for screening (hence p_suspected_ns)
            n_biopsies_sc_age = pca_incidence_sc / p_suspected_ns[:,year-45:]

            # Costs include the costs of those who turn down biopsy
            cost_biopsy_sc_nodiscount_age = (((pca_incidence_sc / p_suspected_ns[:,year-45:])
                                              * cost_biopsy[:,year-45:])

                                             + (((pca_incidence_sc * (1-uptake_biopsy[year-45:]))
                                                 / p_suspected_refuse_biopsy_ns[:,year-45:])
                                                * cost_refuse_biopsy[:,year-45:])

                                            * relative_cost_clinically_detected[:,year-45:])

        if year > 54:

            # Screen-detected cancers
            n_biopsies_screened_age = pca_incidence_screened / p_suspected[:,year-45:]

            cost_biopsy_screened_nodiscount_age = (((pca_incidence_screened / p_suspected[:,year-45:])
                                                    * cost_biopsy[:,year-45:])

                                                   + (((pca_incidence_screened * (1-uptake_biopsy[year-45:]))
                                                       / p_suspected_refuse_biopsy[:,year-45:])
                                                      * cost_refuse_biopsy[:,year-45:]))

            # Assuming all cancers are clinically detected in the post-screening phase
            n_biopsies_post_screening_age = pca_incidence_post_screening / p_suspected_ns[:,year-45:]

            cost_biopsies_post_screening_nodiscount_age = (((pca_incidence_post_screening / p_suspected_ns[:,year-45:])
                                                            * cost_biopsy[:,year-45:])

                                                           + (((pca_incidence_post_screening * (1-uptake_biopsy[year-45:]))
                                                               / p_suspected_refuse_biopsy_ns[:,year-45:])
                                                              * cost_refuse_biopsy[:,year-45:])

                                                           * relative_cost_clinically_detected[:,year-45:])

            # Total biopsies
            n_biopsies_sc_age = (n_biopsies_screened_age + n_biopsies_post_screening_age)

            # Total cost of biopsies
            cost_biopsy_sc_nodiscount_age = (cost_biopsy_screened_nodiscount_age
                                            + cost_biopsies_post_screening_nodiscount_age)

        (s_cost_biopsy_sc_nodiscount_age,
         m_cost_biopsy_sc_nodiscount_age,
         t_cost_biopsy_sc_nodiscount_age) = outcomes(cost_biopsy_sc_nodiscount_age)

        cost_biopsy_sc_discount_age = cost_biopsy_sc_nodiscount_age * discount_factor[:total_cycles]

        (s_cost_biopsy_sc_discount_age,
         m_cost_biopsy_sc_discount_age,
         t_cost_biopsy_sc_discount_age) = outcomes(cost_biopsy_sc_discount_age)

        # Costs of biopsy - non-screened arm
        n_biopsies_ns_age = pca_incidence_ns / p_suspected_ns[:,year-45:]

        cost_biopsy_ns_nodiscount_age = (((pca_incidence_ns / p_suspected_ns[:,year-45:])
                                            * cost_biopsy[:,year-45:])

                                           + (((pca_incidence_ns * (1-uptake_biopsy[year-45:]))
                                               / p_suspected_refuse_biopsy_ns[:,year-45:])
                                              * cost_refuse_biopsy[:,year-45:])

                                        * relative_cost_clinically_detected[:,year-45:])

        (s_cost_biopsy_ns_nodiscount_age,
         m_cost_biopsy_ns_nodiscount_age,
         t_cost_biopsy_ns_nodiscount_age) = outcomes(cost_biopsy_ns_nodiscount_age)

        cost_biopsy_ns_discount_age = cost_biopsy_ns_nodiscount_age * discount_factor[:total_cycles]

        (s_cost_biopsy_ns_discount_age,
         m_cost_biopsy_ns_discount_age,
         t_cost_biopsy_ns_discount_age) = outcomes(cost_biopsy_ns_discount_age)

        # Total costs of biopsy
        #######################
        n_biopsies_age = n_biopsies_sc_age + n_biopsies_ns_age

        (s_n_biopsies_age,
         m_n_biopsies_age,
         total_n_biopsies_age) = outcomes(n_biopsies_age)

        cost_biopsy_nodiscount_age = cost_biopsy_sc_nodiscount_age + cost_biopsy_ns_nodiscount_age

        (s_cost_biopsy_nodiscount_age,
         m_cost_biopsy_nodiscount_age,
         t_cost_biopsy_nodiscount_age) = outcomes(cost_biopsy_nodiscount_age)

        cost_biopsy_discount_age = cost_biopsy_sc_discount_age + cost_biopsy_ns_discount_age

        (s_cost_biopsy_discount_age,
         m_cost_biopsy_discount_age,
         t_cost_biopsy_discount_age) = outcomes(cost_biopsy_discount_age)

        # Cost of staging in the screened arm
        if year < 55:
            cost_staging_sc_nodiscount_age = (cost_assessment
                                              * psa_stage_adv.T
                                              * pca_incidence_sc.T
                                              * relative_cost_clinically_detected[:,year-45:].T).T

        if year > 54:
            cost_staging_screened_nodiscount_age = (cost_assessment
                                                    * psa_stage_screened_adv.T
                                                    * pca_incidence_screened.T).T

            cost_staging_post_screening_nodiscount_age = (cost_assessment
                                                          * psa_stage_adv.T
                                                          * pca_incidence_post_screening.T
                                                          * relative_cost_clinically_detected[:,year-45:].T).T

            cost_staging_sc_nodiscount_age = (cost_staging_screened_nodiscount_age
                                             + cost_staging_post_screening_nodiscount_age)

        (s_cost_staging_sc_nodiscount_age,
         m_cost_staging_sc_nodiscount_age,
         t_cost_staging_sc_nodiscount_age) = outcomes(cost_staging_sc_nodiscount_age)

        cost_staging_sc_discount_age = cost_staging_sc_nodiscount_age * discount_factor[:total_cycles]

        (s_cost_staging_sc_discount_age,
         m_cost_staging_sc_discount_age,
         t_cost_staging_sc_discount_age) = outcomes(cost_staging_sc_discount_age)

        # Cost of staging in the non-screened arm
        cost_staging_ns_nodiscount_age = (cost_assessment
                                          * psa_stage_adv.T
                                          * pca_incidence_ns.T
                                          * relative_cost_clinically_detected[:,year-45:].T).T

        (s_cost_staging_ns_nodiscount_age,
         m_cost_staging_ns_nodiscount_age,
         t_cost_staging_ns_nodiscount_age) = outcomes(cost_staging_ns_nodiscount_age)

        cost_staging_ns_discount_age = cost_staging_ns_nodiscount_age * discount_factor[:total_cycles]

        (s_cost_staging_ns_discount_age,
         m_cost_staging_ns_discount_age,
         t_cost_staging_ns_discount_age) = outcomes(cost_staging_ns_discount_age)

        # Total costs of staging
        ########################
        cost_staging_nodiscount_age = cost_staging_sc_nodiscount_age + cost_staging_ns_nodiscount_age

        (s_cost_staging_nodiscount_age,
         m_cost_staging_nodiscount_age,
         t_cost_staging_nodiscount_age) = outcomes(cost_staging_nodiscount_age)

        cost_staging_discount_age = cost_staging_sc_discount_age + cost_staging_ns_discount_age

        (s_cost_staging_discount_age,
         m_cost_staging_discount_age,
         t_cost_staging_discount_age) = outcomes(cost_staging_discount_age)

        # Cost of treatment in screened arm
        (s_cost_tx_sc_nodiscount_age,
         m_cost_tx_sc_nodiscount_age,
         t_cost_tx_sc_nodiscount_age) = outcomes(costs_tx_sc)

        cost_tx_sc_nodiscount_age = costs_tx_sc * discount_factor[:total_cycles]

        (s_cost_tx_sc_discount_age,
         m_cost_tx_sc_discount_age,
         t_cost_tx_sc_discount_age) = outcomes(cost_tx_sc_nodiscount_age)

        # Cost of treatment in non-screened arm
        (s_cost_tx_ns_nodiscount_age,
         m_cost_tx_ns_nodiscount_age,
         t_cost_tx_ns_nodiscount_age) = outcomes(costs_tx_ns)

        cost_tx_ns_nodiscount_age = costs_tx_ns * discount_factor[:total_cycles]

        (s_cost_tx_ns_discount_age,
         m_cost_tx_ns_discount_age,
         t_cost_tx_ns_discount_age) = outcomes(cost_tx_ns_nodiscount_age)

        # Total costs of treatment
        ##########################
        cost_tx_nodiscount_age = costs_tx_sc + costs_tx_ns

        (s_cost_tx_nodiscount_age,
         m_cost_tx_nodiscount_age,
         t_cost_tx_nodiscount_age) = outcomes(cost_tx_nodiscount_age)

        cost_tx_discount_age = cost_tx_nodiscount_age * discount_factor[:total_cycles]

        (s_cost_tx_discount_age,
         m_cost_tx_discount_age,
         t_cost_tx_discount_age) = outcomes(cost_tx_discount_age)

        # Costs of palliation and death in screened arm
        cost_eol_sc_nodiscount_age = (pca_death_costs * pca_death_sc.T).T

        (s_cost_eol_sc_nodiscount_age,
         m_cost_eol_sc_nodiscount_age,
         t_cost_eol_sc_nodiscount_age) = outcomes(cost_eol_sc_nodiscount_age)

        cost_eol_sc_discount_age = cost_eol_sc_nodiscount_age * discount_factor[:total_cycles]

        (s_cost_eol_sc_discount_age,
         m_cost_eol_sc_discount_age,
         t_cost_eol_sc_discount_age) = outcomes(cost_eol_sc_discount_age)

        # Costs of palliation and death in non-screened arm
        cost_eol_ns_nodiscount_age = (pca_death_costs * pca_death_ns.T).T

        (s_cost_eol_ns_nodiscount_age,
         m_cost_eol_ns_nodiscount_age,
         t_cost_eol_ns_nodiscount_age) = outcomes(cost_eol_ns_nodiscount_age)

        cost_eol_ns_discount_age = cost_eol_ns_nodiscount_age * discount_factor[:total_cycles]

        (s_cost_eol_ns_discount_age,
         m_cost_eol_ns_discount_age,
         t_cost_eol_ns_discount_age) = outcomes(cost_eol_ns_discount_age)

        # Total costs of palliation and death
        cost_eol_nodiscount_age = cost_eol_sc_nodiscount_age + cost_eol_ns_nodiscount_age

        (s_cost_eol_nodiscount_age,
         m_cost_eol_nodiscount_age,
         t_cost_eol_nodiscount_age) = outcomes(cost_eol_nodiscount_age)

        cost_eol_discount_age = cost_eol_sc_discount_age + cost_eol_ns_discount_age

        (s_cost_eol_discount_age,
         m_cost_eol_discount_age,
         t_cost_eol_discount_age) = outcomes(cost_eol_discount_age)

        # TOTAL COSTS AGE-BASED SCREENING
        #################################
        cost_nodiscount_age = (cost_psa_testing_nodiscount_age
                               + cost_biopsy_nodiscount_age
                               + cost_staging_nodiscount_age
                               + cost_tx_nodiscount_age
                               + cost_eol_nodiscount_age)

        s_cost_nodiscount_age, m_cost_nodiscount_age, t_cost_nodiscount_age = outcomes(cost_nodiscount_age)

        cost_discount_age = (cost_psa_testing_discount_age
                             + cost_biopsy_discount_age
                             + cost_staging_discount_age
                             + cost_tx_discount_age
                             + cost_eol_discount_age)

        s_cost_discount_age, m_cost_discount_age, t_cost_discount_age = outcomes(cost_discount_age)

        # Generate a mean dataframe
        age_matrix = [age, m_cases_age, m_cases_sc_detected_age,
                      m_cases_post_screening_age, m_overdiagnosis_age, m_deaths_other_age, m_deaths_pca_age,
                      m_pca_alive_ns, m_healthy_age, m_lyrs_healthy_nodiscount_age,
                      m_lyrs_healthy_discount_age, m_lyrs_pca_discount_age, m_lyrs_discount_age,
                      m_qalys_healthy_discount_age, m_qalys_pca_discount_age, m_qalys_discount_age,
                      m_cost_psa_testing_discount_age, m_cost_biopsy_discount_age, m_cost_staging_discount_age,
                      m_cost_tx_discount_age, m_cost_eol_discount_age, m_cost_discount_age]

        age_columns = ['age', 'pca_cases', 'screen-detected cases',
                       'post-screening cases', 'overdiagnosis', 'deaths_other', 'deaths_pca',
                       'pca_alive', 'healthy','lyrs_healthy_nodiscount', 'lyrs_healthy_discount',
                       'lyrs_pca_discount', 'total_lyrs_discount',
                       'qalys_healthy_discount', 'qalys_pca_discount', 'total_qalys_discount',
                       'cost_psa_testing_discount', 'cost_biopsy_discount', 'cost_staging_discount',
                       'cost_treatment_discount', 'costs_eol_discount', 'total_cost_discount']

        age_cohort = pd.DataFrame(age_matrix, index = age_columns).T

        t_parameters_age = [year, t_cases_age, t_overdiagnosis_age,
                            t_deaths_pca_age, t_deaths_other_age,
                            t_lyrs_healthy_discount_age, t_lyrs_pca_discount_age,
                            t_lyrs_nodiscount_age, t_lyrs_discount_age, t_qalys_healthy_discount_age,
                            t_qalys_pca_discount_age, t_qalys_nodiscount_age, t_qalys_discount_age,
                            t_cost_psa_testing_discount_age, t_cost_psa_testing_discount_age,
                            t_cost_biopsy_nodiscount_age, t_cost_biopsy_discount_age,
                            t_cost_staging_nodiscount_age, t_cost_staging_discount_age,
                            t_cost_tx_nodiscount_age, t_cost_tx_discount_age,
                            t_cost_eol_nodiscount_age, t_cost_eol_discount_age,
                            t_cost_nodiscount_age, t_cost_discount_age,
                            total_n_psa_tests_age, total_n_biopsies_age]

        columns_age = ['cohort_age_at_start', 'pca_cases', 'overdiagnosis',
                       'pca_deaths', 'deaths_other_causes',
                       'lyrs_healthy_discounted', 'lyrs_pca_discounted',
                       'lyrs_undiscounted', 'lyrs_discounted','qalys_healthy_discounted',
                       'qalys_pca_discounted', 'qalys_undiscounted', 'qalys_discounted',
                       'cost_psa_testing_undiscounted', 'cost_psa_testing_discounted',
                       'cost_biopsy_undiscounted', 'cost_biopsy_discounted',
                       'cost_staging_undiscounted', 'cost_staging_discounted',
                       'cost_treatment_undiscounted', 'cost_treatment_discounted',
                       'cost_eol_undiscounted', 'cost_eol_discounted',
                       'costs_undiscounted', 'costs_discounted', 'n_psa_tests', 'n_biopsies']

        outcomes_age_psa = pd.DataFrame(t_parameters_age, index = columns_age).T

        s_qalys_discount_age_df = pd.DataFrame(s_qalys_discount_age)
        s_cost_discount_age_df = pd.DataFrame(s_cost_discount_age)

        parameters_age = [s_qalys_discount_age, s_cost_discount_age,
                          s_deaths_pca_age, s_overdiagnosis_age,
                          age_cohort, outcomes_age_psa]

        for index, parameter in enumerate(parameter_list_age):
            parameter = gen_list_outcomes(parameter_list_age[index], parameters_age[index])

                                #################################################
                                # Polygenic risk tailored screening from age 55 #
                                #################################################

        # Yearly probability of PCa incidence
        smoothed_pca_incidence_prs = psa_function(pca_incidence)
        smoothed_pca_incidence_prs[:,10:25] = (smoothed_pca_incidence_prs[:,10:25].T * rr_incidence[year-45,:]).T
        smoothed_pca_incidence_prs[:,25:35] = smoothed_pca_incidence_prs[:,25:35] * np.linspace(post_sc_incidence_drop,1,10)
        smoothed_pca_incidence_prs = smoothed_pca_incidence_prs[:,year-45:]

        # Yearly probability of death from PCa - smoothed entry and exit
        smoothed_pca_mortality_prs = psa_function(pca_death_baseline)
        smoothed_pca_mortality_prs[:,10:15] = smoothed_pca_mortality_prs[:,10:15] * np.linspace(1,0.79,5)
        smoothed_pca_mortality_prs[:,15:] = smoothed_pca_mortality_prs[:,15:] * rr_death_screening[:,15:]
        smoothed_pca_mortality_prs = smoothed_pca_mortality_prs[:,year-45:]

        # Probability of being screened
        p_screened = np.array(uptake_prs * a_risk.loc[year,'p_above_threshold'])
        p_ns = np.array((1-uptake_prs) * a_risk.loc[year,'p_above_threshold'])
        p_nos = np.array(compliance * (1-a_risk.loc[year,'p_above_threshold']))
        p_nos_screened = np.array((1-compliance) * (1-a_risk.loc[year,'p_above_threshold']))

        if year < 55:
            # Yearly probability of PCa incidence
            p_pca_screened = tr_incidence
            p_pca_ns = tr_incidence
            p_pca_nos = tr_incidence
            p_pca_nos_screened = tr_incidence

            # Yearly probability of death from PCa
            p_pca_death_screened = tr_pca_death_baseline
            p_pca_death_ns = tr_pca_death_baseline
            p_pca_death_nos = tr_pca_death_baseline
            p_pca_death_nos_screened = tr_pca_death_baseline

            # Proportion of cancers detected by screening at a localised / advanced stage
            psa_stage_adv_sc = psa_function(stage_adv[year-45:])
            psa_stage_adv_ns = psa_function(stage_adv[year-45:])
            psa_stage_adv_nos_sc = psa_function(stage_adv[year-45:])
            psa_stage_adv_nos = psa_function(stage_adv[year-45:])

            psa_stage_local_sc = psa_function(stage_local[year-45:])
            psa_stage_local_ns = psa_function(stage_local[year-45:])
            psa_stage_local_nos_sc = psa_function(stage_local[year-45:])
            psa_stage_local_nos = psa_function(stage_local[year-45:])

        elif year > 54:
            # Yearly probability of PCa incidence
            p_pca_screened = smoothed_pca_incidence_prs * a_risk.loc[year, 'rr_high']
            p_pca_ns = tr_incidence * a_risk.loc[year,'rr_high']
            p_pca_nos = tr_incidence * a_risk.loc[year,'rr_low']
            p_pca_nos_screened = smoothed_pca_incidence_prs * a_risk.loc[year,'rr_low']

            # Yearly probability of death from PCa
            p_pca_death_screened = smoothed_pca_mortality_prs * a_risk.loc[year,'rr_high']
            p_pca_death_ns = tr_pca_death_baseline * a_risk.loc[year,'rr_high']
            p_pca_death_nos = tr_pca_death_baseline * a_risk.loc[year,'rr_low']
            p_pca_death_nos_screened = smoothed_pca_mortality_prs * a_risk.loc[year,'rr_low']

            # Proportion of cancers detected by screening at a localised / advanced stage
            stage_screened_adv_sc = (stage_adv
                                     * rr_adv_screening
                                     * a_risk.loc[year, 'rr_high'])

            psa_stage_adv_sc = stage_screened_adv_sc[:,year-45:]

            stage_clinical_adv_ns = stage_adv * a_risk.loc[year, 'rr_high']
            psa_stage_adv_ns = psa_function(stage_clinical_adv_ns[year-45:])

            stage_screened_adv_nos_sc = (stage_adv
                                         * rr_adv_screening
                                         * a_risk.loc[year, 'rr_low'])

            psa_stage_adv_nos_sc = stage_screened_adv_nos_sc[:,year-45:]

            stage_clinical_adv_nos = stage_adv * a_risk.loc[year, 'rr_low']
            psa_stage_adv_nos = psa_function(stage_clinical_adv_nos[year-45:])

            stage_screened_local_sc = 1-stage_screened_adv_sc
            psa_stage_local_sc = stage_screened_local_sc[:,year-45:]

            stage_clinical_local_ns = 1-stage_clinical_adv_ns
            psa_stage_local_ns = psa_function(stage_clinical_local_ns[year-45:])

            stage_screened_local_nos_sc = 1-stage_screened_adv_nos_sc
            psa_stage_local_nos_sc = stage_screened_local_nos_sc[:, year-45:]

            stage_clinical_local_nos = 1-stage_clinical_adv_nos
            psa_stage_local_nos = psa_function(stage_clinical_local_nos[year-45:])

        #####################
        # Year 1 in the model
        #####################
        age = np.arange(year,90)
        length_df = len(age)
        length_screen = len(np.arange(year,70)) # number of screening years depending on age cohort starting

        # Cohorts, numbers 'healthy', and incident cases
        cohort_sc = np.array([np.repeat(pop[year], length_df)] * sims) * p_screened
        cohort_ns = np.array([np.repeat(pop[year], length_df)] * sims) * p_ns
        cohort_nos = np.array([np.repeat(pop[year], length_df)] * sims) * p_nos
        cohort_nos_sc = np.array([np.repeat(pop[year], length_df)] * sims) * p_nos_screened

        pca_alive_sc = np.array([np.zeros(length_df)] * sims)
        pca_alive_ns = np.array([np.zeros(length_df)] * sims)
        pca_alive_nos = np.array([np.zeros(length_df)] * sims)
        pca_alive_nos_sc = np.array([np.zeros(length_df)] * sims)

        healthy_sc = cohort_sc - pca_alive_sc
        healthy_ns = cohort_ns - pca_alive_ns
        healthy_nos = cohort_nos - pca_alive_nos
        healthy_nos_sc = cohort_nos_sc - pca_alive_nos_sc

        pca_incidence_sc = healthy_sc * p_pca_screened
        pca_incidence_nos_sc = healthy_nos_sc * p_pca_nos_screened

        if year > 54:
            pca_incidence_screened = pca_incidence_sc.copy()  # Screen-detected cancers
            pca_incidence_post_screening = np.array([np.zeros(length_df)] * sims)  # Post-screening cancers - 0 until model reaches age 70.

            pca_incidence_nos_sc_screened = pca_incidence_nos_sc.copy()  # Screen-detected cancers
            pca_incidence_nos_sc_post_screening = np.array([np.zeros(length_df)] * sims)  # Post-screening cancers - 0 until model reaches age 70.

        elif year < 55:
            # Zero as no screening in any of these cohorts
            pca_incidence_screened = np.array([np.zeros(length_df)] * sims)
            pca_incidence_post_screening = np.array([np.zeros(length_df)] * sims)

            pca_incidence_nos_sc_screened = np.array([np.zeros(length_df)] * sims)
            pca_incidence_nos_sc_post_screening = np.array([np.zeros(length_df)] * sims)

        pca_incidence_ns = healthy_ns * p_pca_ns
        pca_incidence_nos = healthy_nos * p_pca_nos

        # Deaths
        pca_death_sc = ((pca_alive_sc * p_pca_death_screened)
                        + (healthy_sc * p_pca_death_screened))

        pca_death_ns = ((pca_alive_ns * p_pca_death_ns)
                        + (healthy_ns * p_pca_death_ns))

        pca_death_nos = ((pca_alive_nos * p_pca_death_nos)
                         + (healthy_nos * p_pca_death_nos))

        pca_death_nos_sc = ((pca_alive_nos_sc * p_pca_death_nos_screened)
                            + (healthy_nos_sc * p_pca_death_nos_screened))

        pca_death_other_sc = ((pca_incidence_sc
                              + pca_alive_sc
                              - pca_death_sc)
                             * tr_death_other_causes)

        pca_death_other_ns = ((pca_incidence_ns
                              + pca_alive_ns
                              - pca_death_ns)
                             * tr_death_other_causes)

        pca_death_other_nos = ((pca_incidence_nos
                                + pca_alive_nos
                                - pca_death_nos)
                               * tr_death_other_causes)

        pca_death_other_nos_sc  = ((pca_incidence_nos_sc
                                   + pca_alive_nos_sc
                                   - pca_death_nos_sc)
                                  * tr_death_other_causes)

        healthy_death_other_sc = ((healthy_sc - pca_incidence_sc)
                                  * tr_death_other_causes)

        healthy_death_other_ns = ((healthy_ns - pca_incidence_ns)
                                  * tr_death_other_causes)

        healthy_death_other_nos = ((healthy_nos - pca_incidence_nos)
                                   * tr_death_other_causes)

        healthy_death_other_nos_sc = ((healthy_nos_sc - pca_incidence_nos_sc)
                                      * tr_death_other_causes)

        total_death_sc = (pca_death_sc
                          + pca_death_other_sc
                          + healthy_death_other_sc)

        total_death_ns = (pca_death_ns
                          + pca_death_other_ns
                          + healthy_death_other_ns)

        total_death_nos = (pca_death_nos
                           + pca_death_other_nos
                           + healthy_death_other_nos)

        total_death_nos_sc = (pca_death_nos_sc
                              + pca_death_other_nos_sc
                              + healthy_death_other_nos_sc)

        total_death = (total_death_sc
                       + total_death_ns
                       + total_death_nos
                       + total_death_nos_sc)

        # Prevalent cases & life-years
        pca_prevalence_sc = (pca_incidence_sc
                             - pca_death_sc
                             - pca_death_other_sc)

        pca_prevalence_ns = (pca_incidence_ns
                             - pca_death_ns
                             - pca_death_other_ns)

        pca_prevalence_nos = (pca_incidence_nos
                              - pca_death_nos
                              - pca_death_other_nos)

        pca_prevalence_nos_sc = (pca_incidence_nos_sc
                                 - pca_death_nos_sc
                                 - pca_death_other_nos_sc)

        lyrs_pca_sc_nodiscount = pca_prevalence_sc * 0.5
        lyrs_pca_ns_nodiscount = pca_prevalence_ns * 0.5
        lyrs_pca_nos_nodiscount = pca_prevalence_nos * 0.5
        lyrs_pca_nos_sc_nodiscount = pca_prevalence_nos_sc * 0.5

        # Costs
        if year > 54:
            costs_tx_sc = np.array([np.zeros(length_df)] * sims)
            costs_tx_screened = np.array([np.zeros(length_df)] * sims)
            costs_tx_post_screening = np.array([np.zeros(length_df)] * sims)

            costs_tx_screened[:,0] = ((pca_incidence_screened[:,0]
                                       * psa_stage_local_sc[:,0].T
                                       * tx_costs_local.T).sum(axis=0)

                                      + (pca_incidence_screened[:,0]
                                        * psa_stage_adv_sc[:,0].T
                                        * tx_costs_adv.T).sum(axis=0)) # cost of screen-detected cancers

            costs_tx_post_screening[:,0] = ((pca_incidence_post_screening[:,0]
                                             * psa_stage_local_ns[:,0].T
                                             * tx_costs_local.T).sum(axis=0)

                                            + (pca_incidence_post_screening[:,0]
                                               * psa_stage_adv_ns[:,0].T
                                               * tx_costs_adv.T).sum(axis=0)

                                            * relative_cost_clinically_detected[:,0]) # cost of post-screening cancers

            costs_tx_sc[:,0] = (costs_tx_screened[:,0] + costs_tx_post_screening[:,0]) # total cost in screened arms

            costs_tx_nos_sc = np.array([np.zeros(length_df)] * sims)
            costs_tx_nos_sc_screened = np.array([np.zeros(length_df)] * sims)
            costs_tx_nos_sc_post_screening = np.array([np.zeros(length_df)] * sims)

            costs_tx_nos_sc_screened[:,0] = ((pca_incidence_nos_sc_screened[:,0]
                                              * psa_stage_local_nos_sc[:,0].T
                                              * tx_costs_local.T).sum(axis=0)

                                             + (pca_incidence_nos_sc_screened[:,0]
                                                * psa_stage_adv_nos_sc[:,0].T
                                                * tx_costs_adv.T).sum(axis=0)) # cost of screen-detected cancers

            costs_tx_nos_sc_post_screening[:,0] = ((pca_incidence_nos_sc_post_screening[:,0]
                                                    * psa_stage_local_nos[:,0].T
                                                    * tx_costs_local.T).sum(axis=0)

                                                   + (pca_incidence_nos_sc_post_screening[:,0]
                                                      * psa_stage_adv_nos[:,0].T
                                                      * tx_costs_adv.T).sum(axis=0)

                                                   * relative_cost_clinically_detected[:,0]) # cost of post-screening cancers

            costs_tx_nos_sc[:,0] = (costs_tx_nos_sc_screened[:,0] + costs_tx_nos_sc_post_screening[:,0]) # total cost in screened arms

        elif year < 55:
            costs_tx_sc = np.array([np.zeros(length_df)] * sims)
            costs_tx_sc[:,0] = ((pca_incidence_sc[:,0]
                                 * psa_stage_local_sc[:,0].T
                                 * tx_costs_local.T).sum(axis=0)

                                + (pca_incidence_sc[:,0]
                                   * psa_stage_adv_sc[:,0].T
                                   * tx_costs_adv.T).sum(axis=0)

                               * relative_cost_clinically_detected[:,0])

            costs_tx_nos_sc = np.array([np.zeros(length_df)] * sims)
            costs_tx_nos_sc[:,0] = ((pca_incidence_nos_sc[:,0]
                                     * psa_stage_local_nos_sc[:,0].T
                                     * tx_costs_local.T).sum(axis=0)

                                    + (pca_incidence_nos_sc[:,0]
                                     * psa_stage_adv_nos_sc[:,0].T
                                     * tx_costs_adv.T).sum(axis=0)

                                   * relative_cost_clinically_detected[:,0])

        costs_tx_ns = np.array([np.zeros(length_df)] * sims)
        costs_tx_ns[:,0] = ((pca_incidence_ns[:,0]
                             * psa_stage_local_ns[:,0].T
                             * tx_costs_local.T).sum(axis=0)

                            + (pca_incidence_ns[:,0]
                             * psa_stage_adv_ns[:,0].T
                             * tx_costs_adv.T).sum(axis=0)

                           * relative_cost_clinically_detected[:,0])

        costs_tx_nos = np.array([np.zeros(length_df)] * sims)
        costs_tx_nos[:,0] = ((pca_incidence_nos[:,0]
                              * psa_stage_local_nos[:,0].T
                              * tx_costs_local.T).sum(axis=0)

                             + (pca_incidence_nos[:,0]
                              * psa_stage_adv_nos[:,0].T
                              * tx_costs_adv.T).sum(axis=0)

                            * relative_cost_clinically_detected[:,0])

        # Year 2 onwards
        ################
        for i in range(1, total_cycles):

           # Cohorts, numbers 'healthy', incident & prevalent cases
            cohort_sc[:,i] = (cohort_sc[:,i-1] - total_death_sc[:,i-1])
            cohort_ns[:,i] = (cohort_ns[:,i-1] - total_death_ns[:,i-1])
            cohort_nos[:,i] = (cohort_nos[:,i-1] - total_death_nos[:,i-1])
            cohort_nos_sc[:,i] = (cohort_nos_sc[:,i-1] - total_death_nos_sc[:,i-1])

            pca_alive_sc[:,i] = (pca_alive_sc[:,i-1]
                                 + pca_incidence_sc[:,i-1]
                                 - pca_death_sc[:,i-1]
                                 - pca_death_other_sc[:,i-1])

            pca_alive_ns[:,i] = (pca_alive_ns[:,i-1]
                                 + pca_incidence_ns[:,i-1]
                                 - pca_death_ns[:,i-1]
                                 - pca_death_other_ns[:,i-1])

            pca_alive_nos[:,i] = (pca_alive_nos[:,i-1]
                                  + pca_incidence_nos[:,i-1]
                                  - pca_death_nos[:,i-1]
                                  - pca_death_other_nos[:,i-1])

            pca_alive_nos_sc[:,i] = (pca_alive_nos_sc[:,i-1]
                                     + pca_incidence_nos_sc[:,i-1]
                                     - pca_death_nos_sc[:,i-1]
                                     - pca_death_other_nos_sc[:,i-1])

            healthy_sc[:,i] = cohort_sc[:,i] - pca_alive_sc[:,i]
            healthy_ns[:,i] = cohort_ns[:,i] - pca_alive_ns[:,i]
            healthy_nos[:,i] = cohort_nos[:,i] - pca_alive_nos[:,i]
            healthy_nos_sc[:,i] = cohort_nos_sc[:,i] - pca_alive_nos_sc[:,i]

            pca_incidence_sc[:,i] = healthy_sc[:,i] * p_pca_screened[:,i]
            pca_incidence_nos_sc[:,i] = healthy_nos_sc[:,i] * p_pca_nos_screened[:,i]

            if year > 54:
                if i < length_screen:
                    pca_incidence_screened[:,i] = pca_incidence_sc[:,i].copy()
                    pca_incidence_post_screening[:,i] = 0

                    pca_incidence_nos_sc_screened[:,i] = pca_incidence_nos_sc[:,i].copy()
                    pca_incidence_nos_sc_post_screening[:,i] = 0

                else:
                    pca_incidence_screened[:,i] = 0
                    pca_incidence_post_screening[:,i] = pca_incidence_sc[:,i].copy()

                    pca_incidence_nos_sc_screened[:,i] = 0
                    pca_incidence_nos_sc_post_screening[:,i] = pca_incidence_nos_sc[:,i].copy()

            elif year < 55:
                pca_incidence_screened[:,i] = 0
                pca_incidence_post_screening[:,i] = 0

                pca_incidence_nos_sc_screened[:,i] = 0
                pca_incidence_nos_sc_post_screening[:,i] = 0

            pca_incidence_ns[:,i] = healthy_ns[:,i] * p_pca_ns[:,i]
            pca_incidence_nos[:,i] = healthy_nos[:,i] * p_pca_nos[:,i]

            # Deaths
            pca_death_sc[:,i] = ((pca_alive_sc[:,i] * p_pca_death_screened[:,i])
                                 + (healthy_sc[:,i] * p_pca_death_screened[:,i]))

            pca_death_ns[:,i] = ((pca_alive_ns[:,i] * p_pca_death_ns[:,i])
                                 + (healthy_ns[:,i] * p_pca_death_ns[:,i]))

            pca_death_nos[:,i] = ((pca_alive_nos[:,i] * p_pca_death_nos[:,i])
                                  + (healthy_nos[:,i] * p_pca_death_nos[:,i]))

            pca_death_nos_sc[:,i] = ((pca_alive_nos_sc[:,i] * p_pca_death_nos_screened[:,i])
                                     + (healthy_nos_sc[:,i] * p_pca_death_nos_screened[:,i]))

            pca_death_other_sc[:,i] = ((pca_incidence_sc[:,i]
                                        + pca_alive_sc[:,i]
                                        - pca_death_sc[:,i])
                                       * tr_death_other_causes[:,i])

            pca_death_other_ns[:,i] = ((pca_incidence_ns[:,i]
                                        + pca_alive_ns[:,i]
                                        - pca_death_ns[:,i])
                                       * tr_death_other_causes[:,i])

            pca_death_other_nos[:,i] = ((pca_incidence_nos[:,i]
                                         + pca_alive_nos[:,i]
                                         - pca_death_nos[:,i])
                                        * tr_death_other_causes[:,i])

            pca_death_other_nos_sc[:,i] = ((pca_incidence_nos_sc[:,i]
                                            + pca_alive_nos_sc[:,i]
                                            - pca_death_nos_sc[:,i])
                                           * tr_death_other_causes[:,i])

            healthy_death_other_sc[:,i] = ((healthy_sc[:,i] - pca_incidence_sc[:,i])
                                           * tr_death_other_causes[:,i])

            healthy_death_other_ns[:,i] = ((healthy_ns[:,i] - pca_incidence_ns[:,i])
                                           * tr_death_other_causes[:,i])

            healthy_death_other_nos[:,i] = ((healthy_nos[:,i] - pca_incidence_nos[:,i])
                                            * tr_death_other_causes[:,i])

            healthy_death_other_nos_sc[:,i] = ((healthy_nos_sc[:,i]
                                                - pca_incidence_nos_sc[:,i])
                                               * tr_death_other_causes[:,i])

            total_death_sc[:,i] = (pca_death_sc[:,i]
                                   + pca_death_other_sc[:,i]
                                   + healthy_death_other_sc[:,i])

            total_death_ns[:,i] = (pca_death_ns[:,i]
                                   + pca_death_other_ns[:,i]
                                   + healthy_death_other_ns[:,i])

            total_death_nos[:,i] = (pca_death_nos[:,i]
                                    + pca_death_other_nos[:,i]
                                    + healthy_death_other_nos[:,i])

            total_death_nos_sc[:,i] = (pca_death_nos_sc[:,i]
                                       + pca_death_other_nos_sc[:,i]
                                       + healthy_death_other_nos_sc[:,i])

            total_death[:,i] = (total_death_sc[:,i]
                                + total_death_ns[:,i]
                                + total_death_nos[:,i]
                                + total_death_nos_sc[:,i])

            # Prevalent cases & life-years
            pca_prevalence_sc[:,i] = (pca_incidence_sc[:,i]
                                      + pca_alive_sc[:,i]
                                      - pca_death_sc[:,i]
                                      - pca_death_other_sc[:,i])

            pca_prevalence_ns[:,i] = (pca_incidence_ns[:,i]
                                      + pca_alive_ns[:,i]
                                      - pca_death_ns[:,i]
                                      - pca_death_other_ns[:,i])

            pca_prevalence_nos[:,i] = (pca_incidence_nos[:,i]
                                       + pca_alive_nos[:,i]
                                       - pca_death_nos[:,i]
                                       - pca_death_other_nos[:,i])

            pca_prevalence_nos_sc[:,i] = (pca_incidence_nos_sc[:,i]
                                          + pca_alive_nos_sc[:,i]
                                          - pca_death_nos_sc[:,i]
                                          - pca_death_other_nos_sc[:,i])

            lyrs_pca_sc_nodiscount[:,i] = ((pca_prevalence_sc[:,i-1] + pca_prevalence_sc[:,i]) * 0.5)                               # This calculation is because of the life-table format of the model
            lyrs_pca_ns_nodiscount[:,i]  = ((pca_prevalence_ns[:,i-1] + pca_prevalence_ns[:,i]) * 0.5)
            lyrs_pca_nos_nodiscount[:,i]  = ((pca_prevalence_nos[:,i-1] + pca_prevalence_nos[:,i]) * 0.5)
            lyrs_pca_nos_sc_nodiscount[:,i]  = ((pca_prevalence_nos_sc[:,i-1] + pca_prevalence_nos_sc[:,i]) * 0.5)

            # Costs
            if year > 54:
                costs_tx_screened[:,i] = ((pca_incidence_screened[:,i]
                                           * psa_stage_local_sc[:,i].T
                                           * tx_costs_local.T).sum(axis=0)

                                          + (pca_incidence_screened[:,i]
                                            * psa_stage_adv_sc[:,i].T
                                            * tx_costs_adv.T).sum(axis=0)) # cost of screen-detected cancers

                costs_tx_post_screening[:,i] = ((pca_incidence_post_screening[:,i]
                                                 * psa_stage_local_ns[:,i].T
                                                 * tx_costs_local.T).sum(axis=0)

                                                + (pca_incidence_post_screening[:,i]
                                                   * psa_stage_adv_ns[:,i].T
                                                   * tx_costs_adv.T).sum(axis=0)

                                               * relative_cost_clinically_detected[:,i]) # cost of post-screening cancers

                costs_tx_sc[:,i] = (costs_tx_screened[:,i] + costs_tx_post_screening[:,i]) # total cost in screened arms

                costs_tx_nos_sc_screened[:,i] = ((pca_incidence_nos_sc_screened[:,i]
                                                 * psa_stage_local_nos_sc[:,i].T
                                                 * tx_costs_local.T).sum(axis=0)

                                                + (pca_incidence_nos_sc_screened[:,i]
                                                 * psa_stage_adv_nos_sc[:,i].T
                                                 * tx_costs_adv.T).sum(axis=0)) # cost of screen-detected cancers

                costs_tx_nos_sc_post_screening[:,i] = ((pca_incidence_nos_sc_post_screening[:,i]
                                                       * psa_stage_local_nos[:,i].T
                                                       * tx_costs_local.T).sum(axis=0)

                                                      + (pca_incidence_nos_sc_post_screening[:,i]
                                                       * psa_stage_adv_nos[:,i].T
                                                       * tx_costs_adv.T).sum(axis=0)

                                                      * relative_cost_clinically_detected[:,i]) # cost of post-screening cancers

                costs_tx_nos_sc[:,i] = (costs_tx_nos_sc_screened[:,i] + costs_tx_nos_sc_post_screening[:,i]) # total cost in screened arms

            elif year < 55:
                costs_tx_sc[:,i] = ((pca_incidence_sc[:,i]
                                     * psa_stage_local_sc[:,i].T
                                     * tx_costs_local.T).sum(axis=0)

                                    + (pca_incidence_sc[:,i]
                                       * psa_stage_adv_sc[:,i].T
                                       * tx_costs_adv.T).sum(axis=0)

                                   * relative_cost_clinically_detected[:,i])

                costs_tx_nos_sc[:,i] = ((pca_incidence_nos_sc[:,i]
                                         * psa_stage_local_nos_sc[:,i].T
                                         * tx_costs_local.T).sum(axis=0)

                                        + (pca_incidence_nos_sc[:,i]
                                         * psa_stage_adv_nos_sc[:,i].T
                                         * tx_costs_adv.T).sum(axis=0)

                                       * relative_cost_clinically_detected[:,i])

            costs_tx_ns[:,i] = ((pca_incidence_ns[:,i]
                                 * psa_stage_local_ns[:,i].T
                                 * tx_costs_local.T).sum(axis=0)

                                + (pca_incidence_ns[:,i]
                                 * psa_stage_adv_ns[:,i].T
                                 * tx_costs_adv.T).sum(axis=0)

                               * relative_cost_clinically_detected[:,i])

            costs_tx_nos[:,i] = ((pca_incidence_nos[:,i]
                                  * psa_stage_local_nos[:,i].T
                                  * tx_costs_local.T).sum(axis=0)

                                 + (pca_incidence_nos[:,i]
                                  * psa_stage_adv_nos[:,i].T
                                  * tx_costs_adv.T).sum(axis=0)

                                * relative_cost_clinically_detected[:,i])

        ############
        # Outcomes #
        ############

        # INDEX:
        # s_ = sim (this is the sum across the simulations i.e. one total value per simulation)
        # m_ = mean (this is the mean across the simulations i.e. one value for each year of the model)
        # t_ = total
        # nodiscount = not discounted
        # discount = discounted
        # _prs = outcomes for the polygenic risk-tailored screening cohort

        # Incident cases (screened arms)
        s_cases_sc_prs, m_cases_sc_prs, t_cases_sc_prs = outcomes(pca_incidence_sc)
        s_cases_nos_sc_prs, m_cases_nos_sc_prs, t_cases_nos_sc_prs = outcomes(pca_incidence_nos_sc)

        # Screen-detected cancers
        s_cases_sc_detected_prs, m_cases_sc_detected_prs, t_cases_sc_detected_prs = outcomes(pca_incidence_screened)
        s_cases_nos_sc_detected_prs, m_cases_nos_sc_detected_prs, t_cases_nos_sc_detected_prs = outcomes(pca_incidence_nos_sc_screened)

        # Cancers in the post-screening phase (amongst those who received screening)
        s_cases_post_screening_prs, m_cases_post_screening_prs, t_cases_post_screening_prs = outcomes(pca_incidence_post_screening)
        s_cases_nos_sc_post_screening_prs, m_cases_nos_sc_post_screening_prs, t_cases_nos_sc_post_screening_prs = outcomes(pca_incidence_nos_sc_post_screening)

        # Incident cases (non-screened arms)
        s_cases_ns_prs, m_cases_ns_prs, t_cases_ns_prs = outcomes(pca_incidence_ns)
        s_cases_nos_prs, m_cases_nos_prs, t_cases_nos_prs = outcomes(pca_incidence_nos)

        # Incident cases (total)
        ########################
        s_cases_prs = (s_cases_sc_prs
                       + s_cases_ns_prs
                       + s_cases_nos_prs
                       + s_cases_nos_sc_prs)

        m_cases_prs = (m_cases_sc_prs
                       + m_cases_ns_prs
                       + m_cases_nos_prs
                       + m_cases_nos_sc_prs)

        t_cases_prs = (t_cases_sc_prs
                       + t_cases_ns_prs
                       + t_cases_nos_prs
                       + t_cases_nos_sc_prs)

        # PCa alive
        s_pca_alive_prs, m_pca_alive_prs, t_pca_alive_prs = outcomes((pca_alive_sc
                                                                      + pca_alive_ns
                                                                      + pca_alive_nos
                                                                      + pca_alive_nos_sc))

        # Healthy
        s_healthy_prs, m_healthy_prs, t_healthy_prs = outcomes((healthy_sc
                                                                + healthy_ns
                                                                + healthy_nos
                                                                + healthy_nos_sc))

        # Overdiagnosed cases
        overdiagnosis_prs = pca_incidence_screened * p_overdiagnosis_psa.T[:,year-45:]

        (s_overdiagnosis_prs,
         m_overdiagnosis_prs,
         t_overdiagnosis_prs) = outcomes(overdiagnosis_prs)

        # Deaths from other causes (screened armss)
        deaths_sc_other_prs = pca_death_other_sc + healthy_death_other_sc

        (s_deaths_sc_other_prs,
         m_deaths_sc_other_prs,
         t_deaths_sc_other_prs) = outcomes(deaths_sc_other_prs)

        deaths_nos_sc_other_prs = pca_death_other_nos_sc + healthy_death_other_nos_sc

        (s_deaths_sc_other_prs,
         m_deaths_sc_other_prs,
         t_deaths_sc_other_prs) = outcomes(deaths_sc_other_prs)

        # Deaths from other causes (non-screened arms)
        deaths_ns_other_prs = pca_death_other_ns + healthy_death_other_ns

        (s_deaths_ns_other_prs,
         m_deaths_ns_other_prs,
         t_deaths_ns_other_prs) = outcomes(deaths_ns_other_prs)

        deaths_nos_other_prs = pca_death_other_nos + healthy_death_other_nos

        (s_deaths_nos_other_prs,
         m_deaths_nos_other_prs,
         t_deaths_nos_other_prs) = outcomes(deaths_nos_other_prs)

        # Total deaths from other causes
        ################################
        deaths_other_prs = (deaths_sc_other_prs
                           + deaths_ns_other_prs
                           + deaths_nos_other_prs
                           + deaths_nos_sc_other_prs)

        s_deaths_other_prs, m_deaths_other_prs, t_deaths_other_prs = outcomes(deaths_other_prs)

        # Deaths from prosate cancer (screened arms)
        s_deaths_sc_pca_prs, m_deaths_sc_pca_prs, t_deaths_sc_pca_prs = outcomes(pca_death_sc)

        (s_deaths_nos_sc_pca_prs,
         m_deaths_nos_sc_pca_prs,
         t_deaths_nos_sc_pca_prs) = outcomes(pca_death_nos_sc)

        # Deaths from prosate cancer (non-screened arms)
        s_deaths_ns_pca_prs, m_deaths_ns_pca_prs, t_deaths_ns_pca_prs = outcomes(pca_death_ns)
        s_deaths_nos_pca_prs, m_deaths_nos_pca_prs, t_deaths_nos_pca_prs = outcomes(pca_death_nos)

        # Deaths from prosate cancer (total)
        ####################################
        deaths_pca_prs = (pca_death_sc
                         + pca_death_ns
                         + pca_death_nos
                         + pca_death_nos_sc)

        s_deaths_pca_prs, m_deaths_pca_prs, t_deaths_pca_prs = outcomes(deaths_pca_prs)

        # Healthy life-years (screened arm)
        lyrs_healthy_sc_nodiscount_prs = (healthy_sc
                                          - (0.5 * (healthy_death_other_sc + pca_incidence_sc)))

        lyrs_healthy_sc_discount_prs = lyrs_healthy_sc_nodiscount_prs * discount_factor[:total_cycles]

        (s_lyrs_healthy_sc_discount_prs,
         m_lyrs_healthy_sc_discount_prs,
         t_lyrs_healthy_sc_discount_prs) = outcomes(lyrs_healthy_sc_discount_prs)

        lyrs_healthy_nos_sc_nodiscount_prs = (healthy_nos_sc
                                              - (0.5 * (healthy_death_other_nos_sc + pca_incidence_nos_sc)))

        lyrs_healthy_nos_sc_discount_prs = lyrs_healthy_nos_sc_nodiscount_prs * discount_factor[:total_cycles]

        (s_lyrs_healthy_nos_sc_discount_prs,
         m_lyrs_healthy_nos_sc_discount_prs,
         t_lyrs_healthy_nos_sc_discount_prs) = outcomes(lyrs_healthy_nos_sc_discount_prs)

        # Healthy life-years (non-screened arm)
        lyrs_healthy_ns_nodiscount_prs = (healthy_ns -
                                          (0.5 * (healthy_death_other_ns + pca_incidence_ns)))

        lyrs_healthy_ns_discount_prs = lyrs_healthy_ns_nodiscount_prs * discount_factor[:total_cycles]

        (s_lyrs_healthy_ns_discount_prs,
         m_lyrs_healthy_ns_discount_prs,
         t_lyrs_healthy_ns_discount_prs) = outcomes(lyrs_healthy_ns_discount_prs)

        lyrs_healthy_nos_nodiscount_prs = (healthy_nos
                                           - (0.5 * (healthy_death_other_nos + pca_incidence_nos)))

        lyrs_healthy_nos_discount_prs = lyrs_healthy_nos_nodiscount_prs * discount_factor[:total_cycles]

        (s_lyrs_healthy_nos_discount_prs,
         m_lyrs_healthy_nos_discount_prs,
         t_lyrs_healthy_nos_discount_prs) = outcomes(lyrs_healthy_nos_discount_prs)

        # Total healthy life-years
        lyrs_healthy_nodiscount_prs = (lyrs_healthy_sc_nodiscount_prs
                                       + lyrs_healthy_ns_nodiscount_prs
                                       + lyrs_healthy_nos_nodiscount_prs
                                       + lyrs_healthy_nos_sc_nodiscount_prs)

        (s_lyrs_healthy_nodiscount_prs,
         m_lyrs_healthy_nodiscount_prs,
         t_lyrs_healthy_nodiscount_prs) = outcomes(lyrs_healthy_nodiscount_prs)

        lyrs_healthy_discount_prs = (lyrs_healthy_sc_discount_prs
                                     + lyrs_healthy_ns_discount_prs
                                     + lyrs_healthy_nos_discount_prs
                                     + lyrs_healthy_nos_sc_discount_prs)

        (s_lyrs_healthy_discount_prs,
         m_lyrs_healthy_discount_prs,
         t_lyrs_healthy_discount_prs) = outcomes(lyrs_healthy_discount_prs)

        # Life-years with prostate cancer in screened arms
        lyrs_pca_sc_discount = lyrs_pca_sc_nodiscount * discount_factor[:total_cycles]

        (s_lyrs_pca_sc_discount_prs,
         m_lyrs_pca_sc_discount_prs,
         t_lyrs_pca_sc_discount_prs) = outcomes(lyrs_pca_sc_discount)

        lyrs_pca_nos_sc_discount = lyrs_pca_nos_sc_nodiscount * discount_factor[:total_cycles]

        (s_lyrs_pca_nos_sc_discount_prs,
         m_lyrs_pca_nos_sc_discount_prs,
         t_lyrs_pca_nos_sc_discount_prs) = outcomes(lyrs_pca_nos_sc_discount)

        # Life-years with prostate cancer in non-screened arms
        lyrs_pca_ns_discount = lyrs_pca_ns_nodiscount * discount_factor[:total_cycles]

        (s_lyrs_pca_ns_discount_prs,
         m_lyrs_pca_ns_discount_prs,
         t_lyrs_pca_ns_discount_prs) = outcomes(lyrs_pca_ns_discount)

        lyrs_pca_nos_discount = lyrs_pca_nos_nodiscount * discount_factor[:total_cycles]

        (s_lyrs_pca_nos_discount_prs,
         m_lyrs_pca_nos_discount_prs,
         t_lyrs_pca_nos_discount_prs) = outcomes(lyrs_pca_nos_discount)

        #  Life-years with prostate cancer in both arms
        lyrs_pca_nodiscount_prs = (lyrs_pca_sc_nodiscount
                                   + lyrs_pca_ns_nodiscount
                                   + lyrs_pca_nos_nodiscount
                                   + lyrs_pca_nos_sc_nodiscount)

        lyrs_pca_discount_prs = (lyrs_pca_sc_discount
                                 + lyrs_pca_ns_discount
                                 + lyrs_pca_nos_discount
                                 + lyrs_pca_nos_sc_discount)

        (s_lyrs_pca_discount_prs,
         m_lyrs_pca_discount_prs,
         t_lyrs_pca_discount_prs) = outcomes(lyrs_pca_discount_prs)

        # Total Life-years
        ##################
        lyrs_nodiscount_prs = lyrs_healthy_nodiscount_prs + lyrs_pca_nodiscount_prs

        (s_lyrs_nodiscount_prs,
         m_lyrs_nodiscount_prs,
         t_lyrs_nodiscount_prs) = outcomes(lyrs_nodiscount_prs)

        lyrs_discount_prs = lyrs_healthy_discount_prs + lyrs_pca_discount_prs

        (s_lyrs_discount_prs,
         m_lyrs_discount_prs,
         t_lyrs_discount_prs) = outcomes(lyrs_discount_prs)

        # QALYs (healthy life) - screened arms
        qalys_healthy_sc_nodiscount_prs = lyrs_healthy_sc_nodiscount_prs * utility_background_psa[:,year-45:]
        qalys_healthy_sc_discount_prs = lyrs_healthy_sc_discount_prs * utility_background_psa[:,year-45:]

        (s_qalys_healthy_sc_discount_prs,
         m_qalys_healthy_sc_discount_prs,
         t_qalys_healthy_sc_discount_prs) = outcomes(qalys_healthy_sc_discount_prs)

        qalys_healthy_nos_sc_nodiscount_prs = lyrs_healthy_nos_sc_nodiscount_prs * utility_background_psa[:,year-45:]
        qalys_healthy_nos_sc_discount_prs = lyrs_healthy_nos_sc_discount_prs * utility_background_psa[:,year-45:]

        (s_qalys_healthy_nos_sc_discount_prs,
         m_qalys_healthy_nos_sc_discount_prs,
         t_qalys_healthy_nos_sc_discount_prs) = outcomes(qalys_healthy_nos_sc_discount_prs)

        # QALYs (healthy life) - non-screened arms
        qalys_healthy_ns_nodiscount_prs = lyrs_healthy_ns_nodiscount_prs * utility_background_psa[:,year-45:]
        qalys_healthy_ns_discount_prs = lyrs_healthy_ns_discount_prs * utility_background_psa[:,year-45:]

        (s_qalys_healthy_ns_discount_prs,
         m_qalys_healthy_ns_discount_prs,
         t_qalys_healthy_ns_discount_prs) = outcomes(qalys_healthy_ns_discount_prs)

        qalys_healthy_nos_nodiscount_prs = lyrs_healthy_nos_nodiscount_prs * utility_background_psa[:,year-45:]
        qalys_healthy_nos_discount_prs = lyrs_healthy_nos_discount_prs * utility_background_psa[:,year-45:]

        (s_qalys_healthy_nos_discount_prs,
         m_qalys_healthy_nos_discount_prs,
         t_qalys_healthy_nos_discount_prs) = outcomes(qalys_healthy_nos_discount_prs)

        # Total QALYs (healthy life)
        qalys_healthy_nodiscount_prs = lyrs_healthy_nodiscount_prs * utility_background_psa[:,year-45:]
        qalys_healthy_discount_prs = lyrs_healthy_discount_prs * utility_background_psa[:,year-45:]

        (s_qalys_healthy_discount_prs,
         m_qalys_healthy_discount_prs,
         t_qalys_healthy_discount_prs) = outcomes(qalys_healthy_discount_prs)

        # QALYS with prostate cancer - screened arms
        qalys_pca_sc_nodiscount_prs = lyrs_pca_sc_nodiscount * pca_incidence_utility_psa[:,year-45:]
        qalys_pca_sc_discount_prs = lyrs_pca_sc_discount * pca_incidence_utility_psa[:,year-45:]

        (s_qalys_pca_sc_discount_prs,
         m_qalys_pca_sc_discount_prs,
         t_qalys_pca_sc_discount_prs) = outcomes(qalys_pca_sc_discount_prs)

        qalys_pca_nos_sc_nodiscount_prs = lyrs_pca_nos_sc_nodiscount * pca_incidence_utility_psa[:,year-45:]
        qalys_pca_nos_sc_discount_prs = lyrs_pca_nos_sc_discount * pca_incidence_utility_psa[:,year-45:]

        (s_qalys_pca_nos_sc_discount_prs,
         m_qalys_pca_nos_sc_discount_prs,
         t_qalys_pca_nos_sc_discount_prs) = outcomes(qalys_pca_nos_sc_discount_prs)

        # QALYS with prostate cancer - non-screened arms
        qalys_pca_ns_nodiscount_prs = lyrs_pca_ns_nodiscount * pca_incidence_utility_psa[:,year-45:]
        qalys_pca_ns_discount_prs = lyrs_pca_ns_discount * pca_incidence_utility_psa[:,year-45:]

        (s_qalys_pca_ns_discount_prs,
         m_qalys_pca_ns_discount_prs,
         t_qalys_pca_ns_discount_prs) = outcomes(qalys_pca_ns_discount_prs)

        qalys_pca_nos_nodiscount_prs = lyrs_pca_nos_nodiscount * pca_incidence_utility_psa[:,year-45:]
        qalys_pca_nos_discount_prs = lyrs_pca_nos_discount * pca_incidence_utility_psa[:,year-45:]

        (s_qalys_pca_nos_discount_prs,
         m_qalys_pca_nos_discount_prs,
         t_qalys_pca_nos_discount_prs) = outcomes(qalys_pca_nos_discount_prs)

        # Total QALYS with prostate cancer
        qalys_pca_nodiscount_prs = lyrs_pca_nodiscount_prs * pca_incidence_utility_psa[:,year-45:]
        qalys_pca_discount_prs = lyrs_pca_discount_prs * pca_incidence_utility_psa[:,year-45:]

        (s_qalys_pca_discount_prs,
         m_qalys_pca_discount_prs,
         t_qalys_pca_discount_prs) = outcomes(qalys_pca_discount_prs)

        # Total QALYs
        #############
        qalys_nodiscount_prs = qalys_healthy_nodiscount_prs + qalys_pca_nodiscount_prs

        (s_qalys_nodiscount_prs,
         m_qalys_nodiscount_prs,
         t_qalys_nodiscount_prs) = outcomes(qalys_nodiscount_prs)

        qalys_discount_prs = qalys_healthy_discount_prs + qalys_pca_discount_prs

        (s_qalys_discount_prs,
         m_qalys_discount_prs,
         t_qalys_discount_prs) = outcomes(qalys_discount_prs)

        # Costs of risk-stratification
        cost_screening_prs = cost_prs * uptake_prs * pop[year] # There is no discounting of risk-stratification as done at year 1 of the model.

        # Costs of PSA testing in non-screened arms
        n_psa_tests_ns_prs = (((pca_incidence_ns / p_suspected_ns[:,year-45:])
                              + ((pca_incidence_ns * (1-uptake_biopsy[year-45:]))
                                 / p_suspected_refuse_biopsy_ns[:,year-45:]))
                             * uptake_psa
                             * n_psa_tests[:,year-45:])

        cost_psa_testing_ns_nodiscount_prs = (n_psa_tests_ns_prs
                                              * cost_psa[:,year-45:]
                                              * relative_cost_clinically_detected[:,year-45:])

        (s_cost_psa_testing_ns_nodiscount_prs,
         m_cost_psa_testing_ns_nodiscount_prs,
         t_cost_psa_testing_ns_nodiscount_prs) = outcomes(cost_psa_testing_ns_nodiscount_prs)

        cost_psa_testing_ns_discount_prs = cost_psa_testing_ns_nodiscount_prs * discount_factor[:total_cycles]

        (s_cost_psa_testing_ns_discount_prs,
         m_cost_psa_testing_ns_discount_prs,
         t_cost_psa_testing_ns_discount_prs) = outcomes(cost_psa_testing_ns_discount_prs)

        n_psa_tests_nos_prs = (((pca_incidence_nos / p_suspected_ns[:,year-45:])
                               + ((pca_incidence_nos * (1-uptake_biopsy[year-45:]))
                                  / p_suspected_refuse_biopsy_ns[:,year-45:]))
                               * uptake_psa
                               * n_psa_tests[:,year-45:])

        cost_psa_testing_nos_nodiscount_prs = (n_psa_tests_nos_prs
                                               * cost_psa[:,year-45:]
                                               * relative_cost_clinically_detected[:,year-45:])

        (s_cost_psa_testing_nos_nodiscount_prs,
         m_cost_psa_testing_nos_nodiscount_prs,
         t_cost_psa_testing_nos_nodiscount_prs) = outcomes(cost_psa_testing_nos_nodiscount_prs)

        cost_psa_testing_nos_discount_prs = cost_psa_testing_nos_nodiscount_prs * discount_factor[:total_cycles]

        (s_cost_psa_testing_nos_discount_prs,
         m_cost_psa_testing_nos_discount_prs,
         t_cost_psa_testing_nos_discount_prs) = outcomes(cost_psa_testing_nos_discount_prs)

        # Costs of PSA testing in screened arms
        if year > 54:

            # Get the screened years
            lyrs_healthy_screened_nodiscount_prs = np.array([np.zeros(length_df)] * sims)
            lyrs_healthy_screened_nodiscount_prs[:,:length_screen] = lyrs_healthy_sc_nodiscount_prs[:,:length_screen].copy()
            lyrs_healthy_screened_nodiscount_prs[:,length_screen:] = 0

            # Population-level PSA testing during screening phase
            n_psa_tests_screened_prs = lyrs_healthy_screened_nodiscount_prs * uptake_psa / 4

            # Assuming all cancers are clinically detected in the post-screening phase
            n_psa_tests_post_screening_prs = (((pca_incidence_post_screening / p_suspected_ns[:,year-45:])
                                              + ((pca_incidence_post_screening * (1-uptake_biopsy[year-45:]))
                                                 / p_suspected_refuse_biopsy_ns[:,year-45:]))
                                              * uptake_psa
                                              * n_psa_tests[:,year-45:])

            n_psa_tests_sc_prs = (n_psa_tests_screened_prs + n_psa_tests_post_screening_prs)

            cost_psa_testing_sc_nodiscount_prs = ((n_psa_tests_screened_prs * cost_psa[:,year-45:])
                                                  + (n_psa_tests_post_screening_prs
                                                     * cost_psa[:,year-45:]
                                                     * relative_cost_clinically_detected[:,year-45:]))

            # PSA tests in the not offered screening but screened anyway group
            # Get the screened years
            lyrs_healthy_nos_sc_screened_nodiscount_prs = np.array([np.zeros(length_df)] * sims)
            lyrs_healthy_nos_sc_screened_nodiscount_prs[:,:length_screen] = lyrs_healthy_nos_sc_nodiscount_prs[:,:length_screen].copy()
            lyrs_healthy_nos_sc_screened_nodiscount_prs[:,length_screen:] = 0

            # Population-level PSA testing during screening phase
            n_psa_tests_nos_sc_screened_prs = lyrs_healthy_nos_sc_screened_nodiscount_prs * uptake_psa / 4

            # Assuming all cancers are clinically detected in the post-screening phase
            n_psa_tests_nos_sc_post_screening_prs = (((pca_incidence_nos_sc_post_screening / p_suspected_ns[:,year-45:])
                                                     + ((pca_incidence_nos_sc_post_screening * (1-uptake_biopsy[year-45:]))
                                                        / p_suspected_refuse_biopsy_ns[:,year-45:]))
                                                    * uptake_psa
                                                    * n_psa_tests[:,year-45:])

            n_psa_tests_nos_sc_prs = (n_psa_tests_nos_sc_screened_prs
                                     + n_psa_tests_nos_sc_post_screening_prs)

            cost_psa_testing_nos_sc_nodiscount_prs = ((n_psa_tests_nos_sc_screened_prs * cost_psa[:,year-45:])
                                                      + (n_psa_tests_nos_sc_post_screening_prs
                                                         * cost_psa[:,year-45:]
                                                         * relative_cost_clinically_detected[:,year-45:]))

        elif year < 55:
            n_psa_tests_sc_prs = (((pca_incidence_sc / p_suspected_ns[:,year-45:])
                                  + ((pca_incidence_sc * (1-uptake_biopsy[year-45:]))
                                     / p_suspected_refuse_biopsy_ns[:,year-45:]))
                                  * uptake_psa
                                  * n_psa_tests[:,year-45:])

            n_psa_tests_nos_sc_prs = ((pca_incidence_nos_sc / p_suspected_ns[:,year-45:])
                                      + ((pca_incidence_nos_sc * (1-uptake_biopsy[year-45:]))
                                         / p_suspected_refuse_biopsy_ns[:,year-45:])
                                     * uptake_psa
                                     * n_psa_tests[:,year-45:])

            cost_psa_testing_sc_nodiscount_prs = (n_psa_tests_sc_prs
                                                  * cost_psa[:,year-45:]
                                                  * relative_cost_clinically_detected[:,year-45:])

            cost_psa_testing_nos_sc_nodiscount_prs = (n_psa_tests_nos_sc_prs
                                                      * cost_psa[:,year-45:]
                                                      * relative_cost_clinically_detected[:,year-45:])

        (s_cost_psa_testing_sc_nodiscount_prs,
         m_cost_psa_testing_sc_nodiscount_prs,
         t_cost_psa_testing_sc_nodiscount_prs) = outcomes(cost_psa_testing_sc_nodiscount_prs)

        cost_psa_testing_sc_discount_prs = cost_psa_testing_sc_nodiscount_prs * discount_factor[:total_cycles]

        (s_cost_psa_testing_sc_discount_prs,
         m_cost_psa_testing_sc_discount_prs,
         t_cost_psa_testing_sc_discount_prs) = outcomes(cost_psa_testing_sc_discount_prs)

        (s_cost_psa_testing_nos_sc_nodiscount_prs,
         m_cost_psa_testing_nos_sc_nodiscount_prs,
         t_cost_psa_testing_nos_sc_nodiscount_prs) = outcomes(cost_psa_testing_nos_sc_nodiscount_prs)

        cost_psa_testing_nos_sc_discount_prs = (cost_psa_testing_nos_sc_nodiscount_prs
                                                * discount_factor[:total_cycles])

        (s_cost_psa_testing_nos_sc_discount_prs,
         m_cost_psa_testing_nos_sc_discount_prs,
         t_cost_psa_testing_nos_sc_discount_prs) = outcomes(cost_psa_testing_nos_sc_discount_prs)

        # Total costs of PSA testing
        ############################
        n_psa_tests_prs = (n_psa_tests_sc_prs
                           + n_psa_tests_ns_prs
                           + n_psa_tests_nos_prs
                           + n_psa_tests_nos_sc_prs)

        (s_n_psa_tests_prs,
         m_n_psa_tests_prs,
         total_n_psa_tests_prs) = outcomes(n_psa_tests_prs)

        cost_psa_testing_nodiscount_prs = (cost_psa_testing_sc_nodiscount_prs
                                           + cost_psa_testing_ns_nodiscount_prs
                                           + cost_psa_testing_nos_nodiscount_prs
                                           + cost_psa_testing_nos_sc_nodiscount_prs)

        (s_cost_psa_testing_nodiscount_prs,
         m_cost_psa_testing_nodiscount_prs,
         t_cost_psa_testing_nodiscount_prs) = outcomes(cost_psa_testing_nodiscount_prs)

        cost_psa_testing_discount_prs = (cost_psa_testing_sc_discount_prs
                                         + cost_psa_testing_ns_discount_prs
                                         + cost_psa_testing_nos_discount_prs
                                         + cost_psa_testing_nos_sc_discount_prs)

        (s_cost_psa_testing_discount_prs,
         m_cost_psa_testing_discount_prs,
         t_cost_psa_testing_discount_prs) = outcomes(cost_psa_testing_discount_prs)

        # Costs of biopsy - screened arms
        if year > 54:
            # Screen-detected cancers
            n_biopsies_screened_prs = pca_incidence_screened / p_suspected[:,year-45:]

            cost_biopsy_screened_nodiscount_prs = (((pca_incidence_screened / p_suspected[:,year-45:])
                                                    * cost_biopsy[:,year-45:])

                                                   + (((pca_incidence_screened * (1-uptake_biopsy[year-45:]))
                                                       / p_suspected_refuse_biopsy[:,year-45:])
                                                      * cost_refuse_biopsy[:,year-45:]))

            # Assuming all cancers are clinically detected in the post-screening phase
            n_biopsies_post_screening_prs = pca_incidence_post_screening / p_suspected_ns[:,year-45:]

            cost_biopsies_post_screening_nodiscount_prs = (((pca_incidence_post_screening / p_suspected_ns[:,year-45:])
                                                            * cost_biopsy[:,year-45:])

                                                           + (((pca_incidence_post_screening * (1-uptake_biopsy[year-45:]))
                                                               / p_suspected_refuse_biopsy_ns[:,year-45:])
                                                              * cost_refuse_biopsy[:,year-45:])

                                                           * relative_cost_clinically_detected[:,year-45:])

            n_biopsies_sc_prs = (n_biopsies_screened_prs + n_biopsies_post_screening_prs)

            # Total cost of biopsies
            cost_biopsy_sc_nodiscount_prs = (cost_biopsy_screened_nodiscount_prs
                                            + cost_biopsies_post_screening_nodiscount_prs)

            n_biopsies_nos_sc_screened_prs = pca_incidence_nos_sc_screened / p_suspected[:,year-45:]

            cost_biopsy_nos_sc_screened_nodiscount_prs = (((pca_incidence_nos_sc_screened / p_suspected[:,year-45:])
                                                            * cost_biopsy[:,year-45:])

                                                           + (((pca_incidence_nos_sc_screened * (1-uptake_biopsy[year-45:]))
                                                               / p_suspected_refuse_biopsy[:,year-45:])
                                                              * cost_refuse_biopsy[:,year-45:]))

            # Assuming all cancers are clinically detected in the post-screening phase
            n_biopsies_nos_sc_post_screening_prs = pca_incidence_nos_sc_post_screening / p_suspected_ns[:,year-45:]

            cost_biopsies_nos_sc_post_screening_nodiscount_prs = (((pca_incidence_nos_sc_post_screening / p_suspected_ns[:,year-45:])
                                                                    * cost_biopsy[:,year-45:])

                                                                   + (((pca_incidence_nos_sc_post_screening * (1-uptake_biopsy[year-45:]))
                                                                       / p_suspected_refuse_biopsy_ns[:,year-45:])
                                                                      * cost_refuse_biopsy[:,year-45:])

                                                                   * relative_cost_clinically_detected[:,year-45:])

            # Total biopsies
            n_biopsies_nos_sc_prs = (n_biopsies_nos_sc_screened_prs
                                    + n_biopsies_nos_sc_post_screening_prs)

            # Total cost of biopsies
            cost_biopsy_nos_sc_nodiscount_prs = (cost_biopsy_nos_sc_screened_nodiscount_prs
                                                + cost_biopsies_nos_sc_post_screening_nodiscount_prs)

        elif year < 55:
            n_biopsies_sc_prs = pca_incidence_sc / p_suspected_ns[:,year-45:]

            cost_biopsy_sc_nodiscount_prs = (((pca_incidence_sc / p_suspected_ns[:,year-45:])
                                              * cost_biopsy[:,year-45:])

                                               + (((pca_incidence_sc * (1-uptake_biopsy[year-45:]))
                                                    / p_suspected_refuse_biopsy_ns[:,year-45:])
                                                   * cost_refuse_biopsy[:,year-45:])

                                            * relative_cost_clinically_detected[:,year-45:])

            n_biopsies_nos_sc_prs = pca_incidence_nos_sc / p_suspected_ns[:,year-45:]

            cost_biopsy_nos_sc_nodiscount_prs = (((pca_incidence_nos_sc / p_suspected_ns[:,year-45:])
                                                  * cost_biopsy[:,year-45:])

                                                 + (((pca_incidence_nos_sc * (1-uptake_biopsy[year-45:]))
                                                      / p_suspected_refuse_biopsy_ns[:,year-45:])
                                                     * cost_refuse_biopsy[:,year-45:])

                                                * relative_cost_clinically_detected[:,year-45:])

        (s_cost_biopsy_sc_nodiscount_prs,
         m_cost_biopsy_sc_nodiscount_prs,
         t_cost_biopsy_sc_nodiscount_prs) = outcomes(cost_biopsy_sc_nodiscount_prs)

        cost_biopsy_sc_discount_prs = cost_biopsy_sc_nodiscount_prs * discount_factor[:total_cycles]

        (s_cost_biopsy_sc_discount_prs,
         m_cost_biopsy_sc_discount_prs,
         t_cost_biopsy_sc_discount_prs) = outcomes(cost_biopsy_sc_discount_prs)

        (s_cost_biopsy_nos_sc_nodiscount_prs,
         m_cost_biopsy_nos_sc_nodiscount_prs,
         t_cost_biopsy_nos_sc_nodiscount_prs) = outcomes(cost_biopsy_nos_sc_nodiscount_prs)

        cost_biopsy_nos_sc_discount_prs = cost_biopsy_nos_sc_nodiscount_prs * discount_factor[:total_cycles]

        (s_cost_biopsy_nos_sc_discount_prs,
         m_cost_biopsy_nos_sc_discount_prs,
         t_cost_biopsy_nos_sc_discount_prs) = outcomes(cost_biopsy_nos_sc_discount_prs)

        # Costs of biopsy - non-screened arms
        n_biopsies_ns_prs = pca_incidence_ns / p_suspected_ns[:,year-45:]

        cost_biopsy_ns_nodiscount_prs = (((pca_incidence_ns / p_suspected_ns[:,year-45:])
                                          * cost_biopsy[:,year-45:])

                                           + (((pca_incidence_ns * (1-uptake_biopsy[year-45:]))
                                                / p_suspected_refuse_biopsy_ns[:,year-45:])
                                               * cost_refuse_biopsy[:,year-45:])

                                        * relative_cost_clinically_detected[:,year-45:])

        (s_cost_biopsy_ns_nodiscount_prs,
         m_cost_biopsy_ns_nodiscount_prs,
         t_cost_biopsy_ns_nodiscount_prs) = outcomes(cost_biopsy_ns_nodiscount_prs)

        cost_biopsy_ns_discount_prs = cost_biopsy_ns_nodiscount_prs * discount_factor[:total_cycles]

        (s_cost_biopsy_ns_discount_prs,
         m_cost_biopsy_ns_discount_prs,
         t_cost_biopsy_ns_discount_prs) = outcomes(cost_biopsy_ns_discount_prs)

        n_biopsies_nos_prs = pca_incidence_nos / p_suspected_ns[:,year-45:]

        cost_biopsy_nos_nodiscount_prs = (((pca_incidence_nos / p_suspected_ns[:,year-45:])
                                          * cost_biopsy[:,year-45:])

                                           + (((pca_incidence_nos * (1-uptake_biopsy[year-45:]))
                                                / p_suspected_refuse_biopsy_ns[:,year-45:])
                                               * cost_refuse_biopsy[:,year-45:])

                                         * relative_cost_clinically_detected[:,year-45:])

        (s_cost_biopsy_nos_nodiscount_prs,
         m_cost_biopsy_nos_nodiscount_prs,
         t_cost_biopsy_nos_nodiscount_prs) = outcomes(cost_biopsy_nos_nodiscount_prs)

        cost_biopsy_nos_discount_prs = cost_biopsy_nos_nodiscount_prs * discount_factor[:total_cycles]

        (s_cost_biopsy_nos_discount_prs,
         m_cost_biopsy_nos_discount_prs,
         t_cost_biopsy_nos_discount_prs) = outcomes(cost_biopsy_nos_discount_prs)

        # Total costs of biopsy
        #######################
        n_biopsies_prs = (n_biopsies_sc_prs
                          + n_biopsies_ns_prs
                          + n_biopsies_nos_prs
                          + n_biopsies_nos_sc_prs)

        (s_n_biopsies_prs,
         m_n_biopsies_prs,
         total_n_biopsies_prs) = outcomes(n_biopsies_prs)

        cost_biopsy_nodiscount_prs = (cost_biopsy_sc_nodiscount_prs
                                      + cost_biopsy_ns_nodiscount_prs
                                      + cost_biopsy_nos_nodiscount_prs
                                      + cost_biopsy_nos_sc_nodiscount_prs)

        (s_cost_biopsy_nodiscount_prs,
         m_cost_biopsy_nodiscount_prs,
         t_cost_biopsy_nodiscount_prs) = outcomes(cost_biopsy_nodiscount_prs)

        cost_biopsy_discount_prs = (cost_biopsy_sc_discount_prs
                                    + cost_biopsy_ns_discount_prs
                                    + cost_biopsy_nos_discount_prs
                                    + cost_biopsy_nos_sc_discount_prs)

        (s_cost_biopsy_discount_prs,
         m_cost_biopsy_discount_prs,
         t_cost_biopsy_discount_prs) = outcomes(cost_biopsy_discount_prs)

        # Cost of staging in the screened arms
        if year > 54:
            cost_staging_screened_nodiscount_prs = (cost_assessment
                                                    * psa_stage_adv_sc.T
                                                    * pca_incidence_screened.T).T

            cost_staging_post_screening_nodiscount_prs = (cost_assessment
                                                          * psa_stage_adv_ns.T
                                                          * pca_incidence_post_screening.T
                                                          * relative_cost_clinically_detected[:,year-45:].T).T

            cost_staging_sc_nodiscount_prs = (cost_staging_screened_nodiscount_prs
                                             + cost_staging_post_screening_nodiscount_prs)

            cost_staging_nos_sc_screened_nodiscount_prs = (cost_assessment
                                                          * psa_stage_adv_nos_sc.T
                                                          * pca_incidence_nos_sc_screened.T).T

            cost_staging_nos_sc_post_screening_nodiscount_prs = (cost_assessment
                                                                * psa_stage_adv_nos.T
                                                                * pca_incidence_nos_sc_post_screening.T
                                                                * relative_cost_clinically_detected[:,year-45:].T).T

            cost_staging_nos_sc_nodiscount_prs = (cost_staging_nos_sc_screened_nodiscount_prs
                                                 + cost_staging_nos_sc_post_screening_nodiscount_prs)

        if year < 55:
            cost_staging_sc_nodiscount_prs = (cost_assessment
                                              * psa_stage_adv_sc.T
                                              * pca_incidence_sc.T
                                              * relative_cost_clinically_detected[:,year-45:].T).T

            cost_staging_nos_sc_nodiscount_prs = (cost_assessment
                                                  * psa_stage_adv_nos_sc.T
                                                  * pca_incidence_nos_sc.T
                                                  * relative_cost_clinically_detected[:,year-45:].T).T

        (s_cost_staging_sc_nodiscount_prs,
         m_cost_staging_sc_nodiscount_prs,
         t_cost_staging_sc_nodiscount_prs) = outcomes(cost_staging_sc_nodiscount_prs)

        cost_staging_sc_discount_prs = cost_staging_sc_nodiscount_prs * discount_factor[:total_cycles]

        (s_cost_staging_sc_discount_prs,
         m_cost_staging_sc_discount_prs,
         t_cost_staging_sc_discount_prs) = outcomes(cost_staging_sc_discount_prs)

        (s_cost_staging_nos_sc_nodiscount_prs,
         m_cost_staging_nos_sc_nodiscount_prs,
         t_cost_staging_nos_sc_nodiscount_prs) = outcomes(cost_staging_nos_sc_nodiscount_prs)

        cost_staging_nos_sc_discount_prs = cost_staging_nos_sc_nodiscount_prs * discount_factor[:total_cycles]

        (s_cost_staging_nos_sc_discount_prs,
         m_cost_staging_nos_sc_discount_prs,
         t_cost_staging_nos_sc_discount_prs) = outcomes(cost_staging_nos_sc_discount_prs)

        # Cost of staging in the non-screened arms
        cost_staging_ns_nodiscount_prs = (cost_assessment
                                          * psa_stage_adv_ns.T
                                          * pca_incidence_ns.T
                                          * relative_cost_clinically_detected[:,year-45:].T).T

        (s_cost_staging_ns_nodiscount_prs,
         m_cost_staging_ns_nodiscount_prs,
         t_cost_staging_ns_nodiscount_prs) = outcomes(cost_staging_ns_nodiscount_prs)

        cost_staging_ns_discount_prs = cost_staging_ns_nodiscount_prs * discount_factor[:total_cycles]

        (s_cost_staging_ns_discount_prs,
         m_cost_staging_ns_discount_prs,
         t_cost_staging_ns_discount_prs) = outcomes(cost_staging_ns_discount_prs)

        cost_staging_nos_nodiscount_prs = (cost_assessment
                                           * psa_stage_adv_nos.T
                                           * pca_incidence_nos.T
                                           * relative_cost_clinically_detected[:,year-45:].T).T

        (s_cost_staging_nos_nodiscount_prs,
         m_cost_staging_nos_nodiscount_prs,
         t_cost_staging_nos_nodiscount_prs) = outcomes(cost_staging_nos_nodiscount_prs)

        cost_staging_nos_discount_prs = cost_staging_nos_nodiscount_prs * discount_factor[:total_cycles]

        (s_cost_staging_nos_discount_prs,
         m_cost_staging_nos_discount_prs,
         t_cost_staging_nos_discount_prs) = outcomes(cost_staging_nos_discount_prs)

        # Total costs of staging
        ########################
        cost_staging_nodiscount_prs = (cost_staging_sc_nodiscount_prs
                                       + cost_staging_ns_nodiscount_prs
                                       + cost_staging_nos_nodiscount_prs
                                       + cost_staging_nos_sc_nodiscount_prs)

        (s_cost_staging_nodiscount_prs,
         m_cost_staging_nodiscount_prs,
         t_cost_staging_nodiscount_prs) = outcomes(cost_staging_nodiscount_prs)

        cost_staging_discount_prs = (cost_staging_sc_discount_prs
                                     + cost_staging_ns_discount_prs
                                     + cost_staging_nos_discount_prs
                                     + cost_staging_nos_sc_discount_prs)

        (s_cost_staging_discount_prs,
         m_cost_staging_discount_prs,
         t_cost_staging_discount_prs) = outcomes(cost_staging_discount_prs)

        # Cost of treatment in screened arms
        (s_cost_tx_sc_nodiscount_prs,
         m_cost_tx_sc_nodiscount_prs,
         t_cost_tx_sc_nodiscount_prs) = outcomes(costs_tx_sc)

        cost_tx_sc_nodiscount_prs = costs_tx_sc * discount_factor[:total_cycles]

        (s_cost_tx_sc_discount_prs,
         m_cost_tx_sc_discount_prs,
         t_cost_tx_sc_discount_prs) = outcomes(cost_tx_sc_nodiscount_prs)

        (s_cost_tx_nos_sc_nodiscount_prs,
         m_cost_tx_nos_sc_nodiscount_prs,
         t_cost_tx_nos_sc_nodiscount_prs) = outcomes(costs_tx_nos_sc)

        cost_tx_nos_sc_nodiscount_prs = costs_tx_nos_sc * discount_factor[:total_cycles]

        (s_cost_tx_nos_sc_discount_prs,
         m_cost_tx_nos_sc_discount_prs,
         t_cost_tx_nos_sc_discount_prs) = outcomes(cost_tx_nos_sc_nodiscount_prs)

        # Cost of treatment in non-screened arms
        (s_cost_tx_ns_nodiscount_prs,
         m_cost_tx_ns_nodiscount_prs,
         t_cost_tx_ns_nodiscount_prs) = outcomes(costs_tx_ns)

        cost_tx_ns_nodiscount_prs = costs_tx_ns * discount_factor[:total_cycles]

        (s_cost_tx_ns_discount_prs,
         m_cost_tx_ns_discount_prs,
         t_cost_tx_ns_discount_prs) = outcomes(cost_tx_ns_nodiscount_prs)

        (s_cost_tx_nos_nodiscount_prs,
         m_cost_tx_nos_nodiscount_prs,
         t_cost_tx_nos_nodiscount_prs) = outcomes(costs_tx_nos)

        cost_tx_nos_nodiscount_prs = costs_tx_nos * discount_factor[:total_cycles]

        (s_cost_tx_nos_discount_prs,
         m_cost_tx_nos_discount_prs,
         t_cost_tx_nos_discount_prs) = outcomes(cost_tx_nos_nodiscount_prs)

        # Total costs of treatment
        ##########################
        cost_tx_nodiscount_prs = (costs_tx_sc
                                  + costs_tx_ns
                                  + costs_tx_nos
                                  + costs_tx_nos_sc)

        (s_cost_tx_nodiscount_prs,
         m_cost_tx_nodiscount_prs,
         t_cost_tx_nodiscount_prs) = outcomes(cost_tx_nodiscount_prs)

        cost_tx_discount_prs = cost_tx_nodiscount_prs * discount_factor[:total_cycles]

        (s_cost_tx_discount_prs,
         m_cost_tx_discount_prs,
         t_cost_tx_discount_prs) = outcomes(cost_tx_discount_prs)

        # Costs of palliation and death in screened arm
        cost_eol_sc_nodiscount_prs = (pca_death_costs * pca_death_sc.T).T

        (s_cost_eol_sc_nodiscount_prs,
         m_cost_eol_sc_nodiscount_prs,
         t_cost_eol_sc_nodiscount_prs) = outcomes(cost_eol_sc_nodiscount_prs)

        cost_eol_sc_discount_prs = cost_eol_sc_nodiscount_prs * discount_factor[:total_cycles]

        (s_cost_eol_sc_discount_prs,
         m_cost_eol_sc_discount_prs,
         t_cost_eol_sc_discount_prs) = outcomes(cost_eol_sc_discount_prs)

        cost_eol_nos_sc_nodiscount_prs = (pca_death_costs * pca_death_nos_sc.T).T

        (s_cost_eol_nos_sc_nodiscount_prs,
         m_cost_eol_nos_sc_nodiscount_prs,
         t_cost_eol_nos_sc_nodiscount_prs) = outcomes(cost_eol_nos_sc_nodiscount_prs)

        cost_eol_nos_sc_discount_prs = cost_eol_nos_sc_nodiscount_prs * discount_factor[:total_cycles]

        (s_cost_eol_nos_sc_discount_prs,
         m_cost_eol_nos_sc_discount_prs,
         t_cost_eol_nos_sc_discount_prs) = outcomes(cost_eol_nos_sc_discount_prs)

        # Costs of palliation and death in non-screened arm
        cost_eol_ns_nodiscount_prs = (pca_death_costs * pca_death_ns.T).T

        (s_cost_eol_ns_nodiscount_prs,
         m_cost_eol_ns_nodiscount_prs,
         t_cost_eol_ns_nodiscount_prs) = outcomes(cost_eol_ns_nodiscount_prs)

        cost_eol_ns_discount_prs = cost_eol_ns_nodiscount_prs * discount_factor[:total_cycles]

        (s_cost_eol_ns_discount_prs,
         m_cost_eol_ns_discount_prs,
         t_cost_eol_ns_discount_prs) = outcomes(cost_eol_ns_discount_prs)

        cost_eol_nos_nodiscount_prs = (pca_death_costs * pca_death_nos.T).T

        (s_cost_eol_nos_nodiscount_prs,
         m_cost_eol_nos_nodiscount_prs,
         t_cost_eol_nos_nodiscount_prs) = outcomes(cost_eol_nos_nodiscount_prs)

        cost_eol_nos_discount_prs = cost_eol_nos_nodiscount_prs * discount_factor[:total_cycles]

        (s_cost_eol_nos_discount_prs,
         m_cost_eol_nos_discount_prs,
         t_cost_eol_nos_discount_prs) = outcomes(cost_eol_nos_discount_prs)

        # Total costs of palliation and death
        cost_eol_nodiscount_prs = (cost_eol_sc_nodiscount_prs
                                   + cost_eol_ns_nodiscount_prs
                                   + cost_eol_nos_nodiscount_prs
                                   + cost_eol_nos_sc_nodiscount_prs)

        (s_cost_eol_nodiscount_prs,
         m_cost_eol_nodiscount_prs,
         t_cost_eol_nodiscount_prs) = outcomes(cost_eol_nodiscount_prs)

        cost_eol_discount_prs = (cost_eol_sc_discount_prs
                                 + cost_eol_ns_discount_prs
                                 + cost_eol_nos_discount_prs
                                 + cost_eol_nos_sc_discount_prs)

        (s_cost_eol_discount_prs,
         m_cost_eol_discount_prs,
         t_cost_eol_discount_prs) = outcomes(cost_eol_discount_prs)

        # TOTAL COSTS PRS-BASED SCREENING
        #################################
        cost_nodiscount_prs = (cost_psa_testing_nodiscount_prs
                               + cost_biopsy_nodiscount_prs
                               + cost_staging_nodiscount_prs
                               + cost_tx_nodiscount_prs
                               + cost_eol_nodiscount_prs)

        s_cost_nodiscount_prs, m_cost_nodiscount_prs, t_cost_nodiscount_prs = outcomes(cost_nodiscount_prs)
        s_cost_nodiscount_prs = s_cost_nodiscount_prs + np.mean(cost_screening_prs)
        t_cost_nodiscount_prs = t_cost_nodiscount_prs + np.mean(cost_screening_prs)

        cost_discount_prs = (cost_psa_testing_discount_prs
                             + cost_biopsy_discount_prs
                             + cost_staging_discount_prs
                             + cost_tx_discount_prs
                             + cost_eol_discount_prs)

        s_cost_discount_prs, m_cost_discount_prs, t_cost_discount_prs = outcomes(cost_discount_prs)
        s_cost_discount_prs = s_cost_discount_prs + np.mean(cost_screening_prs)
        t_cost_discount_prs = t_cost_discount_prs + np.mean(cost_screening_prs)

        # Generate a mean dataframe
        prs_matrix = [age, m_cases_prs, m_cases_sc_detected_prs,
                      m_cases_post_screening_prs, m_overdiagnosis_prs, m_deaths_other_prs, m_deaths_pca_prs,
                      m_pca_alive_prs, m_healthy_prs, m_lyrs_healthy_nodiscount_prs,
                      m_lyrs_healthy_discount_prs, m_lyrs_pca_discount_prs, m_lyrs_discount_prs,
                      m_qalys_healthy_discount_prs, m_qalys_pca_discount_prs, m_qalys_discount_prs,
                      m_cost_psa_testing_discount_prs, m_cost_biopsy_discount_prs, m_cost_staging_discount_prs,
                      m_cost_tx_discount_prs, m_cost_eol_discount_prs, m_cost_discount_prs]

        prs_columns = ['age', 'pca_cases', 'screen-detected cases',
                       'post-screening cases', 'overdiagnosis', 'deaths_other', 'deaths_pca',
                       'pca_alive', 'healthy', 'lyrs_healthy_nodiscount', 'lyrs_healthy_discount',
                       'lyrs_pca_discount', 'total_lyrs_discount',
                       'qalys_healthy_discount', 'qalys_pca_discount', 'total_qalys_discount',
                       'cost_psa_testing_discount', 'cost_biopsy_discount', 'cost_staging_discount',
                       'cost_treatment_discount', 'costs_eol_discount', 'total_cost_discount']

        prs_cohort = pd.DataFrame(prs_matrix, index = prs_columns).T

        t_parameters_prs = [year, t_cases_prs, t_overdiagnosis_prs,
                            t_deaths_pca_prs, t_deaths_other_prs,
                            t_lyrs_healthy_discount_prs, t_lyrs_pca_discount_prs,
                            t_lyrs_nodiscount_prs, t_lyrs_discount_prs, t_qalys_healthy_discount_prs,
                            t_qalys_pca_discount_prs, t_qalys_nodiscount_prs, t_qalys_discount_prs,
                            np.mean(cost_screening_prs), t_cost_psa_testing_nodiscount_prs,
                            t_cost_psa_testing_discount_prs, t_cost_biopsy_nodiscount_prs,
                            t_cost_biopsy_discount_prs, t_cost_staging_nodiscount_prs,
                            t_cost_staging_discount_prs, t_cost_tx_nodiscount_prs,
                            t_cost_tx_discount_prs, t_cost_eol_nodiscount_prs,
                            t_cost_eol_discount_prs, t_cost_nodiscount_prs, t_cost_discount_prs,
                            total_n_psa_tests_prs, total_n_biopsies_prs]

        columns_prs = ['cohort_age_at_start', 'pca_cases', 'overdiagnosis',
                       'pca_deaths', 'deaths_other_causes',
                       'lyrs_healthy_discounted', 'lyrs_pca_discounted',
                       'lyrs_undiscounted', 'lyrs_discounted','qalys_healthy_discounted',
                       'qalys_pca_discounted', 'qalys_undiscounted', 'qalys_discounted',
                       'cost_screening', 'cost_psa_testing_undiscounted', 'cost_psa_testing_discounted',
                       'cost_biopsy_undiscounted', 'cost_biopsy_discounted',
                       'cost_staging_undiscounted', 'cost_staging_discounted',
                       'cost_treatment_undiscounted','cost_treatment_discounted',
                       'cost_eol_undiscounted', 'cost_eol_discounted', 'costs_undiscounted', 'costs_discounted',
                       'n_psa_tests', 'n_biopsies']

        outcomes_prs_psa = pd.DataFrame(t_parameters_prs, index = columns_prs).T

        parameters_prs = [s_qalys_discount_prs, s_cost_discount_prs,
                          s_deaths_pca_prs, s_overdiagnosis_prs,
                          prs_cohort, outcomes_prs_psa]

        for index, parameter in enumerate(parameter_list_prs):
            parameter = gen_list_outcomes(parameter_list_prs[index], parameters_prs[index])

    #######################
    # SAVE THE DATAFRAMES #
    #######################

    # Set path to store outputs of models
    path = base_path+(str(np.round(reference_value*100,2)))+"/"

    # write the dataframes to an excel file - one sheet for each cohort
    def save_excel(list_dataframes, name):
        writer = pd.ExcelWriter(name+'.xlsx', engine='xlsxwriter')
        for i, df in enumerate(list_dataframes):
            df.to_excel(writer,'cohort_%s' % (i+45))
        writer.save()

    save_excel(ns_cohort_list, path+'non_screening_cohorts_psa')
    save_excel(age_cohort_list, path+'age_screening_cohorts_psa')
    save_excel(prs_cohort_list, path+'prs_screening_cohorts_psa')

    # Save the collated outcome dataframes
    outcomes_ns_psa = pd.concat(outcomes_ns_psa_list)
    outcomes_age_psa = pd.concat(outcomes_age_psa_list)
    outcomes_prs_psa = pd.concat(outcomes_prs_psa_list)

    outcomes_dfs = [outcomes_ns_psa, outcomes_age_psa, outcomes_prs_psa]
    outcomes_dfs_names = ['outcomes_ns_psa', 'outcomes_age_psa', 'outcomes_prs_psa']

    for i, df in enumerate(outcomes_dfs):
        writer = pd.ExcelWriter(path+outcomes_dfs_names[i]+'_'+(str(np.round(reference_value*100,2)))+'.xlsx', engine='xlsxwriter')
        df.to_excel(writer, sheet_name=outcomes_dfs_names[i], index=False)
        workbook  = writer.book
        worksheet = writer.sheets[outcomes_dfs_names[i]]
        format1 = workbook.add_format({'num_format': '#,##0', 'align': 'center'})
        worksheet.set_column('A:AH', 18, format1)
        writer.save()

    # Set path to store outputs of models
    path = (base_path+(str(np.round(reference_value*100,2)))+"/simulation dataframes/")
    os.makedirs(path, exist_ok=True)

    # Non-screening cohorts
    s_qalys_discount_ns_df = pd.DataFrame(s_qalys_discount_ns_list)
    s_cost_discount_ns_df = pd.DataFrame(s_cost_discount_ns_list)
    s_pca_deaths_ns_df = pd.DataFrame(s_pca_deaths_ns_list)

    # Age-based screening cohorts
    s_qalys_discount_age_df = pd.DataFrame(s_qalys_discount_age_list)
    s_cost_discount_age_df = pd.DataFrame(s_cost_discount_age_list)
    s_pca_deaths_age_df = pd.DataFrame(s_pca_deaths_age_list)
    s_overdiagnosis_age_df = pd.DataFrame(s_overdiagnosis_age_list)

    # Precision screening from age 55 cohorts
    s_qalys_discount_prs_df = pd.DataFrame(s_qalys_discount_prs_list)
    s_cost_discount_prs_df = pd.DataFrame(s_cost_discount_prs_list)
    s_pca_deaths_prs_df = pd.DataFrame(s_pca_deaths_prs_list)
    s_overdiagnosis_prs_df = pd.DataFrame(s_overdiagnosis_prs_list)

    dataframes = [s_qalys_discount_ns_df, s_cost_discount_ns_df, s_pca_deaths_ns_df,
                  s_qalys_discount_age_df, s_cost_discount_age_df, s_pca_deaths_age_df, s_overdiagnosis_age_df,
                  s_qalys_discount_prs_df, s_cost_discount_prs_df, s_pca_deaths_prs_df, s_overdiagnosis_prs_df]

    df_names = ['qalys_ns', 'costs_ns', 'pca_deaths_ns',
                'qalys_age', 'costs_age', 'pca_deaths_age', 'overdiagnosis_age',
                'qalys_prs', 'costs_prs', 'pca_deaths_prs', 'overdiagnosis_prs']

    for i, df in enumerate(dataframes):
        df.to_csv(path+df_names[i]+'_'+(str(np.round(reference_value*100,2)))+'.csv', index=False)
