# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 09:21:24 2022

@author: Leh

Bayesian ANCOVA with Condition and Repetition as factors and Experience as covariate.

Dependent: Bins (segments) or Primitives
Predictors: Condition, Repetition; Covariate: Dance Experience

Code is based on PyMc3 implementations of Chapter 19 and 20 in Kruschke (2015):
https://github.com/JWarmenhoven/DBDA-python/blob/master/Notebooks

"""

import pandas as pd
import numpy as np
import arviz as az
import pymc3 as pm
import theano.tensor as tt
pd.options.mode.chained_assignment = None

# %% Settings
xLabel = 'Experience' 
yLabel = 'Bins' # 'Bins' (Segments) or 'Primitives'
n_chains = 4
cores = 4 
tunes = 10000
draws = 20000
RANDOM_SEED = 375923
condition_labels_dict = {'N': 'Neutral', 'I': 'Improvised', 'C_F_C': 'Constrained',
      'C_F_F': 'Free', 'C_T_Q': 'Quick', 'C_T_S': 'Sustained'}

# %% Load data
df = pd.read_csv('MaxLAPs.csv')

# %% Extract indexes for different levels
df['Experience'] = df['Dance_Years']
data = df[[yLabel, 'Condition', 'Rep'] + [xLabel] + ['Subject']]
data = data.rename(columns = {yLabel: 'y'})
conditions_idx_dict = dict(zip(data['Condition'].unique(), np.arange(data['Condition'].nunique())))
data['Condition_idx'] = data['Condition'].replace(conditions_idx_dict).astype(int)

c_n = data['Condition'].nunique()
c_idx = [conditions_idx_dict[con] for con in data[['Condition', 'Rep']].drop_duplicates()['Condition']] # list indicating which rep belong to which condition

data['Con_Rep'] = list(zip(data.Condition_idx, data.Rep))
conditions_rep_dict = dict(zip(data['Con_Rep'].unique(), np.arange(data['Con_Rep'].nunique())))
c_rep_n = data['Con_Rep'].nunique()
data['Con_Rep_idx'] = [conditions_rep_dict[con] for con in data['Con_Rep']] 
data['Rep_idx'] = data['Rep'] - 1

#%% Model prep
# Calculate Gamma shape and rate from mode and sd
# (e.q 9.8 on P238 in Kruschke, 2015)
def gammaShRaFromModeSD(mode, sd):
    rate = (mode + np.sqrt( mode**2 + 4 * sd**2 ) ) / ( 2 * sd**2 )
    shape = 1 + mode * rate
    return(shape, rate)

# x1 condition predictor
x1 = data['Condition_idx'].values
Nx1Lvl = data['Condition_idx'].nunique()

# x2 rep predictor
x2 = data['Rep_idx'].values
Nx2Lvl = data['Rep_idx'].nunique()

# metric predictors
xMet1 = data[xLabel].values # experience
xMet1Mean = data[xLabel].mean()
xMet1SD = data[xLabel].std()

# Dependent
y = data.y.values
yMean = y.mean()
ySD = y.std()
agammaShRa = gammaShRaFromModeSD(ySD/2 , 2*ySD ) # broad prior on variance of deflections estimated from sd of y

# Estimate prior params
cellSDs = data[['y', 'Condition','Rep']].groupby(['Condition','Rep']).std().dropna() # sd of each rep in each condition (18 x 1)
medianCellSD = cellSDs.median().squeeze() # overall median of standard deviation of y's grouped by condition and rep
sdCellSD = cellSDs.std().squeeze() # sd of sd's of y's grouped by condition and rep
sgammaShRa = gammaShRaFromModeSD(medianCellSD, 2*sdCellSD) # compute shape and rate of gamma from median sd and sd of sds

if yLabel == 'Bins':
    sgammaShRaSD = gammaShRaFromModeSD(medianCellSD/2, 2*sdCellSD) # compute shape and rate of gamma from median sd and sd of sds
elif yLabel == 'Primitives':
    sgammaShRaSD = sgammaShRa

if (yLabel == 'Primitives') & (xLabel == 'Bins'):
    yLabel = 'Prims_Bins' # For inference regarding association between Primitives and Bins

#%% Model definition
with pm.Model() as model: # Heterogenous Variances and Robustness against Outliers

    # Baseline
    a0_tilde = pm.Normal('a0_tilde', mu=0.0, sd=1) # reparameterization
    a0 = pm.Deterministic('a0', 0.0 +ySD*5*a0_tilde) # broad gaussian prior on intercept

    # Predictor 1 (Condition)
    a1SD = pm.Gamma('a1SD', agammaShRa[0], agammaShRa[1]) # prior on variance of conditions
    a1_tilde = pm.Normal('a1_tilde', mu=0, sd=1, shape=Nx1Lvl) # reparameterization
    a1 = pm.Deterministic('a1', 0.0 + a1SD*a1_tilde) # conditions

    # Predictor 2 (Rep)
    a2SD = pm.Gamma('a2SD', agammaShRa[0], agammaShRa[1]) # prior on variance of reps
    a2_tilde = pm.Normal('a2_tilde', mu=0, sd=1, shape=Nx2Lvl) # reparameterization
    a2 = pm.Deterministic('a2', 0.0 + a2SD*a2_tilde) # reps

    # Predictor 3 (Interaction)
    a1a2SD = pm.Gamma('a1a2SD', agammaShRa[0], agammaShRa[1]) # prior on variance of interactions
    a1a2_tilde = pm.Normal('a1a2_tilde', mu=0, sd=1, shape=(Nx1Lvl, Nx2Lvl)) # reparameterization
    a1a2 = pm.Deterministic('a1a2', 0.0 + a1a2SD*a1a2_tilde) # interactions

    # Metric predictor
    aMet1 = pm.Normal('aMet1', 0, tau=1/(2*ySD/xMet1SD)**2, shape = Nx1Lvl)

    mu = a0 + a1[x1] + a2[x2] + a1a2[x1, x2] + aMet1[x1]*(xMet1 - xMet1Mean)

    sigmaMode = pm.Gamma('sigmaMode', sgammaShRa[0], sgammaShRa[1]) # modal cell standard deviation
    sigmaSD = pm.Gamma('sigmaSD', sgammaShRaSD[0], sgammaShRaSD[1]) # the standard deviation of the estimated cell standard deviations
    sigmaRa = pm.Deterministic('sigmaRa', (sigmaMode + np.sqrt(sigmaMode**2 + 4*sigmaSD**2)) / (2*sigmaSD**2)) # rate from mode and sd (e.q 9.8 on P238 in Kruschke, 2015)
    sigmaSh = pm.Deterministic('sigmaSh', 1 + (sigmaMode * sigmaRa)) # shape from mode and rate (e.q 9.8 on P238 in Kruschke, 2015)
    sigma = pm.Gamma('sigma', sigmaSh, sigmaRa, shape=(Nx1Lvl, Nx2Lvl)) # gamma on scale param of t likelihood dist (enables heterogeneous variances)
    ySigma = pm.Deterministic('ySigma', tt.maximum(sigma, medianCellSD/1000)) # prevent from dropping too close to zero

    nu = pm.Exponential('nu', 1/30.)

    like = pm.StudentT('like', nu, mu, lam=1/(ySigma[x1, x2])**2, observed=y)

#%% Sample
with model:
    idata = pm.sample(chains = n_chains, draws = draws, cores=cores, tune = tunes, target_accept=0.9999,
                      progressbar=True, random_seed=RANDOM_SEED, return_inferencedata = True)

sumDf = az.summary(idata, hdi_prob = 0.95)
