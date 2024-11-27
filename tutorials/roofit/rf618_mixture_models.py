## \file
## \ingroup tutorial_roofit
## \notebook
## Use of mixture models in RooFit.
##
## This tutorial shows, how to use mixture models for Likelihood Calculation in ROOT. Instead of directly
## calculating the likelihood we use simulation based inference (SBI) as shown in tutorial 'rf615_simulation_based_inference.py'.
## We train the classifier to discriminate between samples from an background hypothesis here the zz samples and a target
## hypothesis, here the higgs samples. The data preparation is based on the tutorial 'df106_HiggsToFourLeptons.py'.
##
## An introduction to mixture models can be found here https://arxiv.org/pdf/1506.02169.
##
## A short summary:
## We assume the whole probability distribution can be written as a mixture of several components, i.e.
## $$p(x|\theta)= \sum_{c}w_{c}(\theta)p_{c}(x|\theta)$$
## We can write the likelihood ratio in terms of pairwise classification problems
## \begin{align*}
##  \frac{p(x|\mu)}{p(x|0)}&= \frac{\sum_{c}w_{c}(\mu)p_{c}(x|\mu)}{\sum_{c'}w_{c'}(0)p_{c'}(x|0)}\\
##  &=\sum_{c}\Bigg[\sum_{c'}\frac{w_{c'}(0)}{w_{c}(\mu)}\frac{p_{c'}(x|0)}{p_{c}(x|\mu)}\Bigg]^{-1},
## \end{align*}
## where mu is the signal strength, and a value of 0 corresponds to the background hypothesis. Using this decomposition, one is able to use the pairwise likelihood ratios.
##
## Since the only free parameter in our case is mu, the distributions are independent of this parameter and the dependence on the signal strength can be encoded into the weights.
## Thus, the subratios simplify dramatically since they are independent of theta and these ratios can be pre-computed and the classifier does
## not need to be parametrized.
##
## If you wish to see an analysis done with template histograms see 'hf001_example.py'.
##
## \macro_image
## \macro_code
## \macro_output
##
## \date September 2024
## \author Robin Syring

import ROOT
import os
import numpy as np
import xgboost as xgb

# Get Dataframe from tutorial df106_HiggsToFourLeptons.py
# Adjust the path if running locally
df = ROOT.RDataFrame("tree", ROOT.gROOT.GetTutorialDir().Data() + "/analysis/dataframe/df106_HiggsToFourLeptons.root")

# Initialize a dictionary to store counts and weight sums for each category
results = {}


# Extract the relevant columns once and avoid repeated calls
data_dict = df.AsNumpy(columns=["m4l", "sample_category", "weight"])


weights_dict = {
    name: data_dict["weight"][data_dict["sample_category"] == [name]].sum() for name in ("data", "zz", "other", "higgs")
}

# Loop over each sample category
for sample_category in ["data", "higgs", "zz", "other"]:

    weight_sum = weights_dict[sample_category]

    mask = data_dict["sample_category"] == sample_category
    # Normalize each weight
    weights = data_dict["weight"][mask]
    # Extract the weight_modified
    weight_modified = weights / weight_sum

    count = np.sum(mask)

    # Store the count and weight sum in the dictionary
    results[sample_category] = {
        "weight_sum": weight_sum,
        "weight_modified": weight_modified,
        "count": count,
        "weight": weights,
    }


# Extract the mass for higgs and zz
higgs_data = data_dict["m4l"][data_dict["sample_category"] == ["higgs"]]
zz_data = data_dict["m4l"][data_dict["sample_category"] == ["zz"]]


# Prepare sample weights
sample_weight_higgs = np.array([results["higgs"]["weight_modified"]]).flatten()
sample_weight_zz = np.array([results["zz"]["weight_modified"]]).flatten()

# Putting sample weights together in the same manner as the training data
sample_weight = np.concatenate([sample_weight_higgs, sample_weight_zz])

# For Training purposes we have to get rid of the negative weights, since xgb can't handle them
sample_weight[sample_weight < 0] = 1e-6

# Prepare the features and labels
X = np.concatenate((higgs_data, zz_data), axis=0).reshape(-1, 1)
y = np.concatenate([np.ones(len(higgs_data)), np.zeros(len(zz_data))])

# Train the Classifier to discriminate between higgs and zz
model_xgb = xgb.XGBClassifier(n_estimators=1000, max_depth=5, eta=0.2, min_child_weight=1e-6, nthread=1)
model_xgb.fit(X, y, sample_weight=sample_weight)


# Building a RooRealVar based on the observed data
m4l = ROOT.RooRealVar("m4l", "Four Lepton Invariant Mass", 0.0)


# Define functions to compute the learned likelihood.
def calculate_likelihood_xgb(m4l_arr: np.ndarray) -> np.ndarray:
    prob = model_xgb.predict_proba(m4l_arr.T)[:, 0]
    return (1 - prob) / prob


llh = ROOT.RooFit.bindFunction(f"llh", calculate_likelihood_xgb, m4l)

# Number of signals and background
n_signal = results["higgs"]["weight"].sum()
n_back = results["zz"]["weight"].sum()


# Define weight functions
def weight_back(mu):
    return n_back / (n_back + mu * n_signal)


def weight_signal(mu):
    return 1 - weight_back(mu)


# Define the likelihood ratio accordingly to mixture models
def likelihood_ratio(llr: np.ndarray, mu: np.ndarray) -> np.ndarray:

    m = 2

    w_0 = np.array([weight_back(0), weight_signal(0)])
    w_1 = np.array([weight_back(mu[0]), weight_signal(mu[0])])

    w = np.outer(w_1, 1.0 / w_0)

    p = np.ones((m, m, len(llr)))
    p[1, 0] = llr
    for i in range(m):
        for j in range(i):
            p[j, i] = 1.0 / p[i, j]

    return 1.0 / np.sum(1.0 / np.sum(np.expand_dims(w, axis=2) * p, axis=0), axis=0)


mu_var = ROOT.RooRealVar("mu", "mu", 0.1, 5)

nll_ratio = ROOT.RooFit.bindFunction(f"nll", likelihood_ratio, llh, mu_var)
pdf_learned = ROOT.RooWrapperPdf("learned_pdf", "learned_pdf", nll_ratio, selfNormalized=True)

# Plot the likelihood
frame1 = m4l.frame(Title="Likelihood ratio r(m_{4l}|#mu=1);m_{4l};p(#mu=1)/p(#mu=0)", Range=(80, 170))
# llh.plotOn(frame1, ShiftToZero=False, LineColor="kP6Blue")
nll_ratio.plotOn(frame1, ShiftToZero=False, LineColor="kP6Blue")

n_pred = ROOT.RooFormulaVar("n_pred", f"{n_back} + mu * {n_signal}", [mu_var])
pdf_learned_extended = ROOT.RooExtendPdf("final_pdf", "final_pdf", pdf_learned, n_pred)

# Prepare the observed data set and NLL
data = ROOT.RooDataSet.from_numpy({"m4l": data_dict["m4l"][data_dict["sample_category"] == ["data"]]}, [m4l])
nll = pdf_learned_extended.createNLL(data, Extended=True)

# Plot the nll computet by the mixture model
frame2 = mu_var.frame(Title="NLL sum;#mu (signal strength);#Delta NLL", Range=(0.5, 4))
nll.plotOn(frame2, ShiftToZero=True, LineColor="kP6Blue")

# Write the plots into one canvas to show, or into separate canvases for saving.
single_canvas = True

c = ROOT.TCanvas("", "", 1200 if single_canvas else 600, 600)
if single_canvas:
    c.Divide(2)
    c.cd(1)
    ROOT.gPad.SetLeftMargin(0.15)
    frame1.GetYaxis().SetTitleOffset(1.8)
frame1.Draw()

if single_canvas:
    c.cd(2)
    ROOT.gPad.SetLeftMargin(0.15)
    frame2.GetYaxis().SetTitleOffset(1.8)
else:
    c.SaveAs("rf618_plot_1.png")
    c = ROOT.TCanvas("", "", 600, 600)

frame2.Draw()

if not single_canvas:
    c.SaveAs("rf618_plot_2.png")

# Compute the minimum via minuit and display the results
minimizer = ROOT.RooMinimizer(nll)
minimizer.setErrorLevel(0.5)  # Adjust the error level in the minimization to work with likelihoods
minimizer.setPrintLevel(-1)
minimizer.minimize("Minuit2")
result = minimizer.save()
ROOT.SetOwnership(result, True)
result.Print()

del minimizer
del nll
del pdf_learned_extended
del n_pred
del llh
del nll_ratio

import sys

# Hack to bypass ClearProxiedObjects()
del sys.modules["libROOTPythonizations"]
