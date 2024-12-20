## \file
## \ingroup tutorial_roofit
## \notebook
## Use Simulation Based Inference (SBI) in multiple dimensions in RooFit.
##
## This tutorial shows how to use SBI in higher dimension in ROOT.
## This tutorial transfers the simple concepts of the 1D case introduced in
## rf615_simulation_based_inference.py onto the higher dimensional case.
##
## Again as reference distribution we
## choose a simple uniform distribution. The target distribution is chosen to
## be Gaussian with different mean values.
## The classifier is trained to discriminate between the reference and target
## distribution.
## We see how the neural networks generalize to unknown mean values.
##
## Furthermore, we compare SBI to the approach of moment morphing. In this case,
## we can conclude, that SBI is way more sample eficcient when it comes to
## estimating the negative log likelihood ratio.
##
## For an introductory background see rf615_simulation_based_inference.py
##
## \macro_image
## \macro_code
## \macro_output
##
## \date July 2024
## \author Robin Syring

import ROOT
import numpy as np
from sklearn.neural_network import MLPClassifier
import itertools

# Kills warning messages
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

n_samples_morph = 10000  # Number of samples for morphing
n_bins = 4  # Number of 'sampled' Gaussians
n_samples_train = n_samples_morph * n_bins  # To have a fair comparison


# Morphing as baseline
def morphing(setting, n_dimensions):
    # Define binning for morphing

    binning = [ROOT.RooBinning(n_bins, 0.0, n_bins - 1.0) for dim in range(n_dimensions)]
    grid = ROOT.RooMomentMorphFuncND.Grid(*binning)

    # Set bins for each x variable
    for x_var in x_vars:
        x_var.setBins(50)

    # Define mu values as input for morphing for each dimension
    mu_helps = [ROOT.RooRealVar(f"mu{i}", f"mu{i}", 0.0) for i in range(n_dimensions)]

    # Create a product of Gaussians for all dimensions
    gaussians = []
    for j in range(n_dimensions):
        gaussian = ROOT.RooGaussian(f"gdim{j}", f"gdim{j}", x_vars[j], mu_helps[j], sigmas[j])
        gaussians.append(gaussian)

    # Create a product PDF for the multidimensional Gaussian
    gauss_product = ROOT.RooProdPdf("gauss_product", "gauss_product", ROOT.RooArgList(*gaussians))

    templates = dict()

    # Iterate through each tuple
    for idx, nd_idx in enumerate(itertools.product(range(n_bins), repeat=n_dimensions)):
        for i_dim in range(n_dimensions):
            mu_helps[i_dim].setVal(nd_idx[i_dim])

        # Fill the histograms
        hist = gauss_product.generateBinned(ROOT.RooArgSet(*x_vars), n_samples_morph)

        # Ensure that every bin is filled and there are no zero probabilities
        for i_bin in range(hist.numEntries()):
            hist.add(hist.get(i_bin), 1.0)

        templates[nd_idx] = ROOT.RooHistPdf(f"histpdf{idx}", f"histpdf{idx}", ROOT.RooArgSet(*x_vars), hist, 1)

        # Add the PDF to the grid
        grid.addPdf(templates[nd_idx], *nd_idx)

    # Create the morphing function and add it to the ws
    morph_func = ROOT.RooMomentMorphFuncND("morph_func", "morph_func", [*mu_vars], [*x_vars], grid, setting)
    morph_func.setPdfMode(True)
    morph = ROOT.RooWrapperPdf("morph", "morph", morph_func, True)

    ws.Import(morph)

    # Uncomment to see input plots for the first dimension (you might need to increase the morphed samples)
    # f1 = x_vars[0].frame()
    # for i in range(n_bins):
    #    templates[(i, 0)].plotOn(f1)
    # ws["morph"].plotOn(f1, LineColor="r")
    # c0 = ROOT.TCanvas()
    # f1.Draw()
    # input() # Wait for user input to proceed


# Define the observed mean values for the Gaussian distributions
mu_observed = [2.5, 2.0]
sigmas = [1.5, 1.5]


# Class used in this case to demonstrate the use of SBI in Root
class SBI:
    # Initializing the class SBI
    def __init__(self, ws, n_vars):
        # Choose the hyperparameters for training the neural network
        self.classifier = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=1000, random_state=42)
        self.data_model = None
        self.data_ref = None
        self.X_train = None
        self.y_train = None
        self.ws = ws
        self.n_vars = n_vars
        self._training_mus = None
        self._reference_mu = None

    # Defining the target / training data for different values of mean value mu
    def model_data(self, model, x_vars, mu_vars, n_samples):
        ws = self.ws
        samples_gaussian = (
            ws[model].generate([ws[x] for x in x_vars] + [ws[mu] for mu in mu_vars], n_samples).to_numpy()
        )

        self._training_mus = np.array([samples_gaussian[mu] for mu in mu_vars]).T
        data_test_model = np.array([samples_gaussian[x] for x in x_vars]).T

        self.data_model = data_test_model.reshape(-1, self.n_vars)

    # Generating samples for the reference distribution
    def reference_data(self, model, x_vars, mu_vars, n_samples, help_model):
        ws = self.ws
        # Ensuring the normalization with generating as many reference data as target data
        samples_uniform = ws[model].generate([ws[x] for x in x_vars], n_samples)
        data_reference_model = np.array(
            [samples_uniform.get(i).getRealValue(x) for x in x_vars for i in range(samples_uniform.numEntries())]
        )

        self.data_ref = data_reference_model.reshape(-1, self.n_vars)

        samples_mu = ws[help_model].generate([ws[mu] for mu in mu_vars], n_samples)
        mu_data = np.array(
            [samples_mu.get(i).getRealValue(mu) for mu in mu_vars for i in range(samples_mu.numEntries())]
        )

        self._reference_mu = mu_data.reshape(-1, self.n_vars)

    # Bringing the data in the right format for training
    def preprocessing(self):
        thetas = np.concatenate((self._training_mus, self._reference_mu))
        X = np.concatenate([self.data_model, self.data_ref])

        self.y_train = np.concatenate([np.ones(len(self.data_model)), np.zeros(len(self.data_ref))])
        self.X_train = np.concatenate([X, thetas], axis=1)

    # Train the classifier
    def train_classifier(self):
        self.classifier.fit(self.X_train, self.y_train)


# Define the "observed" data in a workspace
def build_ws(mu_observed):
    n_vars = len(mu_observed)
    x_vars = [ROOT.RooRealVar(f"x{i}", f"x{i}", -5, 15) for i in range(n_vars)]
    mu_vars = [ROOT.RooRealVar(f"mu{i}", f"mu{i}", mu_observed[i], 0, 4) for i in range(n_vars)]
    gaussians = [ROOT.RooGaussian(f"gauss{i}", f"gauss{i}", x_vars[i], mu_vars[i], sigmas[i]) for i in range(n_vars)]
    uniforms = [ROOT.RooUniform(f"uniform{i}", f"uniform{i}", x_vars[i]) for i in range(n_vars)]
    uniforms_help = [ROOT.RooUniform(f"uniformh{i}", f"uniformh{i}", mu_vars[i]) for i in range(n_vars)]
    # Create multi-dimensional PDFs
    gauss = ROOT.RooProdPdf("gauss", "gauss", ROOT.RooArgList(*gaussians))
    uniform = ROOT.RooProdPdf("uniform", "uniform", ROOT.RooArgList(*uniforms))
    uniform_help = ROOT.RooProdPdf("uniform_help", "uniform_help", ROOT.RooArgList(*uniforms_help))
    obs_data = gauss.generate(ROOT.RooArgSet(*x_vars), n_samples_morph)
    obs_data.SetName("obs_data")

    # Create and return the workspace
    ws = ROOT.RooWorkspace()
    ws.Import(x_vars)
    ws.Import(mu_vars)
    ws.Import(gauss)
    ws.Import(uniform)
    ws.Import(uniform_help)
    ws.Import(obs_data)

    return ws


# Build the workspace and extract variables
ws = build_ws(mu_observed)


# Export the varibles from ws
x_vars = [ws[f"x{i}"] for i in range(len(mu_observed))]
mu_vars = [ws[f"mu{i}"] for i in range(len(mu_observed))]

# Do the morphing
morphing(ROOT.RooMomentMorphFuncND.Linear, len(mu_observed))

# Calculate the nll for the moprhed distribution
# TODO: Fix RooAddPdf::fixCoefNormalization(nset) warnings with new CPU backend
nll_morph = ws["morph"].createNLL(ws["obs_data"], EvalBackend="legacy")

# Initialize the SBI model
model = SBI(ws, len(mu_observed))

# Generate and preprocess training data
model.model_data("gauss", [x.GetName() for x in x_vars], [mu.GetName() for mu in mu_vars], n_samples_train)
model.reference_data(
    "uniform", [x.GetName() for x in x_vars], [mu.GetName() for mu in mu_vars], n_samples_train, "uniform_help"
)
model.preprocessing()

# Train the neural network classifier
model.train_classifier()
sbi_model = model


# Function to compute the likelihood ratio using the trained classifier
def learned_likelihood_ratio(*args):
    n = max(*(len(a) for a in args))
    X = np.zeros((n, len(args)))
    for i in range(len(args)):
        X[:, i] = args[i]
    prob = sbi_model.classifier.predict_proba(X)[:, 1]
    return prob / (1.0 - prob)


# Create combined variable list for ROOT
combined_vars = ROOT.RooArgList()
for var in x_vars + mu_vars:
    combined_vars.add(var)

# Create a custom likelihood ratio function using the trained classifier
lhr_learned = ROOT.RooFit.bindFunction("MyBinFunc", learned_likelihood_ratio, combined_vars)

# Calculate the 'analytical' likelihood ratio
lhr_calc = ROOT.RooFormulaVar("lhr_calc", "x[0] / x[1]", [ws["gauss"], ws["uniform"]])

# Define the 'analytical' negative logarithmic likelihood ratio
nll_gauss = ws["gauss"].createNLL(ws["obs_data"])

# Create the learned pdf and NLL sum based on the learned likelihood ratio
pdf_learned = ROOT.RooWrapperPdf("learned_pdf", "learned_pdf", lhr_learned, True)

nllr_learned = pdf_learned.createNLL(ws["obs_data"])

# Plot the learned and analytical summed negativelogarithmic likelihood
frame1 = mu_vars[0].frame(
    Title="NLL of SBI vs. Morphing;#mu_{1};NLL",
    Range=(mu_observed[0] - 1, mu_observed[0] + 1),
)
nll_gauss.plotOn(frame1, ShiftToZero=True, LineColor="kP6Blue+1", Name="gauss")
ROOT.RooAbsReal.setEvalErrorLoggingMode("Ignore")  # Silence some warnings
nll_morph.plotOn(frame1, ShiftToZero=True, LineColor="kP6Blue+2", Name="morph")
ROOT.RooAbsReal.setEvalErrorLoggingMode("PrintErrors")
nllr_learned.plotOn(frame1, LineColor="kP6Blue", ShiftToZero=True, Name="learned")


# Declare a helper function in ROOT to dereference unique_ptr
ROOT.gInterpreter.Declare(
    """
RooAbsArg &my_deref(std::unique_ptr<RooAbsArg> const& ptr) { return *ptr; }
"""
)

# Choose normalization set for lhr_calc to plot over
norm_set = ROOT.RooArgSet(x_vars)
lhr_calc_final_ptr = ROOT.RooFit.Detail.compileForNormSet(lhr_calc, norm_set)
lhr_calc_final = ROOT.my_deref(lhr_calc_final_ptr)
lhr_calc_final.recursiveRedirectServers(norm_set)

# Plot the likelihood ratio functions
frame2 = x_vars[0].frame(Title="Likelihood ratio r(x_{1}|#mu_{1}=2.5);x_{1};p_{gauss}/p_{uniform}")
lhr_learned.plotOn(frame2, LineColor="kP6Blue", Name="learned_ratio")
lhr_calc_final.plotOn(frame2, LineColor="kP6Blue+1", Name="exact")

# Write the plots into one canvas to show, or into separate canvases for saving.
single_canvas = True

c = ROOT.TCanvas("", "", 1200 if single_canvas else 600, 600)
if single_canvas:
    c.Divide(2)
    c.cd(1)
    ROOT.gPad.SetLeftMargin(0.15)
    frame1.GetYaxis().SetTitleOffset(1.8)
frame1.Draw()

legend1 = ROOT.TLegend(0.43, 0.63, 0.8, 0.87)
legend1.SetFillColor(ROOT.kWhite)
legend1.SetLineColor(ROOT.kWhite)
legend1.SetTextSize(0.04)
legend1.AddEntry("learned", "learned (SBI)", "L")
legend1.AddEntry("gauss", "true NLL", "L")
legend1.AddEntry("morphed", "moment morphing", "L")
legend1.Draw()

if single_canvas:
    c.cd(2)
    ROOT.gPad.SetLeftMargin(0.15)
    frame2.GetYaxis().SetTitleOffset(1.8)
else:
    c.SaveAs("rf617_plot_1.png")
    c = ROOT.TCanvas("", "", 600, 600)

frame2.Draw()

legend2 = ROOT.TLegend(0.53, 0.73, 0.87, 0.87)
legend2.SetFillColor(ROOT.kWhite)
legend2.SetLineColor(ROOT.kWhite)
legend2.SetTextSize(0.04)
legend2.AddEntry("learned_ratio", "learned (SBI)", "L")
legend2.AddEntry("exact", "true ratio", "L")
legend2.Draw()

if not single_canvas:
    c.SaveAs("rf617_plot_2.png")


# Use ROOT's minimizer to compute the minimum and display the results
for nll in [nll_gauss, nllr_learned, nll_morph]:
    minimizer = ROOT.RooMinimizer(nll)
    minimizer.setErrorLevel(0.5)
    minimizer.setPrintLevel(-1)
    minimizer.minimize("Minuit2")
    result = minimizer.save()
    result.Print()

import sys

# Hack to bypass ClearProxiedObjects()
del sys.modules["libROOTPythonizations"]
