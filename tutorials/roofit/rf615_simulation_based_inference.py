## \file
## \ingroup tutorial_roofit
## \notebook
## Use Simulation Based Inference (SBI) in RooFit.
##
## This tutorial shows how to use SBI in ROOT. As reference distribution we
## choose a simple uniform distribution. The target distribution is chosen to
## be gaussian with different mean values.
## The classifier is trained to discriminate between the reference and target
## distribution.
## We see how the neural networks generalize to unknown mean values.
##
## We compare the approach of using the likelihood ratio trick to morphing.
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

# The samples used for training the classifier in this tutorial / rescale for more accuracy
n_samples = 1000


# Morphing as a baseline
def morphing(setting):

    # Define binning for morphing
    grid = ROOT.RooMomentMorphFuncND.Grid(ROOT.RooBinning(4, 0.0, 4.0))
    x_var.setBins(50)

    # Number of 'sampled' gaussians, if you change it, adjust the binning properly
    n_grid = 5

    for i in range(n_grid):
        # Define the sampled gausians
        mu_help = ROOT.RooRealVar(f"mu{i}", f"mu{i}", i)
        help = ROOT.RooGaussian(f"g{i}", f"g{i}", x_var, mu_help, sigma_var)
        workspace.Import(help)

        # Fill the histograms
        hist = workspace[f"g{i}"].generateBinned([x_var], n_samples)

        # Make sure that every bin is filled and we don't get zero probability
        for i_bin in range(hist.numEntries()):
            hist.add(hist.get(i_bin), 1.0)

        # Add the pdf to the workspace
        workspace.Import(ROOT.RooHistPdf(f"histpdf{i}", f"histpdf{i}", [x_var], hist, 1))

        # Add the pdf to the grid
        grid.addPdf(workspace[f"histpdf{i}"], i)

    # Uncomment to see input plots
    # frame1 = x_var.frame()
    # for i in range(n_grid):
    #    workspace[f"histpdf{i}"].plotOn(frame1)
    # c0 = ROOT.TCanvas()
    # frame1.Draw()
    # input() # wait for user input to proceed

    # Create the morphing and add it to the workspace
    morph_func = ROOT.RooMomentMorphFuncND("morph_func", "morph_func", [mu_var], [x_var], grid, setting)
    morph = ROOT.RooWrapperPdf("morph", "morph", morph_func, True)
    workspace.Import(morph)


# Class used in this case to demonstrate the use of SBI in Root
class SBI:
    # Initializing the class SBI
    def __init__(self, workspace):
        # Choose the hyperparameters for training the neural network
        self.classifier = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=1000, random_state=42)
        self.data_model = None
        self.data_ref = None
        self.X_train = None
        self.y_train = None
        self.workspace = workspace

    # Defining the target / training data for different values of mean value mu
    def model_data(self, model, x, mu, n_samples):
        ws = self.workspace
        data_test_model = []
        samples_gaussian = ws[model].generate([ws[x], ws[mu]], n_samples).to_numpy()
        self._training_mus = samples_gaussian[mu]
        data_test_model.extend(samples_gaussian[x])

        self.data_model = np.array(data_test_model).reshape(-1, 1)

    # Generating samples for the reference distribution
    def reference_data(self, model, x, n_samples):
        ws = self.workspace
        # Ensuring the normalization with generating as many reference data as target data
        samples_uniform = ws[model].generate(ws[x], n_samples)
        data_reference_model = np.array(
            [samples_uniform.get(i).getRealValue("x") for i in range(samples_uniform.numEntries())]
        )
        self.data_ref = data_reference_model.reshape(-1, 1)

    # Bringing the data in the right format for training
    def preprocessing(self):
        thetas_model = self._training_mus.reshape(-1, 1)
        thetas_reference = self._training_mus.reshape(-1, 1)
        thetas = np.concatenate((thetas_model, thetas_reference), axis=0)
        X = np.concatenate([self.data_model, self.data_ref])
        self.y_train = np.concatenate([np.ones(len(self.data_model)), np.zeros(len(self.data_ref))])
        self.X_train = np.concatenate([X, thetas], axis=1)

    # Train the classifier
    def train_classifier(self):
        self.classifier.fit(self.X_train, self.y_train)


# Setting the training and toy data samples; the factor 5 to enable a fair comparison to morphing
n_samples_train = n_samples * 5


# Define the "observed" data in a workspace
def build_ws(mu_observed):
    x_var = ROOT.RooRealVar("x", "x", -5, 15)
    mu_var = ROOT.RooRealVar("mu", "mu", mu_observed, 0, 4)
    sigma_var = ROOT.RooRealVar("sigma", "sigma", 1.5)
    gauss = ROOT.RooGaussian("gauss", "gauss", x_var, mu_var, sigma_var)
    uniform = ROOT.RooUniform("uniform", "uniform", x_var)
    obs_data = gauss.generate(x_var, n_samples)
    obs_data.SetName("obs_data")

    # using a workspace for easier processing inside the class
    workspace = ROOT.RooWorkspace()
    workspace.Import(x_var)
    workspace.Import(mu_var)
    workspace.Import(gauss)
    workspace.Import(uniform)
    workspace.Import(obs_data)

    return workspace


# The "observed" data
mu_observed = 2.5
workspace = build_ws(mu_observed)
x_var = workspace["x"]
mu_var = workspace["mu"]
sigma_var = workspace["sigma"]
gauss = workspace["gauss"]
uniform = workspace["uniform"]
obs_data = workspace["obs_data"]

# Training the model
model = SBI(workspace)
model.model_data("gauss", "x", "mu", n_samples_train)
model.reference_data("uniform", "x", n_samples_train)
model.preprocessing()
model.train_classifier()
sbi_model = model


# Compute the likelihood ratio of the classifier for analysis purposes
def compute_likelihood_ratio(x, mu):
    data_point = np.array([[x, mu]])
    prob = sbi_model.classifier.predict_proba(data_point)[:, 1]
    return prob[0]


# Compute the negative logarithmic likelihood ratio summed,
# the function depends just on one variable, the mean value mu
def compute_log_likelihood_sum(mu):
    mu_arr = np.repeat(mu, obs_data.numEntries()).reshape(-1, 1)
    data_point = np.concatenate([obs_data.to_numpy()["x"].reshape(-1, 1), mu_arr], axis=1)
    prob = sbi_model.classifier.predict_proba(data_point)[:, 1]
    return np.sum(np.log((1 - prob) / prob))


# Compute the learned likelihood ratio
llhr_learned = ROOT.RooFit.bindFunction("MyBinFunc", compute_likelihood_ratio, x_var, mu_var)

# Compute the real likelihood ration
llhr_calc = ROOT.RooFormulaVar("llhr_calc", "x[0] / (x[0] + x[1])", [gauss, uniform])

# Create the exact negative log likelihood functions for Gaussian model
nll_gauss = gauss.createNLL(obs_data)

# Create the NLL based on the template morphing pdf
nllr_learned = ROOT.RooFit.bindFunction("MyBinFunc", compute_log_likelihood_sum, mu_var)

# Compute the morphed nll
morphing(ROOT.RooMomentMorphFuncND.Linear)
nll_morph = workspace["morph"].createNLL(obs_data)

# Plot the negative logarithmic summed likelihood
c1 = ROOT.TCanvas()
frame = mu_var.frame(Title="SBI vs. Morphing")
nll_gauss.plotOn(frame, LineColor="g", ShiftToZero=True, Name="gauss")
nllr_learned.plotOn(frame, LineColor="r", LineStyle="--", ShiftToZero=True, Name="learned")
nll_morph.plotOn(frame, LineColor="c", ShiftToZero=True, Name="morphed")
frame.Draw()

# Create a legend and add entries
legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)  # Adjust coordinates as needed
legend.AddEntry("gauss", "Gaussian", "l")
legend.AddEntry("learned", "SBI", "l")
legend.AddEntry("morphed", "Morphed", "l")
legend.Draw()

# Plot the likelihood functions
c2 = ROOT.TCanvas()
frame_x = x_var.frame(Title="Learned vs analytical likelihhood function")
llhr_learned.plotOn(frame_x, LineColor="r", LineStyle="--", Name="learned_ratio")
llhr_calc.plotOn(frame_x, Name="exact")
frame_x.Draw()
# Create a legend and add entries
legend_1 = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)  # Adjust coordinates as needed
legend_1.AddEntry("learned_ratio", "learned_ratio", "l")
legend_1.AddEntry("exact", "exact", "l")
legend_1.Draw()


# Compute the minimum via minuit and display the results
for nll in [nll_gauss, nllr_learned, nll_morph]:
    min = minimizer = ROOT.RooMinimizer(nll)
    minimizer.setErrorLevel(0.5)  # Adjust the error level in the minimization to work with likelihoods
    minimizer.setPrintLevel(-1)
    minimizer.minimize("Minuit2")
    result = minimizer.save()
    result.Print()
