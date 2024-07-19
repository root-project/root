# \file
## \ingroup tutorial_roofit
## \notebook
## Use Simulation Based Inference (SBI) in RooFit
##
## This tutorial shows how to use SBI in ROOT. As reference distribution we
## choose a simple uniform distribution. The target distribution is chosen to
## be gaussian with different mean values. 
## The classifier is trained to discriminate between the reference and target 
## distribution. 
## We see how the neural networks generalize to unknown mean values.
##
##
## \macro_code
## \macro_output
##
## \date July 2024
## \author Robin Syring



import ROOT
import numpy as np
from sklearn.neural_network import MLPClassifier


# The samples used for training the classifier in this tutorial
n_samples = 100000

# different mean values used for training the classifier
mu_vals = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

# defining a cpp wrapper 
ROOT.gInterpreter.Declare("""
class RooPyLikelihood : public RooAbsReal {
public:
   RooPyLikelihood(const char *name, const char *title, RooArgList &varlist)
      : RooAbsReal(name, title), m_varlist("!varlist", "All variables(list)", this)
   {
      m_varlist.add(varlist);
   }
   // copy constructor
   RooPyLikelihood(const RooPyLikelihood &right, const char *name = nullptr)
      : RooAbsReal(right, name), m_varlist("!varlist", this, right.m_varlist)
   {
   }
   // virtual destructor
   virtual ~RooPyLikelihood() {}
   // clone method
   RooPyLikelihood *clone(const char *name) const override { return new RooPyLikelihood(*this, name); }
   // the actual evaluation of function (will be redefined in Python!)
   Double_t evaluate() const override { return 1; }
   // getter for varlist
   const RooArgList &varlist() const { return m_varlist; }
protected:
   RooListProxy m_varlist; // all variables as list of variables
};
""")


# Overwriting the cpp function
def make_likelihood(name, title, func, variables):
    class MyLlh(ROOT.RooPyLikelihood):
        def __init__(self, name, title, variables):
            super(MyLlh, self).__init__(name, title, ROOT.RooArgList(variables))

        def evaluate(self):
            return func(*(v.getVal() for v in self.varlist()))

        def clone(self, newname=False):
            cl = MyLlh(newname if newname else self.GetName(), self.GetTitle(), self.varlist())
            ROOT.SetOwnership(cl, False)
            return cl

    return MyLlh(name, title, variables)


# class used in this case to demonstate the use of SBI in Root
class SBI:

    # initializing the class SBI
    def __init__(self, workspace):
        # Choose the hyperparameters for training the neural network
        self.classifier = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=1000, random_state=42)
        self.mu_vals = mu_vals
        self.data_model = None
        self.data_ref = None
        self.X_train = None
        self.y_train = None
        self.workspace = workspace

    # defining the target / training data for different values of mean value mu 
    def model_data(self, model, x, mu, n_samples):
        ws = self.workspace
        data_test_model = []
        # Preventing the old mu value from overwriting
        old_val = ws[mu].getVal()

        # getting training data for each value of mu
        for theta in self.mu_vals:
            ws[mu].setVal(theta)
            samples_gaussian = ws[model].generate(ws[x], n_samples)
            data_test_model.extend([samples_gaussian.get(i).getRealValue("x") for i in range(samples_gaussian.numEntries())])
        ws[mu].setVal(old_val)
        self.data_model = np.array(data_test_model).reshape(-1, 1)

    # generating samples for the reference distribution 
    def reference_data(self, model, x, n_samples):
        ws = self.workspace
        # ensuring the normalization with generating as many reference data as target data
        samples_uniform = ws[model].generate(ws[x], n_samples * len(mu_vals))
        data_reference_model = np.array([samples_uniform.get(i).getRealValue("x") for i in range(samples_uniform.numEntries())])
        self.data_ref = data_reference_model.reshape(-1, 1)

    # bringing the data in the right format for training
    def preprocessing(self):
        repeats_model = len(self.data_model) // len(self.mu_vals)
        thetas_model = np.repeat(self.mu_vals, repeats_model).reshape(-1, 1)
        repeats_reference = len(self.data_ref) // len(self.mu_vals)
        thetas_reference = np.repeat(self.mu_vals, repeats_reference).reshape(-1, 1)
        thetas = np.concatenate((thetas_model, thetas_reference), axis=0)
        X = np.concatenate([self.data_model, self.data_ref])
        self.y_train = np.concatenate([np.zeros(len(self.data_model)), np.ones(len(self.data_ref))])
        self.X_train = np.concatenate([X, thetas], axis=1)

    # train the classifier
    def train_classifier(self):
        self.classifier.fit(self.X_train, self.y_train)


# Setting the training and toy data samples 
n_samples_train = n_samples

# The "observed" data 
mu_observed = 2.

# define the "observed" data
x_var = ROOT.RooRealVar("x", "x", -12, 12)
mu_var = ROOT.RooRealVar("mu", "mu", mu_observed, -12, 12)
sigma_var = ROOT.RooRealVar("sigma", "sigma", 1.5, 0.1, 10)
gauss = ROOT.RooGaussian("gauss", "gauss", x_var, mu_var, sigma_var)
uniform = ROOT.RooUniform("uniform", "uniform", x_var)
obs_data = gauss.generate(x_var, n_samples)

# using a workspace for easier processing inside the class
workspace = ROOT.RooWorkspace()
workspace.Import(gauss)
workspace.Import(uniform)
workspace.Import(obs_data)
workspace.Print()

# training the model 
model = SBI(workspace)
model.model_data("gauss", "x", "mu", n_samples_train)
model.reference_data("uniform", "x", n_samples_train)
model.preprocessing()
model.train_classifier()
sbi_model = model


# compute the likelihood ratio of the classifier for analysis purposes
def compute_likelihood_ratio(x, mu):
    data_point = np.array([[x, mu]])
    prob = sbi_model.classifier.predict_proba(data_point)[:, 1]
    return 1 - prob[0]

# compute the negative logarithmic likelihood ratio summed
# the function depends just on one variable, the mean value mu
def compute_likelihood_sum(mu):
    mu_arr = np.repeat(mu, obs_data.numEntries()).reshape(-1, 1)
    data_point = np.concatenate([obs_data.to_numpy()["x"].reshape(-1, 1), mu_arr], axis=1)
    prob = sbi_model.classifier.predict_proba(data_point)[:, 1]
    return -np.sum(np.log(1 - prob))


# compute the likelihood ratio
nll_ratio = make_likelihood("MyLlh", "My Llh", compute_likelihood_ratio, ROOT.RooArgList(x_var, mu_var))

# compute the real likelihood ration
real_ratio = ROOT.RooFormulaVar("real_ratio", "x[0] / (x[0] + x[1])", [gauss, uniform])


# compute the summed logarithmic likelihood
nll_learned = make_likelihood("MyLlh", "My Llh", compute_likelihood_sum, ROOT.RooArgList(mu_var))

# Create likelihood function
nll_ref = gauss.createNLL(obs_data)


# Plot the logarithmic summed likelihood
c1 =ROOT.TCanvas()
frame = mu_var.frame(Title="Learned vs analytical summed logarithmic Likelihood", Range=(mu_observed-0.1, mu_observed+0.1))
# Set the y-axis range
frame.SetMinimum(0)  # Replace y_min with your desired minimum value
frame.SetMaximum(0.3)  # Replace y_max with your desired maximum value
nll_ref.plotOn(frame, ShiftToZero=True)
nll_learned.plotOn(frame, LineColor="r", ShiftToZero=True)
frame.Draw()
c1.SaveAs("Logarithmic_summed.png")

# Plot the likelihood functions
c2 =ROOT.TCanvas()
frame_x = x_var.frame(Title="Learned vs analytical likelihhood function")
real_ratio.plotOn(frame_x, )
nll_ratio.plotOn(frame_x, LineColor="r",)
frame_x.Draw()
c2.SaveAs("llh_function.png")


# compute the minimum via minuit and display the results
for i in [nll_ref, nll_learned]:
    min = minimizer = ROOT.RooMinimizer(i)
    minimizer.setErrorLevel(0.5)    # adjust the error level in the minimization to work with likelihoods
    minimizer.setPrintLevel(-1)
    minimizer.minimize("Minuit2")
    result = minimizer.save()
    result.Print()


