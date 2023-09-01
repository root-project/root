# `RooFit::TestStatistics` usage notes

The `RooFit::TestStatistics` namespace contains a major refactoring of the `RooAbsTestStatistic`-`RooAbsOptTestStatistic`-`RooNLLVar` inheritance tree into:

1. statistics-based classes on the one hand;
2. calculation/evaluation/optimization based classes on the other hand.

The motivation for this refactoring was also twofold:

1. These test statistics classes make a cleaner separation of concerns than the existing `RooAbsTestStatistic` based tree and are hence more maintainable and future proof.
2. They provided a place for us to try out new parallelized gradient calculation methods using the `RooFit::MultiProcess` module. See the usage example below on how to use this.

## Statistics
The likelihood is the central unit on the statistics side.
The `RooAbsL` class is implemented for four kinds of likelihoods: binned, unbinned, "subsidiary" (an optimization for numerical stability that gathers components like global observables) and "sum" (over multiple components of the other types), in the correspondingly named classes `RooBinnedL`, `RooUnbinnedL`, `RooSubsidiaryL` and `RooSumL`.
These classes provide a `evaluatePartition` function that allows for computing them in parallelizable chunks that can be used by the calculator classes as they see fit.

On top of the likelihood classes, we also provide for convenience a likelihood builder `buildLikelihood`, as a free function in the namespace.
This function analyzes the `pdf` and automatically constructs the proper likelihood, built
up from the available `RooAbsL` subclasses.

The new classes are not per se meant to be used outside of `RooMinimizer`, although they can be.
The main reason is that they do not behave as regular `RooAbsReal` objects, but have their own interface which was kept to the minimum necessary for interacting with `RooMinimizer` as an object that encodes purely the statistics concepts.
However, we do provide the `RooRealL` class, which holds a `RooAbsL` object, but does inherit from `RooAbsReal` as well, so that it can be used in contexts where you would normally use a `RooAbsReal` (like for plotting).

### Usage example: Create an unbinned likelihood object
It is possible to directly create `RooAbsL` based likelihood objects from a pdf and dataset, in this example a `RooUnbinnedL` type:
```c++
RooAbsPdf *pdf;
RooDataSet *data;
std::tie(pdf, data) = generate_some_unbinned_pdf_and_dataset(with, some, parameters);

RooFit::TestStatistics::RooUnbinnedL likelihood(pdf, data);
```

However, most of the time, the user will not need **or want** such direct control over the type, but rather just let RooFit figure out what exact likelihood type (`RooAbsL` derived class) is best.
For this situation, the `buildLikelihood` function was created that can be used (for instance) as:
```c++
std::shared_ptr<RooFit::TestStatistics::RooAbsL> likelihood = RooFit::TestStatistics::buildLikelihood(pdf, data);
```
`buildLikelihood` actually returns a `unique_ptr`; storing the result in a `shared_ptr` as done here is just one possible use-case.

### Usage example: Create a likelihood of a simultaneous PDF with constraint terms and global observables (and other optional arguments)
The `RooAbsPdf::fitTo` or `RooAbsPdf::createNLL` interfaces could take in a set of optional parameters as `RooCmdArg` objects.
In `TestStatistics::buildLikelihood`, we have implemented 4 of these options as separate types while an additional one is supported as a simple string:
- `RooAbsL::Extended`: an enum class used to set extended term calculation `on`, `off` or use `Extended::Auto` to determine automatically based on the pdf whether to activate or not.
- `ConstrainedParameters`: Initialized with a `RooArgSet` of parameters that are constrained. Pdf components dependent on these alone are added to a subsidiary likelihood term.
- `ExternalConstraints`: Initialized with a `RooArgSet` of external constraint pdfs, i.e. constraints not necessarily in the pdf itself. These are always added to the subsidiary likelihood.
- `GlobalObservables`: Initialized with a `RooArgSet` of observables that have a constant value, independent of the dataset events. Pdf components dependent on these alone are added to the subsidiary likelihood. \note This overrides all other likelihood parameters (like those in the `ConstrainedParameters` argument) if present.
- Finally, a string argument `global_observables_tag` can be given, which should be set as attribute in any pdf component to indicate that it is a global observable. Can be used instead of or in addition to a `GlobalObservables` argument.

Using these parameters, creating a likelihood of a simultaneous pdf (i.e. a `RooSimultaneous`) with constrained terms and global observables can be done with:

```c++
auto likelihood = RooFit::TestStatistics::buildLikelihood(
  simultaneous_pdf, data,
  RooFit::TestStatistics::ConstrainedParameters(RooArgSet(/*YOUR CONSTRAINED PARAMETERS*/)),
  RooFit::TestStatistics::GlobalObservables(RooArgSet(/*YOUR GLOBAL OBSERVABLES*/)));
```

The resulting object will be a `RooSumL` containing a `RooSubsidiaryL` with the constrained parameters and `RooBinnedL` or `RooUnbinnedL` components for each simultaneous pdf component (depending on whether they are binned or unbinned pdfs).
Note that, just like for `fitTo`, the order of the parameters is arbitrary.
The difference is that `fitTo` has a step to analyze the `RooCmdArg` optional parameters for their content dynamically at runtime, while in `buildLikelihood` the arguments are statically typed and so no further runtime analysis is needed.

As a side-note: one optional parameter of `RooNLLVar` that is not included in the `RooAbsL` tree is offsetting.
Offsetting has instead been implemented in the calculators that we'll describe next.
This is one of the consequences of the conceptual splitting of functionality into statistics and calculator classes.
Offsetting is a feature of calculation in a fitting context; it enhances numerical precision by subtracting the initial likelihood value from the value that the minimizer sees, thus setting it to zero for the minimizer.
Since this does not impact the derivative terms, it does not affect the fitting result, except for added numerical precision.


## Calculators
`RooFit::TestStatistics` provides two abstract base classes for likelihood calculation: `LikelihoodWrapper` and `LikelihoodGradientWrapper`.
These are used by the `RooAbsMinimizerFcn` implementation `MinuitFcnGrad` which expects them to, respectively, provide likelihood and likelihood gradient values for use by `Minuit2` in fitting the pdf to the dataset.

The `Wrapper`s can be implemented for different kinds of algorithms, or with different kinds of optimization "back-ends" in mind.
One implementation of each is ready for use in `RooFit` currently:

1. `LikelihoodSerial` is more or less simply a rewrite of the existing serial calculation of a `RooNLLVar`.
2. `LikelihoodGradientJob` calculates the partial derivatives or the gradient in parallel on multiple CPUs/cores, based on `RooFit::MultiProcess`, which is a fork-based multi-processing task execution framework with dynamic load balancing.

Other possible implementations could use the GPU or external tools like TensorFlow.

The coupling of all these classes to `RooMinimizer` is made via the `MinuitFcnGrad` class, which owns the `Wrappers` that calculate the likelihood components.

Note: a second `LikelihoodWrapper` class called `LikelihoodJob` is also available.
This class emulates the existing `NumCPU(>1)` functionality of the `RooAbsTestStatistic` tree, which is implemented based on `RooRealMPFE`.
This class is not yet thoroughly tested and should not be considered production ready.

### Usage example: `MultiProcess` enabled parallel gradient calculator

The main selling point of using `RooFit::TestStatistics` from a performance point of view is the implementation of the `RooFit::MultiProcess` based `LikelihoodGradientJob` calculator class.
To use it, one should create a `RooMinimizer` using the new constructor that takes a `RooAbsL`-based likelihood instead of a `RooAbsReal`.

Taking any of the above created `likelihood` objects (as long as they are in a `std::shared_ptr`), we can create a `RooMinimizer` with parallel gradient calculation using:
```c++
std::shared_ptr<RooAbsL> likelihood = /* see examples above */;
RooMinimizer m(likelihood);
```

By default, `RooFit::MultiProcess` spins up as many workers as there are cores in the system (as detected by `std::thread::hardware_concurrency()`).
To change the number of workers, call `RooFit::MultiProcess::Config::setDefaultNWorkers(desired_N_workers)` **before** creating the `RooMinimizer`.

As noted above, offsetting is purely a function of the `RooMinimizer` when using `TestStatistics` classes.
Whereas with `fitTo` we can pass in a `RooFit::Offset(true)` optional `RooCmdArg` argument to activate offsetting, here we must do it on the minimizer as follows:
```c++
m.setOffsetting(true);
```

All existing functionality of the `RooMinimizer` can be used on `TestStatistics` likelihoods as well.
For instance, running a `migrad` fit:
```c++
m.migrad()
```

## Constant term optimization
The `RooAbsTestStatistic` based classes not only combine statistics and calculation, but also constant term optimization routines.
These can be run on PDFs and datasets before starting a fit.
They search the calculation graph for parts that are independent of the fit parameters, precalculates them, and adds them to (a clone of) the dataset so that these values can be used during calculation.

In `RooFit::TestStatistics`, we separated this functionality out into the `ConstantTermsOptimizer` class.
In fact, it is not so much a class, as it is a collection of static functions that can be applied to any combination of pdf and dataset.
This class does essentially the same as `constOptimizeTestStatistic` did on a `RooNLLVar`, except that it has been factored out into a separate class.

### Usage example: apply constant term optimization on pdf and dataset inside a likelihood
Applying the default `ConstantTermsOptimizer` optimization routines on the pdf and dataset inside a `RooAbsL` likelihood is as simple as:

```c++
likelihood.constOptimizeTestStatistic();
```
This applies constant term optimization to the cloned pdf and dataset inside the likelihood object.
It will not modify anything outside of the likelihood.

Optimization can also be activated through the minimizer, which may be more familiar to most users.
Given the `RooMinimizer` object `m` as definied in the example above, we can do:
```c++
m.optimizeConst(2);
```

For the adventurous user, it is also possible to apply constant term optimization to a pdf and dataset directly without needing a likelihood object, e.g. given some `RooArgSet` set of observables `normSet`:
```c++
bool applyTrackingOpt = true;
ConstantTermsOptimizer::enableConstantTermsOptimization(&pdf, &normSet, dataset, applyTrackingOpt);
```
We refer to RooFit documentation for more about "tracking optimization" which can be enabled or disabled using the final boolean parameter.

## Caveats
This package is still under development.
Some functionality that users of `RooAbsPdf::fitTo` or `RooAbsPdf::createNLL` were used to has not yet been ported to this namespace.
However, the functionality that is implemented has been tested thoroughly for a set of common usage patterns and should work as expected.

The classes implemented here will give the exact same numerical results for most fits.
One notable exception is fitting _simultaneous_ pdfs with a _constrained_ term _when using offsetting_.
Because offsetting is handled differently in the `TestStatistics` classes compared to the way it was done in the object returned from `RooAbsPdf::createNLL` (a `RooAddition` of an offset `RooNLLVar` and a non-offset `RooConstraintSum`, whereas `RooSumL` applies the offset to the
total sum of its binned, unbinned and constraint components), we cannot always expect exactly equal results for fits with likelihood offsetting enabled.

