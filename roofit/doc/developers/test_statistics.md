\defgroup roofit_dev_docs_test_statistics New RooFit TestStatistics usage notes
\ingroup roofit_dev_docs
\date December 2021, updated October 2024
\author Patrick Bos
\brief Notes on the new `RooFit::TestStatistics` classes

# RooFit::TestStatistics usage notes

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

On top of the likelihood classes, we also provide for convenience a likelihood builder `NLLFactory`.
This factory analyzes the `pdf` and automatically constructs the proper likelihood, built up from the available `RooAbsL` subclasses.
Options, like specifying constraint terms or global observables, can be passed using method chaining.
The `NLLFactory::build` method finally returns the constructed likelihood as a RooRealL object that can be fit to using RooMinimizer.

The new classes are not per se meant to be used outside of `RooMinimizer`, although they can be.
The main reason is that they do not behave as regular `RooAbsReal` objects, but have their own interface which was kept to the minimum necessary for interacting with `RooMinimizer` as an object that encodes purely the statistics concepts.
However, we do provide the `RooRealL` class, which holds a `RooAbsL` object, but does inherit from `RooAbsReal` as well, so that it can be used in contexts where you would normally use a `RooAbsReal` (like for plotting).

### Usage example: Create an unbinned likelihood object
It is possible to directly create `RooAbsL` based likelihood objects from a pdf and dataset, in this example a `RooUnbinnedL` type:
``` {.cpp}
RooAbsPdf *pdf;
RooDataSet *data;
std::tie(pdf, data) = generate_some_unbinned_pdf_and_dataset(with, some, parameters);

RooFit::TestStatistics::RooUnbinnedL likelihood(pdf, data);
```

However, most of the time, the user will not need **or want** such direct control over the type, but rather just let RooFit figure out what exact likelihood type (`RooAbsL` derived class) is best.
For this situation, the `NLLFactory` factory was created that can be used (for instance) as:
``` {.cpp}
std::shared_ptr<RooFit::TestStatistics::RooAbsL> likelihood = RooFit::TestStatistics::NLLFactory(pdf, data).build();
```
`build()` actually returns a `unique_ptr`; storing the result in a `shared_ptr` as done here is just one possible use-case.

### Usage example: Create a likelihood of a simultaneous PDF with constraint terms and global observables (and other optional arguments)
The `RooAbsPdf::fitTo` or `RooAbsPdf::createNLL` interfaces could take in a set of optional parameters as `RooCmdArg` objects.
In `TestStatistics::NLLFactory`, we have implemented 6 of these options as methods on the factory class, allowing to pass them to a factory object using method chaining.
These option-methods are:
- `NLLFactory::Extended(RooAbsL::Extended extended)`: pass in an enum class used to set extended term calculation `on`, `off` or use `Extended::Auto` to determine automatically based on the pdf whether to activate or not.
- `NLLFactory::ConstrainedParameters(const RooArgSet &constrainedParameters)`: Initialized with a `RooArgSet` of parameters that are constrained. Pdf components dependent on these alone are added to a subsidiary likelihood term.
- `NLLFactory::ExternalConstraints(const RooArgSet &externalconstraints)`: Initialized with a `RooArgSet` of external constraint pdfs, i.e. constraints not necessarily in the pdf itself. These are always added to the subsidiary likelihood.
- `NLLFactory::GlobalObservables(const RooArgSet &globalObservables)`: Initialized with a `RooArgSet` of observables that have a constant value, independent of the dataset events. Pdf components dependent on these alone are added to the subsidiary likelihood. \note This overrides all other likelihood parameters (like those passed as `ConstrainedParameters`) if present.
- `NLLFactory::GlobalObservablesTag(const char *globalObservablesTag)`: a string argument can be given, which should be set as attribute in any pdf component to indicate that it is a global observable. Can be used instead of or in addition to a `GlobalObservables` argument.
- `NLLFactory::EvalBackend(RooFit::EvalBackend evalBackend)`: pass in the enum class that determines which back-end to use for RooFit pdf calculation, like the CPU vectorization enabled back-end, the legacy non-vectorized one or the GPU back-end. \note This is a different set of back-ends than those mentioned below under [Calculators](#calculators) which are independent of the `TestStatistics`/`MultiProcess` framework described in this document.

Using these parameters, creating a likelihood of a simultaneous pdf (i.e. a `RooSimultaneous`) with constrained terms and global observables can be done with:

``` {.cpp}
auto factory = RooFit::TestStatistics::NLLFactory(simultaneous_pdf, data);
factory.ConstrainedParameters(RooArgSet(/*YOUR CONSTRAINED PARAMETERS*/));
factory.GlobalObservables(RooArgSet(/*YOUR GLOBAL OBSERVABLES*/));
auto likelihood = factory.build();
```

or in a single statement, chaining all methods and discarding the factory object itself:

``` {.cpp}
auto likelihood = RooFit::TestStatistics::NLLFactory(simultaneous_pdf, data)
       .ConstrainedParameters(RooArgSet(/*YOUR CONSTRAINED PARAMETERS*/))
       .GlobalObservables(RooArgSet(/*YOUR GLOBAL OBSERVABLES*/))
       .build();
```

The resulting likelihood object will be a `RooSumL` containing a `RooSubsidiaryL` with the constrained parameters and `RooBinnedL` or `RooUnbinnedL` components for each simultaneous pdf component (depending on whether they are binned or unbinned pdfs).
Note that, just like with `fitTo`, the order of setting options does not matter for the created likelihood.

As a side-note: one optional parameter of `RooNLLVar` that is not included in the `RooAbsL` tree is offsetting.
Offsetting has instead been implemented in the calculators that we'll describe next.
This is one of the consequences of the conceptual splitting of functionality into statistics and calculator classes.
Offsetting is a feature of calculation in a fitting context; it enhances numerical precision by subtracting the initial likelihood value from the value that the minimizer sees, thus setting it to zero for the minimizer.
Since this does not impact the derivative terms, it does not affect the fitting result, except for added numerical precision.


## Calculators
`RooFit::TestStatistics` provides two abstract base classes for likelihood calculation: `LikelihoodWrapper` and `LikelihoodGradientWrapper`.
These are used by the `RooAbsMinimizerFcn` implementation `MinuitFcnGrad` which expects them to, respectively, provide likelihood and likelihood gradient values for use by `Minuit2` in fitting the pdf to the dataset.

The `Wrapper`s can be implemented for different kinds of algorithms, or with different kinds of optimization "back-ends" in mind.
Some implementations of each are ready for use in `RooFit` currently:

1. `LikelihoodSerial` is more or less simply a rewrite of the existing serial calculation of a `RooNLLVar`.
2. `LikelihoodJob` also calculates a "`RooNLLVar`", but splits the calculation over worker processes, i.e. over multiple CPU cores, based on `RooFit::MultiProcess`, which is a fork-based multi-processing task execution framework with dynamic load balancing.
3. `LikelihoodGradientJob` calculates gradient components (partial derivatives) in parallel on multiple workers, also based on `RooFit::MultiProcess`.

Other possible implementations could use the GPU or external tools like TensorFlow or use automatically calculated analytical gradients.

The coupling of all these classes to `RooMinimizer` is made via the `MinuitFcnGrad` class, which owns the `Wrappers` that calculate the likelihood components.


### Usage example: `MultiProcess` enabled parallel gradient calculator

The main selling point of using `RooFit::TestStatistics` from a performance point of view is the implementation of the `RooFit::MultiProcess` based `LikelihoodGradientJob` calculator class.
This parallelizes Minuit's MIGRAD's largest bottleneck, providing most of the speed-up in large complex fits.
To use it, one should create a `RooMinimizer` using the new constructor that takes a `RooAbsL`-based likelihood instead of a `RooAbsReal`.

Taking any of the above created `likelihood` objects (as long as they are in a `std::shared_ptr`), we can create a `RooMinimizer` with parallel gradient calculation using:
``` {.cpp}
std::shared_ptr<RooAbsL> likelihood = /* see examples above */;
RooMinimizer::Config cfg;
cfg.parallelize = -1;
RooMinimizer m(likelihood, cfg);
```

By setting `cfg.parallelize` to `-1`, `RooFit::MultiProcess` spins up as many workers as there are cores in the system (as detected by `std::thread::hardware_concurrency()`).
To change the number of workers, set `cfg.parallelize` to the desired number.

As noted above, offsetting is purely a function of the `RooMinimizer` when using `TestStatistics` classes.
Whereas with `fitTo` we can pass in a `RooFit::Offset(true)` optional `RooCmdArg` argument to activate offsetting, here we must do it on the minimizer as follows:
``` {.cpp}
m.setOffsetting(true);
```

All existing functionality of the `RooMinimizer` can be used on `TestStatistics` likelihoods as well.
For instance, running a MIGRAD fit:
``` {.cpp}
m.migrad()
```

### Usage example: fully `MultiProcess` enabled fit (likelihood and gradient)

After the gradient calculation, the next bottleneck to tackle in the MIGRAD process is the line-search phase.
However, this phase consists simply of a couple of likelihood evaluations.
To parallelize this, one can enable the `MultiProcess` enabled likelihood calculator as well:
``` {.cpp}
RooMinimizer::Config cfg;
cfg.parallelize = -1;
cfg.enableParallelDescent = true;
RooMinimizer m(likelihood, cfg);
```
By default, this is turned off, so it needs to be explicitly enabled.
Make sure to also try the alternative calculation backends from the `RooFit::EvalBackend` set to speed up the line-search phase (and all other phases as well).
It will depend on your use-case which combination of options gives you the best computational performance.


## Constant term optimization
The `RooAbsTestStatistic` based classes not only combine statistics and calculation, but also constant term optimization routines.
These can be run on PDFs and datasets before starting a fit.
They search the calculation graph for parts that are independent of the fit parameters, precalculates them, and adds them to (a clone of) the dataset so that these values can be used during calculation.

In `RooFit::TestStatistics`, we separated this functionality out into the `ConstantTermsOptimizer` class.
In fact, it is not so much a class, as it is a collection of static functions that can be applied to any combination of pdf and dataset.
This class does essentially the same as `constOptimizeTestStatistic` did on a `RooNLLVar`, except that it has been factored out into a separate class.

### Usage example: apply constant term optimization on pdf and dataset inside a likelihood
Applying the default `ConstantTermsOptimizer` optimization routines on the pdf and dataset inside a `RooAbsL` likelihood is as simple as:

``` {.cpp}
likelihood.constOptimizeTestStatistic();
```
This applies constant term optimization to the cloned pdf and dataset inside the likelihood object.
It will not modify anything outside of the likelihood.

Optimization can also be activated through the minimizer, which may be more familiar to most users.
Given the `RooMinimizer` object `m` as defined in the example above, we can do:
``` {.cpp}
m.optimizeConst(2);
```

For the adventurous user, it is also possible to apply constant term optimization to a pdf and dataset directly without needing a likelihood object, e.g. given some `RooArgSet` set of observables `normSet`:
``` {.cpp}
bool applyTrackingOpt = true;
ConstantTermsOptimizer::enableConstantTermsOptimization(&pdf, &normSet, dataset, applyTrackingOpt);
```
We refer to RooFit documentation for more about "tracking optimization" which can be enabled or disabled using the final boolean parameter.

## Load balancing options

A number of calculation strategy options are available to tune load balancing.
Depending on the exact pdf, the effect of these options can be quite dramatic.
The ideal to strive for is to distribute tasks over worker CPUs equally in terms of total amount of work.
Since tasks can vary in length, this means all workers simultaneously have something to do and they all finish at the same time.
The following options using `RooFit::MultiProcess::Config` should all be set before creating the `RooMinimizer` that will perform the fit.


### Number of workers
The most basic setting to tweak is the number of workers $N_w$ using `RooFit::MultiProcess::Config::setDefaultNWorkers`.
By default, the number of workers is set equal to the number of CPUs.
Possibly, in some situations it may be better to decrease this number by one or two.
This is because ``RooFit::MultiProcess`` uses two more processes: the master and the queue processes, giving a total of $N_w + 2$.
This means two cores by default will have to host two processes, which may slow down the workers on those cores a bit.
While the worker processes are working, the master and queue processes should not be doing much, so interference should be minimal and you may not want two CPU cores to be doing almost nothing.
On the other hand, if adding two workers does not yield much extra speed (due to Amdahl's law), giving the master and worker their own cores may give a shorter total runtime.


### Splitting likelihoods
When using the parallel likelihood `LikelihoodJob`, it will split the likelihood into parts and calculate them at the workers in parallel.

Calculation of each part costs a little extra wallclock time because of communication overhead.
Splitting in a lower number of larger parts thus minimizes the total overhead time.
However, increasing the number of parts may improve load balancing, which could lead to lower total runtimes.

With this in mind, there are two ways of splitting the likelihood: we can split into (blocks of) events and can split `RooSumL` likelihoods into (groups of) components.
When splitting over events, this class emulates the existing `NumCPU(>1)` functionality of the `RooAbsTestStatistic` tree, which was implemented based on `RooRealMPFE`.

By default, `LikelihoodJob` will split over components, making every component a single worker task.
To reduce the number of (groups of) components, set `RooFit::MultiProcess::Config::LikelihoodJob::defaultNComponentTasks` to a different desired number of tasks.
Set `RooFit::MultiProcess::Config::LikelihoodJob::defaultNEventTasks` to anything higher than 1 to split into blocks of events.
Both splitting strategies can be used simultaneously, resulting in worker tasks
that calculate (groups of) components for partial event ranges.

In the end all components will automatically be summed by the master process.


### Manual task queue order
If the above two options still do not yield the desired scaling performance, an imperfect load balancing of tasks over workers could be to blame.
This could be because the execution order of tasks by default is FIFO: "first in, first out".
In most cases, the fact that workers "steal" tasks off the queue when they are idle prevents them from being idle for too long, thus yielding acceptable load balancing automatically.
A pathological case for this strategy is when a very long task is placed at the end of the queue.
This means that the worker that steals this task will probably still be working while all other workers with shorter final jobs are long done.
Ideally one would like to move such long tasks to the beginning of the queue.
In fact, ordering tasks by decreasing runtime is ideal.

The task order can be manually adjusted.
First enable this feature by calling `RooFit::MultiProcess::Config::Queue::setQueueType(RooFit::MultiProcess::Config::Queue::QueueType::Priority)`.
Then, there are two ways to set desired execution order.
The first is by setting task priorities with `RooFit::MultiProcess::Config::Queue::setTaskPriorities`.
Higher priority means it will be put ahead in the queue of lower priority tasks.
The second is by directly suggesting an order with `...::suggestTaskOrder`.
In this case, a lower number sets the task ahead in the queue.
Note that in both cases a strict execution order cannot be guaranteed.
This is because the workers are separate processes that may get unpredictably delayed because of operating system controlled scheduling.

To measure the runtime of tasks, the `TimingAnalysis` option can be switched on by setting the `timingAnalysis` member of the `RooMinimizer::Config` object to `true`.
Note that this is an experimental feature.
The resulting timing JSON output can be analyzed visually with the `RooFit::MultiProcess::HeatmapAnalyzer` class.


## Caveats
Some functionality that users of `RooAbsPdf::fitTo` or `RooAbsPdf::createNLL` were used to has not yet been ported to this namespace.
However, the functionality that is implemented has been tested thoroughly for a set of common usage patterns and should work as expected.

The classes implemented here will give the exact same numerical results for most fits.
One notable exception is fitting _simultaneous_ pdfs with a _constrained_ term _when using offsetting_.
Because offsetting is handled differently in the `TestStatistics` classes compared to the way it was done in the object returned from `RooAbsPdf::createNLL` (a `RooAddition` of an offset `RooNLLVar` and a non-offset `RooConstraintSum`, whereas `RooSumL` applies the offset to the
total sum of its binned, unbinned and constraint components), we cannot always expect exactly equal results for fits with likelihood offsetting enabled.

