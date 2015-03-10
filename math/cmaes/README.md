### This is support for black-box optimization with CMA-ES from within ROOT (http://root.cern.ch/drupal/).

This work was supported by the ANR-2010-COSI-002 grant of the French NationalA Research Agency.

** This work is improved when needed, see status here: https://github.com/beniz/libcmaes/issues/13 **

** This work may be about to be merged into ROOT repository, follow integration here: 
https://github.com/root-mirror/root/pull/40 **

libcmaes can be used from CERN's ROOT6 as a seamless replacement or addition to Minuit2 optimizer. It is designed to be used from ROOT6 **exactly** as Minuit2 is used, so code using Minuit2 should be easily run against CMA-ES.

Below are instructions for testing it out. 

### Building ROOT6 and libcmaes
As for now, the only way to use libcmaes is from ROOT6, using the following special repository, and compiling it from sources (1): https://github.com/beniz/root

* get ROOT6 from https://github.com/beniz/root/tree/cmaes4root_master, configure & compile it (this will take a while) (2):
````
git clone https://github.com/beniz/root.git
cd root
```
It is recommended to build with cmake, see below.

#### Build with autoconf (configure) ####
For this, you need to have libcmaes installed already, see https://github.com/beniz/libcmaes/wiki
```
./configure --enable-minuit2 --enable-roofit --enable-python --with-cmaes-incdir=/home/yourusername/include/libcmaes --with-cmaes-libdir=/home/yourusername/lib
make
````
use make -jx where x is the number of cores on your system in order to minimize the building time.

#### Build with cmake ####
````
mkdir mybuild
cd mybuild
cmake ../ -Dall=on -Dtesting=on -Dlibcmaes=on
make
````
use make -jx where x is the number of cores on your system in order to minimize the building time.

### Running an example with CMA-ES
To run the basic fitting of a Gaussian, originally taken from Minuit2's tutorial files, do:
````
root
.L tutorials/fit/cmaesGausFit.C++g
cmaesGausFit()
````
You should see a plot similar to 
![cmaes_gaus_fit_root_errors](https://cloud.githubusercontent.com/assets/3530657/2890890/4d96ae1c-d52d-11e3-9610-f24790b23e98.png)

To quick test competitiveness against Minuit2:
````
root
.L tutorials/fit/cmaesFitBench.C
 cmaesFitBench()
````
You should witness a plot similar to
![](http://juban.free.fr/stuff/libcmaes/cmaes_minuit2_competitive.png)

### Running a benchmark comparison of CMA-ES and Minuit2

To run the current benchmark and visualize results, take the following steps:
````
root
.L tutorials/fit/cmaesFullBench.C
run_experiments(10)
python math/cmaes/test/cmaesFullBench.py
````

This should show a series of histograms comparing results from both optimizers on a selection of problems.

### Options to the CMA-ES minimizers within ROOT
There's built-in control for several hyper-parameters and options of CMA-ES:
* several flavors of the algorithm are available, and can be choosen at creation of the Minimizer object:
````
TVirtualFitter::SetDefaultFitter(``acmaes'');
````
or
````
ROOT::Fit::Fitter fitter;
fitter.Config().SetMinimizer(``cmaes'',''acmaes'');
````
The available algorithms are: `cmaes, ipop, bipop, acmaes, aipop, abipop, sepcmaes, sepipop, sepbipop`. 

'acmaes' should be the most appropriate in most cases, and 'sepacmaes' when the number of dimensions nears a thousand.

The options below are not required, but can be used by filling up a MinimizerOptions object beforehand:
````
const char *fitter = "acmaes"
TVirtualFitter::SetDefaultFitter(fitter);
ROOT::Math::IOptions &opts = ROOT::Math::MinimizerOptions::Default(fitter);
opts.SetIntValue("lambda",100);
````
Options below are not activated by default:
* 'sigma': initial step-size
* 'lambda': number of offsprings at each generation
* 'noisy': flag that updates some hyper-parameters if the objective function is noisy
* 'restarts': maximum number of restarts, only applies to ipop, bipop, aipop, abipop, sepipop and sepbipop
* 'ftarget': the objective function target that stops optimization when reached, useful when the final value is known, e.g. 0
* 'fplot': output file in libcmaes format for later plotting of eigenvalues and state convergence, mostly for debug purposes
* 'lscaling': automatic linear scaling of parameters with auto-selection of step-size sigma, usually recommended if results are not satisfactory
* 'mt_feval': allows for parallel calls of the objective function, faster but the objective function is required to be thread-safe.

### Using CMA-ES from RooFit
libcmaes support within ROOT extends to RooFit without effort.

From Roofit, it is enough to set the Minimizer with
```C++

```
and from PyROOT for example
```Python
RooFit.Minimizer("cmaes","acmaes")
```

For setting CMA-ES custom options, such as 'sigma', 'lambda' or 'lscaling', it is enough to set the options as explained in the non RooFit case:
```C++
ROOT::Math::IOptions &opts = ROOT::Math::MinimizerOptions::Default(fitter);
opts.SetIntValue("lambda",100);
```
and from PyRoot
```Python
opt = ROOT.Math.MinimizerOptions.Default("cmaes")
opt.SetIntValue("lambda",100)
```

(1) more convenient ways will be provided.
(2) we recommend building support for both Minuit2 (i.e. for comparison to CMA-ES) and debug. 