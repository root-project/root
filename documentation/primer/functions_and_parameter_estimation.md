# Functions and Parameter Estimation #

After going through the previous chapters, you already know how to use
analytical functions (class `TF1`), and you got some insight into the
graph (`TGraphErrors`) and histogram classes (`TH1F`) for data
visualisation. In this chapter we will add more detail to the previous
approximate explanations to face the fundamental topic of parameter
estimation by fitting functions to data. For graphs and histograms, ROOT
offers an easy-to-use interface to perform fits - either the fit panel
of the graphical interface, or the `Fit` method. The class `TFitResult`
allows access to the detailed results.

Very often it is necessary to study the statistical properties of
analysis procedures. This is most easily achieved by applying the
analysis to many sets of simulated data (or "pseudo data"), each
representing one possible version of the true experiment. If the
simulation only deals with the final distributions observed in data, and
does not perform a full simulation of the underlying physics and the
experimental apparatus, the name "Toy Monte Carlo" is frequently used
[^5]. Since the true values of all parameters are known in the
pseudo-data, the differences between the parameter estimates from the
analysis procedure w.r.t. the true values can be determined, and it is
also possible to check that the analysis procedure provides correct
error estimates.

## Fitting Functions to Pseudo Data ##

In the example below, a pseudo-data set is produced and a model fitted
to it.

ROOT offers various minimisation algorithms to minimise a chi2 or a
negative log-likelihood function. The default minimiser is MINUIT, a
package originally implemented in the FORTRAN programming language. A
C++ version is also available, MINUIT2, as well as Fumili [@Fumili] an
algorithm optimised for fitting. The
minimisation algorithms can be selected using the static functions of
the `ROOT::Math::MinimizerOptions` class. Steering options for the
minimiser, such as the convergence tolerance or the maximum number of
function calls, can also be set using the methods of this class. All
currently implemented minimisers are documented in the reference
documentation of ROOT: have a look for example to the
`ROOT::Math::Minimizer` class documentation.
\newpage
The complication level of the code below is intentionally a little
higher than in the previous examples. The graphical output of the macro
is shown in Figure [6.1](#f61):

``` {.cpp .numberLines}
@ROOT_INCLUDE_FILE macros/macro8.C
```

Some step by step explanation is at this point necessary:

-   Lines *1-3*: A simple function to ease the make-up of lines.
    Remember that the class `TF1` inherits from `TAttLine`.

-   Lines *5-7* : Definition of a customised function, namely a Gaussian
    (the "signal") plus a parabolic function, the "background".

-   Lines *10-12*: Some make-up for the Canvas. In particular we want
    that the parameters of the fit appear very clearly and nicely on the
    plot.

-   Lines *20-25*: Define and initialise an instance of `TF1`.

-   Lines *27-31*: Define and fill a histogram.

-   Lines *33-38*: For convenience, the same function as for the
    generation of the pseudo-data is used in the fit; hence, we need to
    reset the function parameters. This part of the code is very
    important for each fit procedure, as it sets the initial values of
    the fit.

-   Line *41*: A very simple command, well known by now: fit the
    function to the histogram.

-   Lines *42-46*: Retrieve the output from the fit. Here, we simply
    print the fit result and access and print the covariance matrix of
    the parameters.

-   Lines *54-end*: Plot the pseudo-data, the fitted function and the
    signal and background components at the best-fit values.

[f61]: figures/functions.png "f61"
<a name="f61"></a>

![Fit of pseudo data: a signal shape over a background trend. This plot
is another example of how making a plot "self-explanatory" can help you
better displaying your results. \label{f61}][f61]

## Toy Monte Carlo Experiments ##

Let us look at a simple example of a toy experiment comparing two
methods to fit a function to a histogram, the  $\chi^{2}$

method and a method called "binned log-likelihood fit", both available in ROOT.

As a very simple yet powerful quantity to check the quality of the fit
results, we construct for each pseudo-data set the so-called "pull", the
difference of the estimated and the true value of a parameter,
normalised to the estimated error on the parameter,
$\frac{(p_{estim} - p_{true})}{\sigma_{p}}$. If everything is OK, the
distribution of the pull values is a standard normal distribution, i.e.
a Gaussian distribution centred around zero with a standard deviation of one.

The macro performs a rather big number of toy experiments, where a
histogram is repeatedly filled with Gaussian distributed numbers,
representing the pseudo-data in this example. Each time, a fit is
performed according to the selected method, and the pull is calculated
and filled into a histogram. Here is the code:

``` {.cpp .numberLines}
@ROOT_INCLUDE_FILE macros/macro9.C
```

Your present knowledge of ROOT should be enough to understand all the
technicalities behind the macro. Note that the variable `pull` in line
*61* is different from the definition above: instead of the parameter
error on `mean`, the fitted standard deviation of the distribution
divided by the square root of the number of entries,
`sig/sqrt(n_tot_entries)`, is used.

-   What method exhibits the better performance with the default
    parameters ?

-   What happens if you increase the number of entries per histogram by
    a factor of ten ? Why ?

The answers to these questions are well beyond the scope of this guide.
Basically all books about statistical methods provide a complete
treatment of the aforementioned topics.

[^5]: "Monte Carlo" simulation means that random numbers play a role here
which is as crucial as in games of pure chance in the Casino of Monte Carlo.
