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
[^4]. Since the true values of all parameters are known in the
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
algorithm optimised for fitting. Genetic algorithms and a stochastic
minimiser based on simulated annealing are also available. The
minimisation algorithms can be selected using the static functions of
the `ROOT::Math::MinimizerOptions` class. Steering options for the
minimiser, such as the convergence tolerance or the maximum number of
function calls, can also be set using the methods of this class. All
currently implemented minimisers are documented in the reference
documentation of ROOT: have a look for example to the
`ROOT::Math::Minimizer` class documentation.

The complication level of the code below is intentionally a little
higher than in the previous examples. The graphical output of the macro
is shown in Figure [6.1](#f61):

``` {.cpp .numberLines}
 void format_line(TAttLine* line,int col,int sty){
     line->SetLineWidth(5); line->SetLineColor(col);
     line->SetLineStyle(sty);}

 double the_gausppar(double* vars, double* pars){
     return pars[0]*TMath::Gaus(vars[0],pars[1],pars[2])+
         pars[3]+pars[4]*vars[0]+pars[5]*vars[0]*vars[0];}

 int macro8(){
     gStyle->SetOptTitle(0); gStyle->SetOptStat(0);
     gStyle->SetOptFit(1111); gStyle->SetStatBorderSize(0);
     gStyle->SetStatX(.89); gStyle->SetStatY(.89);

     TF1 parabola("parabola","[0]+[1]*x+[2]*x**2",0,20);
     format_line(&parabola,kBlue,2);

     TF1 gaussian("gaussian","[0]*TMath::Gaus(x,[1],[2])",0,20);
     format_line(&gaussian,kRed,2);

     TF1 gausppar("gausppar",the_gausppar,-0,20,6);
     double a=15; double b=-1.2; double c=.03;
     double norm=4; double mean=7; double sigma=1;
     gausppar.SetParameters(norm,mean,sigma,a,b,c);
     gausppar.SetParNames("Norm","Mean","Sigma","a","b","c");
     format_line(&gausppar,kBlue,1);

     TH1F histo("histo","Signal plus background;X vals;Y Vals",
                50,0,20);
     histo.SetMarkerStyle(8);

     // Fake the data
     for (int i=1;i<=5000;++i) histo.Fill(gausppar.GetRandom());

     // Reset the parameters before the fit and set
     // by eye a peak at 6 with an area of more or less 50
     gausppar.SetParameter(0,50);
     gausppar.SetParameter(1,6);
     int npar=gausppar.GetNpar();
     for (int ipar=2;ipar<npar;++ipar)
         gausppar.SetParameter(ipar,1);

     // perform fit ...
     TFitResultPtr frp = histo.Fit(&gausppar, "S");

     // ... and retrieve fit results
     frp->Print(); // print fit results
     // get covariance Matrix an print it
     TMatrixDSym covMatrix (frp->GetCovarianceMatrix());
     covMatrix.Print();

     // Set the values of the gaussian and parabola
     for (int ipar=0;ipar<3;ipar++){
         gaussian.SetParameter(ipar,
                               gausppar.GetParameter(ipar));
         parabola.SetParameter(ipar,
                               gausppar.GetParameter(ipar+3));}

     histo.GetYaxis()->SetRangeUser(0,250);
     histo.DrawClone("PE");
     parabola.DrawClone("Same"); gaussian.DrawClone("Same");
     TLatex latex(2,220,
                  "#splitline{Signal Peak over}{background}");
     latex.DrawClone("Same");
 }
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

-   Lines *27-32*: Define and fill a histogram.

-   Lines *34-40*: For convenience, the same function as for the
    generation of the pseudo-data is used in the fit; hence, we need to
    reset the function parameters. This part of the code is very
    important for each fit procedure, as it sets the initial values of
    the fit.

-   Line *43*: A very simple command, well known by now: fit the
    function to the histogram.

-   Lines *45-49*: Retrieve the output from the fit. Here, we simply
    print the fit result and access and print the covariance matrix of
    the parameters.

-   Lines *58-end*: Plot the pseudo-data, the fitted function and the
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
 // Toy Monte Carlo example.
 // Check pull distribution to compare chi2 and binned
 // log-likelihood methods.

 pull( int n_toys = 10000,
       int n_tot_entries = 100,
       int nbins = 40,
       bool do_chi2=true ){

     TString method_prefix("Log-Likelihood ");
     if (do_chi2)
         method_prefix="#chi^{2} ";

     // Create histo
     TH1F* h4 = new TH1F(method_prefix+"h4",
                         method_prefix+" Random Gauss",
                         nbins,-4,4);
     h4->SetMarkerStyle(21);
     h4->SetMarkerSize(0.8);
     h4->SetMarkerColor(kRed);

     // Histogram for sigma and pull
     TH1F* sigma = new TH1F(method_prefix+"sigma",
                            method_prefix+"sigma from gaus fit",
                            50,0.5,1.5);
     TH1F* pull = new TH1F(method_prefix+"pull",
                           method_prefix+"pull from gaus fit",
                           50,-4.,4.);

     // Make nice canvases
     TCanvas* c0 = new TCanvas(method_prefix+"Gauss",
                             method_prefix+"Gauss",0,0,320,240);
     c0->SetGrid();

     // Make nice canvases
     TCanvas* c1 = new TCanvas(method_prefix+"Result",
                             method_prefix+"Sigma-Distribution",
                             0,300,600,400);
     c0->cd();

     float sig, mean;
     for (int i=0; i<n_toys; i++){
      // Reset histo contents
         h4->Reset();
      // Fill histo
         for ( int j = 0; j<n_tot_entries; j++ )
         h4->Fill(gRandom->Gaus());
      // perform fit
         if (do_chi2) h4->Fit("gaus","q"); // Chi2 fit
         else h4->Fit("gaus","lq"); // Likelihood fit
      // some control output on the way
         if (!(i%100)){
             h4->Draw("ep");
             c0->Update();}

      // Get sigma from fit
         TF1 *fit = h4->GetFunction("gaus");
         sig = fit->GetParameter(2);
         mean= fit->GetParameter(1);
         sigma->Fill(sig);
         pull->Fill(mean/sig * sqrt(n_tot_entries));
        } // end of toy MC loop
      // print result
         c1->cd();
         pull->Draw();
 }

 void macro9(){
     int n_toys=10000;
     int n_tot_entries=100;
     int n_bins=40;
     cout << "Performing Pull Experiment with chi2 \n";
     pull(n_toys,n_tot_entries,n_bins,true);
     cout << "Performing Pull Experiment with Log Likelihood\n";
     pull(n_toys,n_tot_entries,n_bins,false);
 }

```

Your present knowledge of ROOT should be enough to understand all the
technicalities behind the macro. Note that the variable `pull` in line
*59* is different from the definition above: instead of the parameter
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

[^4]: "Monte Carlo" simulation means that random numbers play a role here
which is as crucial as in games of pure chance in the Casino of Monte Carlo.
