/// \file
/// \ingroup tutorial_hist
/// \notebook
/// Fill a 1D histogram from a user-defined parametric function.
///
/// \macro_image
/// \macro_code
///
/// \date November 2024
/// \author Rene Brun, Giacomo Parolini

void hist002_TH1_fillrandom_userfunc()
{
   // Create a user-defined formula.
   // A function (any dimension) or a formula may reference an already defined formula
   TFormula form1("form1", "abs(sin(x)/x)");

   // Create a 1D function using the formula defined above and the predefined "gaus" formula.
   double rangeMin = 0.0;
   double rangeMax = 10.0;
   TF1 sqroot("sqroot", "x*gaus(0) + [3]*form1", rangeMin, rangeMax);
   sqroot.SetLineColor(4);
   sqroot.SetLineWidth(6);
   // Set parameters to the functions "gaus" and "form1".
   double gausScale = 10.0;  // [0]
   double gausMean = 4.0;    // [1]
   double gausVar = 1.0;     // [2]
   double form1Scale = 20.0; // [3]
   sqroot.SetParameters(gausScale, gausMean, gausVar, form1Scale);

   // Create a one dimensional histogram and fill it following the distribution in function sqroot.
   int nBins = 200;
   TH1D h1d("h1d", "Test random numbers", nBins, rangeMin, rangeMax);

   // Use our user-defined function to fill the histogram with random values sampled from it.
   h1d.FillRandom("sqroot", 10000);

   // Open a ROOT file and save the formula, function and histogram
   auto myFile = std::unique_ptr<TFile>(TFile::Open("fillrandom_userfunc.root", "RECREATE"));
   myFile->WriteObject(&form1, form1.GetName());
   myFile->WriteObject(&sqroot, sqroot.GetName());
   myFile->WriteObject(&h1d, h1d.GetName());
}
