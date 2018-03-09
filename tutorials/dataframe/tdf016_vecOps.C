/// \file
/// \ingroup tutorial_tdataframe
/// \notebook -draw
/// This tutorial shows the potential of the VecOps approach for treating collections
/// stored in datasets, a situation very common in HEP data analysis.
///
/// \macro_code
///
/// \date February 2018
/// \author Danilo Piparo

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::VecOps;

int tdf016_vecOps()
{
   // We re-create a set of points in a square.
   // This is a technical detail, just to create a dataset to play with!
   auto unifGen = [](double) { return gRandom->Uniform(-1.0, 1.0); };
   auto vGen = [&](int len) {
      TVec<double> v(len);
      std::transform(v.begin(), v.end(), v.begin(), unifGen);
      return v;
   };
   TDataFrame d(1024);
   auto d0 = d.Define("len", []() { return (int)gRandom->Uniform(0, 16); })
      .Define("x", vGen, {"len"})
      .Define("y", vGen, {"len"});

   // Now we have in hands d, a TDataFrame with two columns, x and y, which
   // hold collections of coordinates. The size of these collections vary.
   // Let's now define radii out of x and y. We'll do it treating the collections
   // stored in the columns without looping on the individual elements.
   auto d1 = d0.Define("r", "sqrt(x*x + y*y)");

   // Now we want to plot 2 quarters of a ring with radii .5 and 1
   // Note how the cuts are performed on TVecs, comparing them with integers and
   // among themselves
   auto ring_h = d1.Define("rInFig", "r > .4 && r < .8 && x*y < 0")
                    .Define("yFig", "y[rInFig]")
                    .Define("xFig", "x[rInFig]")
                    .Histo2D({"fig", "Two quarters of a ring", 64, -1, 1, 64, -1, 1}, "xFig", "yFig");

   TCanvas cring;
   ring_h->Draw("Colz");
   cring.Print("tdf016_vecOps.png");

   return 0;
}
