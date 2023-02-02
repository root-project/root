// Tests for the ParamHistFunc
// Authors: Jonas Rembser, CERN  08/2022

#include <RooArgSet.h>
#include <RooArgSet.h>
#include <RooConstVar.h>
#include <RooDataSet.h>
#include <RooRandom.h>
#include <RooRealVar.h>
#include <RooStats/HistFactory/ParamHistFunc.h>

#include <gtest/gtest.h>

/// Validate the ParamHistFunc in the n-dimensional case, comparing both the
/// BatchMode and the old implementation results to a manually-compute
/// reference result.
TEST(ParamHistFunc, ValidateND)
{

   // Define the number of bins in each dimension
   std::array<std::size_t, 3> nbins{{11, 32, 8}};
   std::size_t nbinstot = nbins[0] * nbins[1] * nbins[2];

   // The bin mltiplication factors to look up the right parameters
   std::array<std::size_t, 3> binMult{{1, nbins[0], nbins[0] * nbins[1]}};

   // Create the variables and set their bin numbers. The range is tweaked
   // such that each integer values falls in a different bin, starting from
   // zero.
   RooRealVar x{"x", "x", -0.5, nbins[0] - 0.5};
   RooRealVar y{"y", "y", -0.5, nbins[1] - 0.5};
   RooRealVar z{"z", "z", -0.5, nbins[2] - 0.5};
   RooArgSet vars{x, y, z};
   for (std::size_t i = 0; i < nbins.size(); ++i) {
      static_cast<RooRealVar &>(*vars[i]).setBins(nbins[i]);
   }

   // Simple set of parameters that just return their index in the parameter
   // list
   RooArgList params;
   for (std::size_t i = 0; i < nbinstot; ++i) {
      params.add(RooFit::RooConst(i));
   }

   ParamHistFunc paramHistFunc{"phf", "phf", vars, params};

   std::size_t nEntries = 100;

   RooDataSet data{"data", "data", vars};
   std::vector<double> resultsRef(nEntries);
   std::vector<double> resultsScalar(nEntries);

   // Do some things in one go:
   //   * assing random integer values to each variable in each iteration
   //   * fill the dataset used for batched evaluation
   //   * compute the reference result manually
   //   * compute the result with the ParamHistFunc without BatchMode
   for (std::size_t i = 0; i < nEntries; ++i) {
      for (std::size_t iVar = 0; iVar < vars.size(); ++iVar) {
         auto var = static_cast<RooRealVar *>(vars[iVar]);
         var->setVal(int(RooRandom::uniform() * nbins[iVar]));
      }
      data.add(vars);
      resultsRef[i] = binMult[0] * x.getVal() + binMult[1] * y.getVal() + binMult[2] * z.getVal();
      resultsScalar[i] = paramHistFunc.getVal();
   }

   // Get the results in BatchMode using the dataset
   auto resultsBatch = paramHistFunc.getValues(data);

   // Validate the results
   for (std::size_t i = 0; i < nEntries; ++i) {
      EXPECT_EQ(int(resultsScalar[i]), int(resultsRef[i])) << "Scalar result is not correct!";
      EXPECT_EQ(int(resultsBatch[i]), int(resultsRef[i])) << "BatchMode result is not correct!";
   }
}
