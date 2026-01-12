/// \file
/// \ingroup tutorial_histv7
///
/// Multidimensional RHist with different axis types.
///
/// \macro_code
/// \macro_output
///
/// \date January 2026
/// \author The ROOT Team

#include <ROOT/RBinIndex.hxx>
#include <ROOT/RHist.hxx>
#include <ROOT/RRegularAxis.hxx>
#include <ROOT/RVariableBinAxis.hxx>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

void hist003_RHist_multi()
{
   // Define the bin edges and variable bin axis for the eta dimension.
   std::vector<double> etaEdges = {-5.0, -3.0, -2.4, -1.5, 0.0, 1.5, 2.4, 3.0, 5.0};
   ROOT::Experimental::RVariableBinAxis etaAxis(etaEdges);

   // Define the regular axis for the phi dimension (in radians), disabling under- and overflow bins.
   static constexpr double Pi = 3.14159;
   ROOT::Experimental::RRegularAxis phiAxis(20, {-Pi / 2, Pi / 2}, false);

   // Define the logarithmic axis for the energy (in GeV).
   ROOT::Experimental::RVariableBinAxis energyAxis({1, 10, 100, 1000});

   // Create the multidimensional histogram.
   ROOT::Experimental::RHist<int> hist(etaAxis, phiAxis, energyAxis);

   // Print some information about the histogram.
   std::cout << "Created a histogram with " << hist.GetNDimensions() << " dimensions and " << hist.GetTotalNBins()
             << "\n";

   // Generate angles and energies to fill the histogram.
   std::mt19937 gen;
   std::uniform_real_distribution dist(0.0, 1.0);
   static constexpr double maxE = 1050;
   for (std::size_t i = 0; i < 100000; i++) {
      double delta = 2 * Pi * dist(gen);
      double eta = -std::log(std::tan(delta / 2));
      double phi = std::acos(2 * dist(gen) - 1) - Pi / 2;
      double E = maxE * dist(gen);
      hist.Fill(eta, phi, E);
   }
   std::cout << " ... filled with " << hist.GetNEntries() << " entries.\n\n";

   // Get the content for specific bins:
   std::cout << "eta in [1.5, 2.4), phi in [0, 0.157), E between 10 GeV and 100 GeV\n";
   auto content = hist.GetBinContent(5, 10, 1);
   std::cout << "  content = " << content << "\n";

   std::cout << "eta in [-1.5, 0), phi in [-0.785, -0.628), E between 1 GeV and 10 GeV\n";
   content = hist.GetBinContent(3, 5, 0);
   std::cout << "  content = " << content << "\n";
}
