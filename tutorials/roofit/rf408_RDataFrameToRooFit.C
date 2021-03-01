/// \file
/// \ingroup tutorial_roofit
/// \notebook
/// Fill RooDataSet/RooDataHist in RDataFrame.
///
/// This tutorial shows how to fill RooFit data classes directly from RDataFrame.
/// Using two small helpers, we tell RDataFrame where the data has to go.
///
/// \macro_code
/// \macro_output
///
/// \date Mar 2021
/// \author Stephan Hageboeck (CERN)

#include <RooAbsDataHelper.h>

#include <TROOT.h>
#include <TRandom.h>

#include <initializer_list>

void rf408_RDataFrameToRooFit()
{
  // Set up
  // ------------------------

  // We enable implicit parallelism, so RDataFrame runs in parallel.
  ROOT::EnableImplicitMT();

  // We create an RDataFrame with two columns filled with 2 million random numbers.
  ROOT::RDataFrame d(2000000);
  auto dd = d.Define("x", [](){ return gRandom->Uniform(-5.,  5.); })
             .Define("y", [](){ return gRandom->Gaus(1., 3.); });


  // We create RooFit variables that will represent the dataset.
  RooRealVar x("x", "x", -5.,   5.);
  RooRealVar y("y", "y", -50., 50.);
  x.setBins(10);
  y.setBins(20);



  // Booking the creation of RooDataSet / RooDataHist in RDataFrame
  // ----------------------------------------------------------------

  // Method 1:
  // We directly book the RooDataSetMaker action.
  // We need to pass
  // - the RDataFrame column types as template parameters
  // - the constructor arguments for RooDataSet (they follow the same syntax as the usual RooDataSet constructors)
  // - the column names that RDataFrame should fill into the dataset
  //
  // NOTE: RDataFrame columns are matched to RooFit variables by position, *not by name*!
  auto rooDataSet = dd.Book<double, double>(
      RooDataSetHelper("dataset", // Name
          "Title of dataset",     // Title
          RooArgSet(x, y)         // Variables in this dataset
          ),
      {"x", "y"}                  // Column names in RDataFrame.
  );


  // Method 2:
  // We first declare the RooDataHistMaker
  RooDataHistHelper rdhMaker{"datahist",  // Name
    "Title of data hist",                 // Title
    RooArgSet(x, y)                       // Variables in this dataset
  };

  // Then, we move it into the RDataFrame action:
  auto rooDataHist = dd.Book<double, double>(std::move(rdhMaker), {"x", "y"});



  // Run it and inspect the results
  // -------------------------------

  // Let's inspect the dataset / datahist.
  // Note that the first time we touch one of those objects, the RDataFrame event loop will run.
  for (const RooAbsData* data : std::initializer_list<const RooAbsData*>{rooDataSet.GetPtr(), rooDataHist.GetPtr()} ) {
    std::cout << std::endl;
    data->Print();

    for (int i=0; i < data->numEntries() && i < 20; ++i) {
      std::cout << "(";
      for (auto var : *data->get(i)) {
        std::cout << std::setprecision(3) << std::right << std::fixed << std::setw(8) << static_cast<const RooAbsReal*>(var)->getVal() << ", ";
      }
      std::cout << ")\tweight=" << std::setw(10) << data->weight() << std::endl;
    }

    std::cout << "mean(x) = " << data->mean(x) << "\tsigma(x) = " << std::sqrt(data->moment(x, 2.))
      << "\n" << "mean(y) = " << data->mean(y) << "\tsigma(y) = " << std::sqrt(data->moment(y, 2.)) << std::endl;
  }
}

int main() {
  rf408_RDataFrameToRooFit();
  return 0;
}
