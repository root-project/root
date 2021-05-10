/// \file
/// \ingroup tutorial_v7
///
/// This ROOT 7 example demonstrates how to use RNTuple in combination with ROOT 6 features like RDataframe and visualizations.
/// It ingests climate data and creates a model with fields like AverageTemperature. Then it uses RDataframe to process and filter the climate data for
/// average temperature per city by season. Then it does the same for average temperature per city for the years between 1993-2002, and 2003-2013.
/// Finally, the tutorial visualizes this processed data through histograms.
///
/// \macro_image (json)
/// \macro_code
///
///
/// NOTE: Until C++ runtime modules are universally used, we explicitly load the ntuple library.  Otherwise
/// triggering autoloading from the use of templated types would require an exhaustive enumeration
/// of "all" template instances in the LinkDef file.
///
/// \warning The RNTuple classes are experimental at this point.
/// Functionality, interface, and data format is still subject to changes.
/// Do not use for real data! During ROOT setup, configure the following flags:
/// `-DCMAKE_CXX_STANDARD=14 -Droot7=ON -Dwebgui=ON`
///
/// \date 2021-02-26
/// \author John Yoon

R__LOAD_LIBRARY(ROOTNTuple)
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDS.hxx>
#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RCanvas.hxx>
#include <ROOT/RColor.hxx>
#include <ROOT/RHistDrawable.hxx>
#include <ROOT/TObjectDrawable.hxx>
#include <ROOT/RRawFile.hxx>
#include <TH1D.h>
#include <TLegend.h>
#include <TSystem.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;
using RRawFile = ROOT::Internal::RRawFile;
using namespace ROOT::Experimental;

// Helper function to handle histogram pointer ownership.
std::shared_ptr<TH1D> GetDrawableHist(ROOT::RDF::RResultPtr<TH1D> &h) {
   auto result = std::shared_ptr<TH1D>(static_cast<TH1D *>(h.GetPtr()->Clone()));
   result->SetDirectory(nullptr);
   return result;
}

// Climate data is downloadable at the following URL:
// https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data
// The original data set is from http://berkeleyearth.org/archive/data/
// License CC BY-NC-SA 4.0
constexpr const char *kRawDataUrl = "http://root.cern./files/tutorials/GlobalLandTemperaturesByCity.csv";
constexpr const char *kNTupleFileName = "GlobalLandTemperaturesByCity.root";

void Ingest() {
   int nRecords = 0;
   int nSkipped = 0;
   std::cout << "Converting " << kRawDataUrl << " to " << kNTupleFileName << std::endl;

   auto t1 = Clock::now();

   // Create a unique pointer to an empty data model.
   auto model = RNTupleModel::Create();
   // To define the data model, create fields with a given C++ type and name.  Fields are roughly TTree branches.
   // MakeField returns a shared pointer to a memory location to fill the ntuple with data.
   auto fieldYear       = model->MakeField<std::uint32_t>("Year");
   auto fieldMonth      = model->MakeField<std::uint32_t>("Month");
   auto fieldDay        = model->MakeField<std::uint32_t>("Day");
   auto fieldAvgTemp    = model->MakeField<float>("AverageTemperature");
   auto fieldTempUncrty = model->MakeField<float>("AverageTemperatureUncertainty");
   auto fieldCity       = model->MakeField<std::string>("City");
   auto fieldCountry    = model->MakeField<std::string>("Country");
   auto fieldLat        = model->MakeField<float>("Latitude");
   auto fieldLong       = model->MakeField<float>("Longitude");

   // Hand-over the data model to a newly created ntuple of name "globalTempData", stored in kNTupleFileName.
   // In return, get a unique pointer to a fillable ntuple (first compress the file).
   RNTupleWriteOptions options;
   options.SetCompression(ROOT::RCompressionSetting::EDefaults::kUseGeneralPurpose);
   auto ntuple = RNTupleWriter::Recreate(std::move(model), "GlobalTempData", kNTupleFileName, options);

   auto file = RRawFile::Create(kRawDataUrl);
   std::string record;
   constexpr int kMaxCharsPerLine = 128;
   while (file->Readln(record)) {
      if (record.length() >= kMaxCharsPerLine)
         throw std::runtime_error("record too long: " + record);

      // Parse lines of the form:
      // 1743-11-01,6.068,1.7369999999999999,Århus,Denmark,57.05N,10.33E
      // and skip records with empty fields.
      std::replace(record.begin(), record.end(), ',', ' ');
      char country[kMaxCharsPerLine];
      char city[kMaxCharsPerLine];
      int nFields = sscanf(record.c_str(), "%u-%u-%u %f %f %s %s %fN %fE",
                           fieldYear.get(), fieldMonth.get(), fieldDay.get(),
                           fieldAvgTemp.get(), fieldTempUncrty.get(), country, city,
                           fieldLat.get(), fieldLong.get());
      if (nFields != 9) {
         nSkipped++;
         continue;
      }
      *fieldCountry = country;
      *fieldCity = city;

      ntuple->Fill();

      if (++nRecords % 1000000 == 0)
         std::cout << "  ... converted " << nRecords << " records" << std::endl;
   }

   // Display the total time to process the data.
   std::cout << nSkipped << " records skipped" << std::endl;
   std::cout << nRecords << " records processed" << std::endl;

   auto t2 = Clock::now();
   std::cout << std::endl
             << "Processing Time: "
             << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()
             << " seconds\n" << std::endl;
}

// Every data result that we want to get is declared first, and it is only upon their declaration that
// they are actually used. This stems from motivations relating to efficiency and optimization.
void Analyze() {
   // Create a RDataframe by wrapping around NTuple.
   auto df = ROOT::Experimental::MakeNTupleDataFrame("GlobalTempData", kNTupleFileName);
   df.Display()->Print();

   // Declare the minimum and maximum temperature from the dataset.
   auto min_value = df.Min("AverageTemperature");
   auto max_value = df.Max("AverageTemperature");

   // Functions to filter by each season from date formatted "1944-12-01."
   auto fnWinter = [](int month) { return month == 12 || month == 1  || month == 2;  };
   auto fnSpring = [](int month) { return month == 3  || month == 4  || month == 5;  };
   auto fnSummer = [](int month) { return month == 6  || month == 7  || month == 8;  };
   auto fnFall   = [](int month) { return month == 9  || month == 10 || month == 11; };

   // Create a RDataFrame per season.
   auto dfWinter = df.Filter(fnWinter, {"Month"});
   auto dfSpring = df.Filter(fnSpring, {"Month"});
   auto dfSummer = df.Filter(fnSummer, {"Month"});
   auto dfFall = df.Filter(fnFall, {"Month"});

   // Get the count for each season.
   auto winterCount = dfWinter.Count();
   auto springCount = dfSpring.Count();
   auto summerCount = dfSummer.Count();
   auto fallCount = dfFall.Count();

   // Functions to filter for the time period between 2003-2013, and 1993-2002.
   auto fn1993_to_2002 = [](int year) { return year >= 1993 && year <= 2002; };
   auto fn2003_to_2013 = [](int year) { return year >= 2003 && year <= 2013; };

   // Create a RDataFrame for decades 1993_to_2002 & 2003_to_2013.
   auto df1993_to_2002 = df.Filter(fn1993_to_2002, {"Year"});
   auto df2003_to_2013 = df.Filter(fn2003_to_2013, {"Year"});

   // Get the count for each decade.
   auto decade_1993_to_2002_Count = *df1993_to_2002.Count();
   auto decade_2003_to_2013_Count = *df2003_to_2013.Count();

   // Configure histograms for each season.
   auto fallHistResultPtr = dfFall.Histo1D({"Fall Average Temp", "Average Temperature by Season", 100, -40, 40}, "AverageTemperature");
   auto winterHistResultPtr = dfWinter.Histo1D({"Winter Average Temp", "Average Temperature by Season", 100, -40, 40}, "AverageTemperature");
   auto springHistResultPtr = dfSpring.Histo1D({"Spring Average Temp", "Average Temperature by Season", 100, -40, 40}, "AverageTemperature");
   auto summerHistResultPtr = dfSummer.Histo1D({"Summer Average Temp", "Average Temperature by Season", 100, -40, 40}, "AverageTemperature");

   // Configure histograms for each decade.
   auto hist_1993_to_2002_ResultPtr = df1993_to_2002.Histo1D({"1993_to_2002 Average Temp", "Average Temperature: 1993_to_2002 vs. 2003_to_2013", 100, -40, 40}, "AverageTemperature");
   auto hist_2003_to_2013_ResultPtr = df2003_to_2013.Histo1D({"2003_to_2013 Average Temp", "Average Temperature: 1993_to_2002 vs. 2003_to_2013", 100, -40, 40}, "AverageTemperature");

   //____________________________________________________________________________________

   // Display the minimum and maximum temperature values.
   std::cout << std::endl << "The Minimum temperature is: " << *min_value << std::endl;
   std::cout << "The Maximum temperature is: " << *max_value << std::endl;

   // Display the count for each season.
   std::cout << std::endl << "The count for Winter: " << *winterCount<< std::endl;
   std::cout << "The count for Spring: " << *springCount << std::endl;
   std::cout << "The count for Summer: " << *summerCount << std::endl;
   std::cout << "The count for Fall: " << *fallCount << std::endl;

   // Display the count for each decade.
   std::cout << std::endl << "The count for 1993_to_2002: " << decade_1993_to_2002_Count << std::endl;
   std::cout << "The count for 2003_to_2013: " <<decade_2003_to_2013_Count << std::endl;

   // Transform histogram in order to address ROOT 7 v 6 version compatibility
   auto fallHist = GetDrawableHist(fallHistResultPtr);
   auto winterHist = GetDrawableHist(winterHistResultPtr);
   auto springHist = GetDrawableHist(springHistResultPtr);
   auto summerHist  = GetDrawableHist(summerHistResultPtr);

   // Set an orange histogram for fall.
   fallHist->SetLineColor(kOrange);
   fallHist->SetLineWidth(6);
   // Set a blue histogram for winter.
   winterHist->SetLineColor(kBlue);
   winterHist->SetLineWidth(6);
   // Set a green histogram for spring.
   springHist->SetLineColor(kGreen);
   springHist->SetLineWidth(6);
   // Set a red histogram for summer.
   summerHist->SetLineColor(kRed);
   summerHist->SetLineWidth(6);

   // Transform histogram in order to address ROOT 7 v 6 version compatibility
   auto hist_1993_to_2002 = GetDrawableHist(hist_1993_to_2002_ResultPtr);
   auto hist_2003_to_2013 = GetDrawableHist(hist_2003_to_2013_ResultPtr);

   // Set a violet histogram for 1993_to_2002.
   hist_1993_to_2002->SetLineColor(kViolet);
   hist_1993_to_2002->SetLineWidth(6);
   // Set a spring-green histogram for 2003_to_2013.
   hist_2003_to_2013->SetLineColor(kSpring);
   hist_2003_to_2013->SetLineWidth(6);


   // Create a canvas to display histograms for average temperature by season.
   auto canvas = RCanvas::Create("Average Temperature by Season");
   canvas->Draw<TObjectDrawable>(fallHist, "L");
   canvas->Draw<TObjectDrawable>(winterHist, "L");
   canvas->Draw<TObjectDrawable>(springHist, "L");
   canvas->Draw<TObjectDrawable>(summerHist, "L");

   // Create a legend for the seasons canvas.
   auto legend = std::make_shared<TLegend>(0.15,0.65,0.53,0.85);
   legend->AddEntry(fallHist.get(),"fall","l");
   legend->AddEntry(winterHist.get(),"winter","l");
   legend->AddEntry(springHist.get(),"spring","l");
   legend->AddEntry(summerHist.get(),"summer","l");
   canvas->Draw<TObjectDrawable>(legend, "L");
   canvas->Show();

   // Create a canvas to display histograms for average temperature for 1993_to_2002 & 2003_to_2013.
   auto canvas2 = RCanvas::Create("Average Temperature: 1993_to_2002 vs. 2003_to_2013");
   canvas2->Draw<TObjectDrawable>(hist_1993_to_2002, "L");
   canvas2->Draw<TObjectDrawable>(hist_2003_to_2013, "L");

   // Create a legend for the two decades canvas.
   auto legend2 = std::make_shared<TLegend>(0.1,0.7,0.48,0.9);
   legend2->AddEntry(hist_1993_to_2002.get(),"1993_to_2002","l");
   legend2->AddEntry(hist_2003_to_2013.get(),"2003_to_2013","l");
   canvas2->Draw<TObjectDrawable>(legend2, "L");
   canvas2->Show();
}

void global_temperatures() {
   //if NOT zero (the file does NOT already exist), then Ingest
   if (gSystem->AccessPathName(kNTupleFileName) != 0) {
      Ingest();

   }
   Analyze();
}