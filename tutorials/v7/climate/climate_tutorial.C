// Tutorial that highlights ROOT 7 features, including: RNTuple, RDataframe, and visualizations.

// The tutorial first uses RNTuple to ingest climate data and create a model with fields like
// AverageTemperature. Then it uses RDataframe to process and filter the climate data for
// average temperature per city by season. Then it does the same for average temperature
// per city for the years between 1993-2002, and 2003-2013. Finally, the tutorial
// visualizes this processed data through histograms.

// During ROOT setup, configure the following flags: "-DCMAKE_CXX_STANDARD=14 -Droot7=ON -Dwebgui=ON"

// NOTE: The RNTuple classes are experimental at this point.
// Functionality, interface, and data format is still subject to changes.
// Do not use for real data!

// Until C++ runtime modules are universally used, we explicitly load the ntuple library.  Otherwise
// triggering autoloading from the use of templated types would require an exhaustive enumeration
// of "all" template instances in the LinkDef file.
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
#include <cstdio> // for sscanf()
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

// Climate data is downloadable at the followink URL:
// https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data
// The original data set is from http://berkeleyearth.org/archive/data/
// License CC BY-NC-SA 4.0
constexpr char const* kRawDataUrl = "http://root.cern./files/tutorials/GlobalLandTemperaturesByCity.csv";
constexpr char const* kNTupleFileName = "GlobalLandTemperaturesByCity.root";

void Ingest() {
   int nRecords = 0;
   int nSkipped = 0;
   std::cout << "Converting " << kRawDataUrl << " to " << kNTupleFileName << std::endl;

   auto t1 = Clock::now();

   // Create a unique pointer to an empty data model
   auto model = RNTupleModel::Create();
   // To define the data model, create fields with a given C++ type and name.  Fields are roughly TTree branches.
   // MakeField returns a shared pointer to a memory location to fill the ntuple with data
   auto fieldYear       = model->MakeField<std::uint32_t>("Year");
   auto fieldMonth      = model->MakeField<std::uint32_t>("Month");
   auto fieldDay        = model->MakeField<std::uint32_t>("Day");
   auto fieldAvgTemp    = model->MakeField<float>("AverageTemperature");
   auto fieldTempUncrty = model->MakeField<float>("AverageTemperatureUncertainty");
   auto fieldCity       = model->MakeField<std::string>("City");
   auto fieldCountry    = model->MakeField<std::string>("Country");
   auto fieldLat        = model->MakeField<float>("Latitude");
   auto fieldLong       = model->MakeField<float>("Longitude");

   // Hand-over the data model to a newly created ntuple of name "globalTempData", stored in kNTupleFileName
   // In return, get a unique pointer to a fillable ntuple (first compress the file)
   RNTupleWriteOptions options;
   options.SetCompression(ROOT::RCompressionSetting::EDefaults::kUseGeneralPurpose);
   auto ntuple = RNTupleWriter::Recreate(std::move(model), "GlobalTempData", kNTupleFileName, options);

   auto file = RRawFile::Create(kRawDataUrl);
   std::string record;
   constexpr int kMaxCharsPerLine = 128;
   while (file->Readln(record)) {
      if (record.length() >= kMaxCharsPerLine)
         throw std::runtime_error("record too long: " + record);

      // Parse lines of the form
      // 1743-11-01,6.068,1.7369999999999999,Århus,Denmark,57.05N,10.33E
      // and skip records with empty fields
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

   // Display the total time to process the data
   std::cout << nSkipped << " records skipped" << std::endl;
   std::cout << nRecords << " records processed" << std::endl;

   auto t2 = Clock::now();
   std::cout << std::endl
             << "Processing Time: "
             << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()
             << " seconds\n" << std::endl;
}


void Analyze() {
   // Get a unique pointer to an empty RNTuple model
   auto model = RNTupleModel::Create();
   // Define only the necessary fields for reading
   model->MakeField<float>("AverageTemperature");
   model->MakeField<float>("AverageTemperatureUncertainty");
   model->MakeField<std::uint32_t>("Year");
   model->MakeField<std::uint32_t>("Month");
   model->MakeField<std::uint32_t>("Day");
   model->MakeField<std::string>("City");

   // Create an ntuple and attach the read model to it
   auto ntuple = RNTupleReader::Open(std::move(model), "GlobalTempData", kNTupleFileName);
   // Overview of the ntuple and list of fields.
   ntuple->PrintInfo();
   std::cout << std::endl
             << "The tenth entry in JSON format:" << std::endl;
   ntuple->Show(9);

   // Create a RDataframe by wrapping around NTuple
   auto df = ROOT::Experimental::MakeNTupleDataFrame("GlobalTempData", kNTupleFileName);
   // Display the minimum and maximum temperature from the dataset
   auto minimum_value = df.Min("AverageTemperature");
   cout << "\nThe Minimum temperature is: " << *minimum_value << std::endl;
   auto max_value = df.Max("AverageTemperature");
   cout << "The Maximum temperature is: " << *max_value << std::endl;

   // Functions to filter by each season from date formatted "1944-12-01"
   auto fnWinter = [](int month) { return month == 12 || month == 1  || month == 2;  };
   auto fnSpring = [](int month) { return month == 3  || month == 4  || month == 5;  };
   auto fnSummer = [](int month) { return month == 6  || month == 7  || month == 8;  };
   auto fnFall   = [](int month) { return month == 9  || month == 10 || month == 11; };

   // Create a RDataFrame per season by filtering ntupleToDF and display the count for each season
   auto dfWinter = df.Filter(fnWinter, {"Month"});
   cout << "\nThe count for Winter: " << *dfWinter.Count() << std::endl;
   auto dfSpring = df.Filter(fnSpring, {"Month"});
   cout << "The count for Spring: " << *dfSpring.Count() << std::endl;
   auto dfSummer = df.Filter(fnSummer, {"Month"});
   cout << "The count for Summer: " << *dfSummer.Count() << std::endl;
   auto dfFall = df.Filter(fnFall, {"Month"});
   cout << "The count for Fall: " << *dfFall.Count() << std::endl;

   // Create an orange histogram for fall
   auto fallHist = dfFall.Histo1D({"Fall Average Temp", "Average Temperature by Season", 100, -40, 40}, "AverageTemperature");
   fallHist->SetLineColor(kOrange);
   fallHist->SetLineWidth(6);
   // Create a blue histogram for winter
   auto winterHist = dfWinter.Histo1D({"Winter Average Temp", "Average Temperature by Season", 100, -40, 40}, "AverageTemperature");
   winterHist->SetLineColor(kBlue);
   winterHist->SetLineWidth(6);
   // Create a green histogram for spring
   auto springHist = dfSpring.Histo1D({"Spring Average Temp", "Average Temperature by Season", 100, -40, 40}, "AverageTemperature");
   springHist->SetLineColor(kGreen);
   springHist->SetLineWidth(6);
   // Create a red histogram for summer
   auto summerHist = dfSummer.Histo1D({"Summer Average Temp", "Average Temperature by Season", 100, -40, 40}, "AverageTemperature");
   summerHist->SetLineColor(kRed);
   summerHist->SetLineWidth(6);

   // Functions to filter for the time period between 2003-2013, and 1993-2002
   auto fn1993To2002 = [](int year) { return year >= 1993 && year <= 2002; };
   auto fn2003To2013 = [](int year) { return year >= 2003 && year <= 2013; };

   // Create a RDataFrame for decades 1993_to_2002 & 2003_to_2013 by filtering ntupleToDF
   // Display the count for each decade
   auto df1993To2002 = df.Filter(fn1993To2002, {"Year"});
   cout << "\nThe count for 1993_to_2002: " << *df1993To2002.Count() << std::endl;
   auto df2003To2013 = df.Filter(fn2003To2013, {"Year"});
   cout << "The count for 2003_to_2013: " << *df2003To2013.Count() << std::endl;

   // Create a violet histogram for 1993_to_2002
   auto hist_1993_to_2002 = df1993To2002.Histo1D({"1993_to_2002 Average Temp", "Average Temperature: 1993_to_2002 vs. 2003_to_2013", 100, -40, 40}, "AverageTemperature");
   hist_1993_to_2002->SetLineColor(kViolet);
   hist_1993_to_2002->SetLineWidth(6);
   // Create a spring-green histogram for 2003_to_2013
   auto hist_2003_to_2013 = df2003To2013.Histo1D({"2003_to_2013 Average Temp", "Average Temperature: 1993_to_2002 vs. 2003_to_2013", 100, -40, 40}, "AverageTemperature");
   hist_2003_to_2013->SetLineColor(kSpring);
   hist_2003_to_2013->SetLineWidth(6);

   // Create a canvas to display histograms for average temperature by season.
   auto canvas = RCanvas::Create("Average Temperature by Season");
   canvas->Draw<TObjectDrawable>(std::shared_ptr<TH1D>(fallHist.GetPtr()), "L");
   canvas->Draw<TObjectDrawable>(std::shared_ptr<TH1D>(winterHist.GetPtr()), "L");
   canvas->Draw<TObjectDrawable>(std::shared_ptr<TH1D>(springHist.GetPtr()), "L");
   canvas->Draw<TObjectDrawable>(std::shared_ptr<TH1D>(summerHist.GetPtr()), "L");
   // Create a legend for the seasons canvas
   auto legend = std::make_shared<TLegend>(0.1,0.7,0.48,0.9);
   legend->AddEntry(fallHist.GetPtr(),"fall","l");
   legend->AddEntry(winterHist.GetPtr(),"winter","l");
   legend->AddEntry(springHist.GetPtr(),"spring","l");
   legend->AddEntry(summerHist.GetPtr(),"summer","l");
   canvas->Draw<TObjectDrawable>(legend, "L");
   canvas->Show();

   // Create a canvas to display histograms for average temperature for 1993_to_2002 & 2003_to_2013.
   auto canvas2 = RCanvas::Create("Average Temperature: 1993_to_2002 vs. 2003_to_2013");
   canvas2->Draw<TObjectDrawable>(std::shared_ptr<TH1D>(hist_1993_to_2002.GetPtr()), "L");
   canvas2->Draw<TObjectDrawable>(std::shared_ptr<TH1D>(hist_2003_to_2013.GetPtr()), "L");
   // Create a legend for the two decades canvas
   auto legend2 = std::make_shared<TLegend>(0.1,0.7,0.48,0.9);
   legend2->AddEntry(hist_1993_to_2002.GetPtr(),"1993_to_2002","l");
   legend2->AddEntry(hist_2003_to_2013.GetPtr(),"2003_to_2013","l");
   canvas2->Draw<TObjectDrawable>(legend2, "L");
   canvas2->Show();

   // Logic to reconcile applying ROOT 6 histograms on a ROOT 7 canvas
   // Synchronous, wait until painting is finished
   canvas->Update(false,
   [](bool res) { std::cout << "First Update done = " << (res ? "true" : "false") << std::endl; });
   // Invalidate canvas and force repainting with next Update()
   canvas->Modified();
   // Call Update again, should return immediately if canvas was not modified
   canvas->Update(false,
   [](bool res) { std::cout << "Second Update done = " << (res ? "true" : "false") << std::endl; });
}

void tutorial() {
   ROOT::EnableImplicitMT();
   //if NOT zero (the file does NOT already exist), then Ingest
   if (gSystem->AccessPathName(kNTupleFileName) != 0) {
      Ingest();
   }
   Analyze();
}
