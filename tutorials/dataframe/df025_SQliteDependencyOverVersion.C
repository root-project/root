/// \file
/// \ingroup tutorial_dataframe
/// \notebook -js
///
/// This tutorial demonstrates how RDataFrame can be used to create a
/// connection with a SQlite3 database. It accesses the Sqlite data base, and makes
/// a query selecting the entire table.
/// In order to demonstrate the dependency over ROOT version 6.14, this tutorial uses the Reduce
/// function which allows to extract the minimum time stored in the SQlite3 database.
/// The next step is to create a TH1F Histogram, which will be filled with the values stored in
/// two different columns from the database. This procedure is simplified with a lambda
/// expression that takes as parameters the values stored in the "Time" and "Version" columns.
/// This product includes GeoLite2 data created by MaxMind, available from
/// <a href="http://www.maxmind.com">http://www.maxmind.com</a>.
///
/// \macro_code
/// \macro_image
///
/// \author Alexandra-Maria Dobrescu 08/2018

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDataSource.hxx"
#include "ROOT/RSqliteDS.hxx"
#include <TString.h>
#include <sqlite3.h>

#include <algorithm>

void df025_SQliteDependencyOverVersion () {

   auto rdf = ROOT::RDF::MakeSqliteDataFrame( "https://root.cern.ch/download/root_download_stats.sqlite", "SELECT * FROM accesslog;" );

   auto minTime = *rdf.Reduce([](std::string a, std::string b) {return std::min(a, b);}, "Time", std::string("Z"));
   std::cout << "Minimum time is '" << minTime << "'" << std::endl;
   TDatime minTimeFormat(minTime.c_str());

   TDatime now;
   TH1F hTime( "hTime", "Duration of ROOT dependency over version 6.14", 10, minTimeFormat.Convert(), now.Convert() );

   auto fillTimeHisto = [&hTime] ( const std::string &time, const std::string &version ) {
      TDatime timeRes(time.c_str());
      TString copyVersion = version;
      TString shortVersion(copyVersion(0,4));

      if ( shortVersion == "6.14" ) {
         hTime.Fill(timeRes.Convert());
      }
   };

   rdf.Foreach(fillTimeHisto, { "Time", "Version" });

   TCanvas *timeHistogram = new TCanvas();

   gStyle->SetTimeOffset(0);
   hTime.GetXaxis()->SetTimeDisplay(1);
   hTime.GetXaxis()->SetLabelSize(0.02);
   hTime.GetXaxis()->SetNdivisions(512, kFALSE);
   hTime.GetXaxis()->SetTimeFormat("%Y-%m-%d");

   hTime.DrawClone();

}
