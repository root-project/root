/// \file
/// \ingroup tutorial_dataframe
/// \notebook -js
///
/// This tutorial demonstrates how RDataFrame can be used to create a
/// connection with a SQlite3 database. It accesses the Sqlite data base, and makes
/// a query selecting the entire table.
/// In order to display the Platform Distribution of ROOT, we choose to create two TH1F
/// histograms: one that includes all types of platforms, other filtering and classifying them.
/// This procedure is using a lambda expression taking as parameter the values
/// stored in the "Platform" column from the database. At the end, the histograms are filled
/// with their specific demand regarding the platform's type.
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

void df027_SQlitePlatformDistribution() {

   auto rdf = ROOT::RDF::MakeSqliteDataFrame( "https://root.cern.ch/download/root_download_stats.sqlite", "SELECT * FROM accesslog;" );

   TH1F hRootPlatform("hrootPlatform", "Platform Distribution", 7, 0, -1);
   TH1F hShortRootPlatform("hShortRootPlatform", "Short Platform Distribution", 7, 0, -1);

   auto fillRootPlatform = [&hRootPlatform, &hShortRootPlatform] ( const std::string &platform ) {
      TString Platform = platform;
      TString Platform_0(Platform(0,5));
      TString Platform_1(Platform(0,6));
      TString Platform_2(Platform(0,8));

      if ( Platform.Contains("win32") ){
        hShortRootPlatform.Fill(Platform_0,1);
      } else if ( Platform.Contains("Linux") ){
        hShortRootPlatform.Fill(Platform_0,1);
      } else if ( Platform.Contains("source") ){
        hShortRootPlatform.Fill(Platform_1,1);
      } else if ( Platform.Contains("macosx64") ){
        hShortRootPlatform.Fill(Platform_2,1);
      } else if ( Platform.Contains("IRIX64") ){
        hShortRootPlatform.Fill(Platform_1,1);
      }

      hRootPlatform.Fill(Platform,1);
   };

   rdf.Foreach( fillRootPlatform, { "Platform" } );

   TCanvas *PlatformDistributionHistogram = new TCanvas();

   hRootPlatform.GetXaxis()->LabelsOption("a");
   hRootPlatform.LabelsDeflate("X");
   hRootPlatform.DrawClone();

   TCanvas *shortPlatformDistributionHistogram = new TCanvas();

   hShortRootPlatform.GetXaxis()->LabelsOption("a");
   hShortRootPlatform.LabelsDeflate("X");
   hShortRootPlatform.DrawClone();
}
