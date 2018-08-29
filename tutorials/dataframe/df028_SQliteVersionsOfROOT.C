/// \file
/// \ingroup tutorial_dataframe
/// \notebook -js
///
/// This tutorial demonstrates how RDataFrame can be used to create a
/// connection with a SQlite3 database. It accesses the Sqlite data base, and makes
/// a query selecting the entire table.
/// Then a TH1F histogram is created and filled
/// using a lambda expression which receives the recorded
/// values in the “version” column of the sqlite3 database.
/// The histogram shows the usage of the ROOT development version.
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

void df028_SQliteVersionsOfROOT() {

   auto rdf = ROOT::RDF::MakeSqliteDataFrame("https://root.cern.ch/download/root_download_stats.sqlite", "SELECT Version FROM accesslog;");

   TH1F hVersionOfRoot("hVersionOfRoot", "Development Versions of ROOT", 7, 0, -1);

   auto fillVersionHisto = [&hVersionOfRoot] (const std::string &version) {
      TString copyVersion = version;
      TString shortVersion(copyVersion(0,4));
      hVersionOfRoot.Fill(shortVersion, 1);
   };

   rdf.Foreach( fillVersionHisto, { "Version" } );

   TCanvas *VersionOfRootHistogram = new TCanvas();

   hVersionOfRoot.GetXaxis()->LabelsOption("a");
   hVersionOfRoot.LabelsDeflate("X");
   hVersionOfRoot.DrawClone();
}
