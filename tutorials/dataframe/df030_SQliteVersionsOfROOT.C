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
/// This product includes GeoLite2 data created by MaxMind, available from
/// <a href="http://www.maxmind.com">http://www.maxmind.com</a>.
///
/// \macro_code
/// \macro_image
///
/// \author Alexandra-Maria Dobrescu 08/2018

void df030_SQliteVersionsOfROOT() {

   // Davix has an issue at the moment: we download the file
   gEnv->SetValue("Davix.GSI.CACheck", "n");
   auto sqliteURL =  "https://root.cern.ch/download/root_download_stats.sqlite";
   TFile::Cp(sqliteURL, "root_download_stats_df030.sqlite");


   auto rdf = ROOT::RDF::MakeSqliteDataFrame("root_download_stats_df030.sqlite", "SELECT Version FROM accesslog;");

   TH1F hVersionOfRoot("hVersionOfRoot", "Development Versions of ROOT", 8, 0, -1);

   auto fillVersionHisto = [&hVersionOfRoot] (const std::string &version) {
      TString copyVersion = version;
      TString shortVersion(copyVersion(0,4));
      hVersionOfRoot.Fill(shortVersion, 1);
   };

   rdf.Foreach( fillVersionHisto, { "Version" } );

   auto VersionOfRootHistogram = new TCanvas();

   gStyle->SetOptStat(0);
   hVersionOfRoot.GetXaxis()->LabelsOption("a");
   hVersionOfRoot.LabelsDeflate("X");
   hVersionOfRoot.DrawClone("");
}
