/// \file
/// \ingroup tutorial_sql
/// \notebook -js
///
/// This tutorial demonstrates how TSQLServer can be used to create a
/// connection with a SQlite3 database. It accesses the Sqlite data base.
/// Download from https://root.cern/download/root_download_stats.sqlite
/// Then a TH1F histogram is created and filled
/// using a expression which receives the recorded
/// values in the "version" column of the sqlite3 database.
/// The histogram shows the usage of the ROOT development version.
/// This product includes GeoLite2 data created by MaxMind, available from
/// <a href="http://www.maxmind.com">http://www.maxmind.com</a>.
///
/// \macro_code
///
/// \author Alexandra-Maria Dobrescu 08/2018

#include <TSQLiteServer.h>
#include <TSQLiteResult.h>
#include <TSQLRow.h>
#include <TString.h>

void SQLiteVersionsOfRoot(){

   TSQLServer *db = TSQLServer::Connect("sqlite://root_download_stats.sqlite", "", "");

   const char *rootSourceVersion = "SELECT Version FROM accesslog;";

   TSQLResult *rootSourceVersionRes = db->Query(rootSourceVersion);

   TH1F *hVersionOfRoot= new TH1F("hVersionOfRoot", "Development Versions of ROOT", 7, 0, -1);

   while (TSQLRow *row = rootSourceVersionRes->Next()) {
      TString rowVersion(row->GetField(0));
      TString shortVersion(rowVersion(0,4));
      hVersionOfRoot->Fill(shortVersion,1);
      delete row;
   }

   TCanvas *VersionOfRootHistogram = new TCanvas();

   hVersionOfRoot->GetXaxis()->LabelsOption("a");
   hVersionOfRoot->LabelsDeflate("X");
   hVersionOfRoot->Draw();
}
