/// \file
/// \ingroup tutorial_sql
/// \notebook -js
///
/// This tutorial demonstrates how TSQLServer can be used to create a
/// connection with a SQlite3 database. It accesses the Sqlite data base.
/// Download from https://root.cern/download/root_download_stats.sqlite
/// In order to demonstrate the dependency over ROOT version 6.14, this tutorial uses the TSQLResult
/// function which allows to extract the minimum time stored in the SQlite3 database.
/// The next step is to create a TH1F Histogram, which will be filled with the values stored in
/// two different columns from the database, the "Time" and "Version" columns.
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

void SQLiteTimeVersionOfRoot(){

   TSQLServer *db = TSQLServer::Connect("sqlite://root_download_stats.sqlite", "", "");

   const char *minTime = "SELECT min(Time) FROM accesslog;";
   TSQLResult *minTimeRes = db->Query(minTime);

   std::string strMinTimeField = minTimeRes->Next()->GetField(0);
   TDatime minTimeFormat(strMinTimeField.c_str());

   TDatime now;
   TH1F *hTime = new TH1F("hTime", "Duration of ROOT dependency over version 6.14", 10, minTimeFormat.Convert(), now.Convert());

   const char *time = "SELECT Time, Version FROM accesslog;";
   TSQLResult *timeRes = db->Query(time);

   while (TSQLRow *row = timeRes->Next()) {
      TDatime rowTime(row->GetField(0));
      TString rowVersion(row->GetField(1));
      TString shortVersion(rowVersion(0,4));
      if ( shortVersion == "6.14" ) {
         hTime->Fill(rowTime.Convert());
      }
      delete row;
   }

   TCanvas *timeHistogram = new TCanvas();

   gStyle->SetTimeOffset(0);
   hTime->GetXaxis()->SetTimeDisplay(1);
   hTime->GetXaxis()->SetLabelSize(0.02);
   hTime->GetXaxis()->SetNdivisions(512, kFALSE);
   hTime->GetXaxis()->SetTimeFormat("%Y-%m-%d");

   hTime->Draw();
}
