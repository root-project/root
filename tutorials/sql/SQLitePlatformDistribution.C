/// \file
/// \ingroup tutorial_sql
/// \notebook -js
///
/// This tutorial demonstrates how TSQLServer can be used to create a
/// connection with a SQlite3 database. It accesses the Sqlite data base.
/// Download from https://root.cern/download/root_download_stats.sqlite
/// In order to display the Platform Distribution of ROOT, we choose to create two TH1F
/// histograms: one that includes all types of platforms, other filtering and classifying them.
/// This procedure is taking as parameter the values stored in the "Platform" column from the 
/// database. At the end, the histograms are filled
/// with their specific demand regarding the platform's type.
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

void SQLitePlatformDistribution(){

   TSQLServer *db = TSQLServer::Connect("sqlite://root_download_stats.sqlite", "", "");

   const char *rootPlatform = "SELECT Platform FROM accesslog;";

   TSQLResult *rootPlatformRes = db->Query(rootPlatform);

   TH1F *hrootPlatform = new TH1F("hrootPlatform", "Platform Distribution", 7, 0, -1);
   TH1F *shorthrootPlatform = new TH1F("shorthrootPlatform", "Short Platform Distribution", 7, 0, -1);

   while (TSQLRow *row = rootPlatformRes->Next()) {
      TString rowPlatform(row->GetField(0));
      TString Platform(rowPlatform);
      TString Platform_0(rowPlatform(0,5));
      TString Platform_1(rowPlatform(0,6));
      TString Platform_2(rowPlatform(0,8));
      if ( rowPlatform.Contains("win32") ){
         shorthrootPlatform->Fill(Platform_0,1);
      } else if ( rowPlatform.Contains("Linux") ){
         shorthrootPlatform->Fill(Platform_0,1);
      } else if ( rowPlatform.Contains("source") ){
         shorthrootPlatform->Fill(Platform_1,1);
      } else if ( rowPlatform.Contains("macosx64") ){
         shorthrootPlatform->Fill(Platform_2,1);
      } else if ( rowPlatform.Contains("IRIX64") ){
         shorthrootPlatform->Fill(Platform_1,1);
      }

      hrootPlatform->Fill(Platform,1);

      delete row;
   }

   TCanvas *PlatformDistributionHistogram = new TCanvas();

   hrootPlatform->GetXaxis()->LabelsOption("a");
   hrootPlatform->LabelsDeflate("X");
   hrootPlatform->Draw();

   TCanvas *shortPlatformDistributionHistogram = new TCanvas();

   shorthrootPlatform->GetXaxis()->LabelsOption("a");
   shorthrootPlatform->LabelsDeflate("X");
   shorthrootPlatform->Draw();
}
