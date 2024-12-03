/// \file
/// \ingroup tutorial_dataframe
/// \notebook -js
/// Plot the ROOT downloads based on the version reading a remote sqlite3 file.
///
/// This tutorial uses the Reduce method which allows to extract the minimum time
/// stored in the SQlite3 database.
/// The next step is to create a TH1F Histogram, which will be filled with the values stored in
/// two different columns from the database. This procedure is simplified with a lambda
/// expression that takes as parameters the values stored in the "Time" and "Version" columns.
///
/// \macro_code
/// \macro_image
///
/// \date August 2018
/// \authors Alexandra-Maria Dobrescu, Sergey Linev


void df027_SQliteDependencyOverVersion ()
{
   auto rdfb = ROOT::RDF::FromSqlite("http://root.cern/files/root_download_stats.sqlite", "SELECT * FROM accesslog;");

   auto minTimeStr = *rdfb.Reduce([](std::string a, std::string b) {return std::min(a, b);}, "Time", std::string("Z"));

   std::cout << "Minimum time is '" << minTimeStr << "'" << std::endl;

   double minTime = TDatime(minTimeStr.c_str()).Convert();
   double maxTime = minTime + 3600.*24*365.25*4; // cover approx 4 years from minimal time

   auto rdf = rdfb.Define("datime", [](const std::string &time){return TDatime(time.c_str()).Convert();}, {"Time"});

   auto h614 = rdf.Filter([](const std::string &v){ return 0 == v.find("6.14");}, {"Version"})
                  .Histo1D({"h614", "Download time for version 6.14", 64, minTime, maxTime}, {"datime"});

   auto h616 = rdf.Filter([](const std::string &v){ return 0 == v.find("6.16");}, {"Version"})
                  .Histo1D({"h616", "Download time for version 6.16", 64, minTime, maxTime}, {"datime"});

   auto h618 = rdf.Filter([](const std::string &v){ return 0 == v.find("6.18");}, {"Version"})
                  .Histo1D({"h618", "Download time for version 6.18", 64, minTime, maxTime}, {"datime"});

   auto customize_histo = [](TH1D &histo) {
      auto *xaxis = histo.GetXaxis();
      xaxis->SetTimeDisplay(1);
      xaxis->SetLabelSize(0.02);
      xaxis->SetNdivisions(512, kFALSE);
      xaxis->SetTimeFormat("%Y-%m-%d%F1970-00-00 00:00:00");
      histo.SetStats(kFALSE);
   };

   customize_histo(*h614);
   customize_histo(*h616);
   customize_histo(*h618);

   std::vector drawables{h614->Clone(), h616->Clone(), h618->Clone()};

   auto c1 = new TCanvas("c1","Download time", 800, 1500);
   c1->Divide(1, drawables.size());
   for (unsigned n = 0; n < drawables.size(); ++n)
      c1->GetPad(n + 1)->Add(drawables[n]);
}
