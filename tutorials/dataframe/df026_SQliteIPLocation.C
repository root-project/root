/// \file
/// \ingroup tutorial_dataframe
/// \notebook -js
///
/// This tutorial demonstrates how RDataFrame can be used to create a
/// connection with a SQlite3 database. It accesses the Sqlite data base, and makes
/// a query selecting the entire table.
/// The world map is hold by a TH2Poly histogram which, after filling, will show
/// the world wide dispersion of ROOT's users.
/// To histogram filling, is done thanks to a lambda expression having as input parameters
/// the two columns of the database: "IPLongitude' - for the longitude, and the
/// "IPLatitude" - for the latitude.
/// The data related to the latitude and the longitude has been provided from the
/// log files storing the users IP.
/// This product includes GeoLite2 data created by MaxMind, available from
/// <a href="http://www.maxmind.com">http://www.maxmind.com</a>.
///
/// \macro_code
/// \macro_image
///
/// \author Alexandra-Maria Dobrescu 08/2018

void df026_SQliteIPLocation() {

   auto rdf = ROOT::RDF::MakeSqliteDataFrame( "https://root.cern.ch/download/root_download_stats.sqlite", "SELECT * FROM accesslog;" );

   auto F = TFile::Open("http://root.cern.ch/files/WM.root");
   TH2Poly *WM;
   WM = (TH2Poly*) F->Get("WMUSA");

   auto fillIPLocation = [&WM] ( const std::string &sLongitude, const std::string &sLatitude ) {
      if (!( sLongitude == "" ) && !( sLatitude == "" )) {
         float latitude = std::stof(sLatitude);
         float longitude = std::stof(sLongitude);
         WM->Fill(longitude, latitude);
      }
   };

   rdf.Foreach( fillIPLocation, { "IPLongitude", "IPLatitude" } );

   TCanvas *locationHistogram = new TCanvas();

   locationHistogram->SetLogz(1);
   locationHistogram->ToggleEventStatus();
   WM->DrawClone("colz");
}
