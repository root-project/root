/// \file
/// \ingroup tutorial_hist
/// \notebook -js
/// This tutorial illustrates how to create a histogram with polygonal
/// bins (TH2Poly), fill it, and draw it using the `col` option. The initial data
/// are stored in TMultiGraphs. They represent the USA map. Such histograms can
/// be rendered in 3D using the option `legogl`.
///
/// The initial data have been downloaded from: http://www.maproom.psu.edu/dcw/
/// This database was developed in 1991/1992 and national boundaries reflect
/// political reality as of that time.
///
/// \macro_code
/// \macro_image
///
/// \author Olivier Couet

void th2polyUSA() {
   std::vector<std::pair<std::string, UInt_t>> statePop
   {
         {"alabama", 4708708}, {"arizona", 6595778}, {"arkansas", 2889450}, {"california", 36961664},
         {"colorado", 5024748}, {"connecticut", 3518288}, {"delaware", 885122}, {"florida", 18537969},
         {"georgia", 9829211}, {"idaho", 1545801}, {"illinois", 12910409}, {"indiana", 6423113}, {"iowa", 3007856},
         {"kansas", 2818747}, {"kentucky", 4314113}, {"louisiana", 4492076}, {"maine", 1318301}, {"maryland", 5699478},
         {"massachusetts", 6593587}, {"michigan", 9969727}, {"minnesota", 5266214}, {"mississippi", 2951996},
         {"missouri", 5987580}, {"montana", 974989}, {"nebraska", 1796619}, {"nevada", 2643085},
         {"new_hampshire", 1324575}, {"new_jersey", 8707739}, {"new_mexico", 2009671}, {"new_york", 19541453},
         {"north_carolina", 9380884}, {"north_dakota", 646844}, {"ohio", 11542645}, {"oklahoma", 3687050},
         {"oregon", 3825657}, {"pennsylvania", 12604767}, {"rhode_island", 1053209}, {"south_carolina", 4561242},
         {"south_dakota", 812383}, {"tennessee", 6296254}, {"texas", 24782302}, {"utah", 2784572}, {"vermont", 621760},
         {"virginia", 7882590}, {"washington", 6664195}, {"west_virginia", 1819777}, {"wisconsin", 5654774},
         {"wyoming", 544270}
   };

   auto lon1 = -130.;
   auto lon2 = -65.;
   auto lat1 = 24.;
   auto lat2 = 50.;
   auto p = new TH2Poly("USA","USA Population",lon1,lon2,lat1,lat2);
   TFile::SetCacheFileDir(".");
   auto f = TFile::Open("http://root.cern/files/usa.root", "CACHEREAD");

   if (!f) {
      printf("Cannot access usa.root. Is internet working ?\n");
      return;
   }

   // Define the TH2Poly bins.
   for (auto [state, pop] : statePop) {
      if (auto stateGraph = f->Get<TMultiGraph>(state.c_str())) {
         p->AddBin(stateGraph);
         p->Fill(state.c_str(), pop);
      } else {
         std::cerr << "Error reading object " << state << " from the file." << std::endl;
      }
   }

   // Draw
   gStyle->SetOptStat(11);
   p->Draw("colz textn");
}
