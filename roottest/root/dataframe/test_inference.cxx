#include "ROOT/RDataFrame.hxx"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3D.h"
#include "TH1D.h"
#include "TH1I.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TTree.h"

int main() {

   TTree t("t", "t");
   int v1 = 1, v2 = 2, v3 = 3;
   double w = 42.;
   t.Branch("v1", &v1);
   t.Branch("v2", &v2);
   t.Branch("v3", &v3);
   t.Branch("w", &w);
   t.Fill();

   std::cout << "histo1D, all calls must yield same results" << std::endl;
   ROOT::RDataFrame d1(t, {"v1", "w"});
   // histogram w/o weights -- these calls are all equivalent
   std::vector<ROOT::RDF::RResultPtr<TH1D>> results1D;
   results1D.emplace_back(d1.Histo1D());
   results1D.emplace_back(d1.Histo1D({"", "", 128u, 0., 4.}));
   results1D.emplace_back(d1.Histo1D<int>());
   results1D.emplace_back(d1.Histo1D<int>({"", "", 128u, 0., 4.}));
   results1D.emplace_back(d1.Histo1D("v1"));
   results1D.emplace_back(d1.Histo1D({"", "", 128u, 0., 4.}, "v1"));
   results1D.emplace_back(d1.Histo1D<int>("v1"));
   results1D.emplace_back(d1.Histo1D<int>({"", "", 128u, 0., 4.}, "v1"));

   // histogram w/ weights -- these calls are all equivalent
   results1D.emplace_back(d1.Histo1D("v1", "w"));
   results1D.emplace_back(d1.Histo1D({"", "", 128u, 0., 4.}, "v1", "w"));
   results1D.emplace_back(d1.Histo1D<int>("v1", "w"));
   results1D.emplace_back(d1.Histo1D<int>({"", "", 128u, 0., 4.}, "v1", "w"));
   results1D.emplace_back(d1.Histo1D<int, double>("v1", "w"));
   results1D.emplace_back(d1.Histo1D<int, double>({"", "", 128u, 0., 4.}, "v1", "w"));
   results1D.emplace_back(d1.Histo1D<int, double>());
   results1D.emplace_back(d1.Histo1D<int, double>({"", "", 128u, 0., 4.}, "v1", "w"));

   for(auto& res : results1D)
      std::cout << res->GetEntries() << " " << res->GetMean() << std::endl;

   // TODO throughout the test the model histogram/profiles are moved multiple times. This is safe, because histograms
   // and profiles do not actually invalidate anything when move-constructed. If this changes in the future, each one
   // of the calls to Histo* and Profile* actions should provide a different model.
   std::cout << "histo2D, all calls must yield same results" << std::endl;
   ROOT::RDataFrame d2(t, {"v1", "v2", "w"});
   // histo2D w/o weights
   std::vector<ROOT::RDF::RResultPtr<TH2D>> results2D;
   TH2D model2D{"h2", "h2", 128u, 0., 4., 128u, 0., 4.};
   results2D.emplace_back(d2.Histo2D(std::move(model2D)));
   results2D.emplace_back(d2.Histo2D(std::move(model2D), "v1"));
   results2D.emplace_back(d2.Histo2D(std::move(model2D), "v1", "v2"));
   results2D.emplace_back(d2.Histo2D<int>(std::move(model2D)));
   results2D.emplace_back(d2.Histo2D<int>(std::move(model2D), "v1"));
   results2D.emplace_back(d2.Histo2D<int>(std::move(model2D), "v1", "v2"));
   results2D.emplace_back(d2.Histo2D<int,int>(std::move(model2D)));
   results2D.emplace_back(d2.Histo2D<int,int>(std::move(model2D), "v1"));
   results2D.emplace_back(d2.Histo2D<int,int>(std::move(model2D), "v1", "v2"));

   // histo2D w/ weights
   results2D.emplace_back(d2.Histo2D(std::move(model2D), "v1", "v2", "w"));
   results2D.emplace_back(d2.Histo2D<int>(std::move(model2D), "v1", "v2", "w"));
   results2D.emplace_back(d2.Histo2D<int,int>(std::move(model2D), "v1", "v2", "w"));
   results2D.emplace_back(d2.Histo2D<int,int,double>(std::move(model2D), "v1", "v2", "w"));
   results2D.emplace_back(d2.Histo2D<int,int,double>(std::move(model2D)));

   for(auto& res : results2D)
      std::cout << res->GetEntries() << " " << res->GetMean(1) << " " << res->GetMean(2) << std::endl;


   std::cout << "histo3D, all calls must yield same results" << std::endl;
   ROOT::RDataFrame d3(t, {"v1", "v2", "v3", "w"});
   // Histo3D w/o weights
   std::vector<ROOT::RDF::RResultPtr<TH3D>> results3D;
   TH3D model3D{"h3", "h3", 128u, 0., 4., 128u, 0., 4., 128u, 0., 4.};
   results3D.emplace_back(d3.Histo3D(std::move(model3D)));
   results3D.emplace_back(d3.Histo3D(std::move(model3D), "v1"));
   results3D.emplace_back(d3.Histo3D(std::move(model3D), "v1", "v2"));
   results3D.emplace_back(d3.Histo3D(std::move(model3D), "v1", "v2", "v3"));
   results3D.emplace_back(d3.Histo3D<int>(std::move(model3D)));
   results3D.emplace_back(d3.Histo3D<int>(std::move(model3D), "v1"));
   results3D.emplace_back(d3.Histo3D<int>(std::move(model3D), "v1", "v2"));
   results3D.emplace_back(d3.Histo3D<int>(std::move(model3D), "v1", "v2", "v3"));
   results3D.emplace_back(d3.Histo3D<int,int>(std::move(model3D)));
   results3D.emplace_back(d3.Histo3D<int,int>(std::move(model3D), "v1"));
   results3D.emplace_back(d3.Histo3D<int,int>(std::move(model3D), "v1", "v2"));
   results3D.emplace_back(d3.Histo3D<int,int>(std::move(model3D), "v1", "v2", "v3"));
   results3D.emplace_back(d3.Histo3D<int,int,int>(std::move(model3D), "v1"));
   results3D.emplace_back(d3.Histo3D<int,int,int>(std::move(model3D), "v1", "v2"));
   results3D.emplace_back(d3.Histo3D<int,int,int>(std::move(model3D), "v1", "v2", "v3"));

   // Histo3D w/ weights
   results3D.emplace_back(d3.Histo3D(std::move(model3D), "v1", "v2", "v3", "w"));
   results3D.emplace_back(d3.Histo3D<int>(std::move(model3D), "v1", "v2", "v3", "w"));
   results3D.emplace_back(d3.Histo3D<int,int>(std::move(model3D), "v1", "v2", "v3", "w"));
   results3D.emplace_back(d3.Histo3D<int,int,int>(std::move(model3D), "v1", "v2", "v3", "w"));
   results3D.emplace_back(d3.Histo3D<int,int,int,double>(std::move(model3D), "v1", "v2", "v3", "w"));
   results3D.emplace_back(d3.Histo3D<int,int,int,double>(std::move(model3D)));

   for(auto& res : results3D)
      std::cout << res->GetEntries() << " " << res->GetMean(1) << " " << res->GetMean(2) << " " << res->GetMean(3)
                << std::endl;

   std::cout << "Profile1D, all calls must yield same results" << std::endl;
   ROOT::RDataFrame d4(t, {"v1", "v2", "w"});
   // Profile1D w/o weights
   std::vector<ROOT::RDF::RResultPtr<TProfile>> resultsProf1D;
   TProfile modelProf1D{"", "", 128u, 0., 4., 0., 4.};
   resultsProf1D.emplace_back(d4.Profile1D(std::move(modelProf1D)));
   resultsProf1D.emplace_back(d4.Profile1D(std::move(modelProf1D), "v1"));
   resultsProf1D.emplace_back(d4.Profile1D(std::move(modelProf1D), "v1", "v2"));
   resultsProf1D.emplace_back(d4.Profile1D<int>(std::move(modelProf1D)));
   resultsProf1D.emplace_back(d4.Profile1D<int>(std::move(modelProf1D), "v1"));
   resultsProf1D.emplace_back(d4.Profile1D<int>(std::move(modelProf1D), "v1", "v2"));
   resultsProf1D.emplace_back(d4.Profile1D<int,int>(std::move(modelProf1D)));
   resultsProf1D.emplace_back(d4.Profile1D<int,int>(std::move(modelProf1D), "v1"));
   resultsProf1D.emplace_back(d4.Profile1D<int,int>(std::move(modelProf1D), "v1", "v2"));

   // Profile1D w/ weights
   resultsProf1D.emplace_back(d4.Profile1D(std::move(modelProf1D), "v1", "v2", "w"));
   resultsProf1D.emplace_back(d4.Profile1D<int>(std::move(modelProf1D), "v1", "v2", "w"));
   resultsProf1D.emplace_back(d4.Profile1D<int,int>(std::move(modelProf1D), "v1", "v2", "w"));
   resultsProf1D.emplace_back(d4.Profile1D<int,int,double>(std::move(modelProf1D), "v1", "v2", "w"));
   resultsProf1D.emplace_back(d4.Profile1D<int,int,double>(std::move(modelProf1D)));

   for(auto& res : resultsProf1D)
      std::cout << res->GetEntries() << " " << res->GetMean(1) << " " << res->GetMean(2) << std::endl;


   std::cout << "Profile2D, all calls must yield same results" << std::endl;
   ROOT::RDataFrame d5(t, {"v1", "v2", "v3", "w"});
   // Profile2D w/ weights
   std::vector<ROOT::RDF::RResultPtr<TProfile2D>> resultsProf2D;
   TProfile2D modelProf2D{"", "", 128u, 0., 4., 128u, 0., 4., 0., 4.};
   resultsProf2D.emplace_back(d5.Profile2D(std::move(modelProf2D)));
   resultsProf2D.emplace_back(d5.Profile2D(std::move(modelProf2D), "v1"));
   resultsProf2D.emplace_back(d5.Profile2D(std::move(modelProf2D), "v1", "v2"));
   resultsProf2D.emplace_back(d5.Profile2D(std::move(modelProf2D), "v1", "v2", "v3"));
   resultsProf2D.emplace_back(d5.Profile2D<int>(std::move(modelProf2D)));
   resultsProf2D.emplace_back(d5.Profile2D<int>(std::move(modelProf2D), "v1"));
   resultsProf2D.emplace_back(d5.Profile2D<int>(std::move(modelProf2D), "v1", "v2"));
   resultsProf2D.emplace_back(d5.Profile2D<int>(std::move(modelProf2D), "v1", "v2", "v3"));
   resultsProf2D.emplace_back(d5.Profile2D<int,int>(std::move(modelProf2D)));
   resultsProf2D.emplace_back(d5.Profile2D<int,int>(std::move(modelProf2D), "v1"));
   resultsProf2D.emplace_back(d5.Profile2D<int,int>(std::move(modelProf2D), "v1", "v2"));
   resultsProf2D.emplace_back(d5.Profile2D<int,int>(std::move(modelProf2D), "v1", "v2", "v3"));
   resultsProf2D.emplace_back(d5.Profile2D<int,int,int>(std::move(modelProf2D), "v1"));
   resultsProf2D.emplace_back(d5.Profile2D<int,int,int>(std::move(modelProf2D), "v1", "v2"));
   resultsProf2D.emplace_back(d5.Profile2D<int,int,int>(std::move(modelProf2D), "v1", "v2", "v3"));

   // Profile2D w/ weights
   resultsProf2D.emplace_back(d5.Profile2D(std::move(modelProf2D), "v1", "v2", "v3", "w"));
   resultsProf2D.emplace_back(d5.Profile2D<int>(std::move(modelProf2D), "v1", "v2", "v3", "w"));
   resultsProf2D.emplace_back(d5.Profile2D<int,int>(std::move(modelProf2D), "v1", "v2", "v3", "w"));
   resultsProf2D.emplace_back(d5.Profile2D<int,int,int>(std::move(modelProf2D), "v1", "v2", "v3", "w"));
   resultsProf2D.emplace_back(d5.Profile2D<int,int,int,double>(std::move(modelProf2D), "v1", "v2", "v3", "w"));
   resultsProf2D.emplace_back(d5.Profile2D<int,int,int,double>(std::move(modelProf2D)));

   for(auto& res : resultsProf2D)
      std::cout << res->GetEntries() << " " << res->GetMean(1) << " " << res->GetMean(2) << " " << res->GetMean(3)
                << std::endl;

   std::cout << "Fill" << std::endl;
   // Fill
   ROOT::RDataFrame d6(t);
   auto fill1 = d6.Fill<int>(TH1F("", "", 64, 0, 128), {"v1"});
   auto fill2 = d6.Fill(TH1F("", "", 64, 0, 128), {"v1"});
   auto fill3 = d6.Fill<int,double>(TH1I("", "", 64, 0, 128), {"v1","w"});
   auto fill4 = d6.Fill(TH1I("", "", 64, 0, 128), {"v1","w"});
   auto fill5 = d6.Fill<int, int>(TH2F("", "", 64, 0, 128, 64, 0, 1024), {"v1", "v2"});
   auto fill6 = d6.Fill(TH2F("", "", 64, 0, 128, 64, 0, 1024), {"v1", "v2"});
   std::cout << fill1->GetEntries() << " " << fill1->GetMean() << std::endl;
   std::cout << fill2->GetEntries() << " " << fill2->GetMean() << std::endl;
   std::cout << fill3->GetEntries() << " " << fill3->GetMean() << std::endl;
   std::cout << fill4->GetEntries() << " " << fill4->GetMean() << std::endl;
   std::cout << fill5->GetEntries() << " " << fill5->GetMean(1) << " " << fill5->GetMean(2) << std::endl;
   std::cout << fill6->GetEntries() << " " << fill6->GetMean(1) << " " << fill6->GetMean(2) << std::endl;

   return 0;
}
