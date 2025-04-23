#include "no_autoparse.h"

#include "TBranch.h"
#include "TError.h"
#include "TInterpreter.h"

void writer(const char *filename /* = "autoparse_test.root" */ )
{
   // Currently spliting a map requires the ClassInfo for the pair!
   // (yes, does not sound right at all ... later fix)
   gInterpreter->ProcessLine("std::pair<reco::Muon::MuonTrackType, double> p;");

   TFile *f = TFile::Open(filename, "RECREATE");
   TTree *t = new TTree("Events","");
   std::pair<edm::Value, int> p;
   std::map<reco::Muon::MuonTrackType, double> m;
   m[ reco::Muon::kDefault ] = 3.0;
   edmNew::dstvdetails::DetSetVectorTrans::Item i;
#ifdef ITEM_V10
   i.fOldValue = 7.0;
#endif
   std::vector<edmNew::dstvdetails::DetSetVectorTrans::Item> vi;
   vi.push_back(i);
   std::vector<reco::Muon> muons;
   muons.resize(1);
   // The map is empty otherwise it would add the TStreamerInfo for the map
   reco::Muon muon;
   // muon.refittedTrackMap_[ reco::Muon::kDefault ] = 4.0;

   t->Branch("p.", &p);
   // Avoid this (split?) induces the addition of std::pair<reco::Muon::MuonTrackType, double>
   // which is currently not the case in the CMS file, hence delaying preventing it from doing
   // auto-parsing
   auto b = t->Branch("m_split.", &m, 32000, 99);
   if (!b || b->GetListOfBranches()->GetEntries() != 2)
      Fatal("no_autoparse writer", "The branch for map<reco::Muon::MuonTrackType, double> is not split");
   // unsplit seems to do it too.
   t->Branch("m_unsplit", &m, 32000, 0);
   t->Branch("muons_unsplit", &muons, 32000, 0);
   t->Branch("muons_split.", &muons, 32000, 99);
   t->Branch("muon_unsplit", &muon, 32000, 0);
   t->Branch("muon_split.", &muon, 32000, 99);
   b = t->GetBranch("muon_split.refittedTrackMap_");
   if (!b || b->GetListOfBranches()->GetEntries() != 2)
      Fatal("no_autoparse writer", "The branch for map<reco::Muon::MuonTrackType, double> inside muon_split is not split");
   t->Branch("i.", &i);
   t->Branch("vi.", &vi);


   t->Fill();
   f->Write();
   delete f;
};


