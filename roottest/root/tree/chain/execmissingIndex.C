#include "TFile.h"
#include "TChain.h"
#include "Riostream.h"

void writechain(const char *filename = "missingindex.root", bool debug = false) 
{
   TFile *output = TFile::Open(filename, "RECREATE");
   TTree *tree = new TTree("tree","main tree");
   Int_t evtnum = 0;
   tree->Branch("evtnum",&evtnum);
   for(evtnum = 0; evtnum < 21; ++evtnum) {
      tree->Fill();
   }
   tree->Write();
   delete tree; tree = 0;
   
   TTree *treefriend = new TTree("treefriend","friend of the main tree");
   treefriend->Branch("evtnum",&evtnum);
   
   for(evtnum = 0; evtnum < 21; ++evtnum) {
      if (debug) cout << evtnum << endl;
      if ( evtnum && ( (evtnum%3) == 0) ) {
         int keep = evtnum;
         if (evtnum%9 != 0) treefriend->BuildIndex("evtnum");
         else if (debug) { cout << "Skipping the build index at " << evtnum << '\n'; }
         evtnum = keep;
         treefriend->Write();
         treefriend->Reset();
         treefriend->SetName(TString::Format("treefriend%d",evtnum/3));
      }
      treefriend->Fill();
   }
   treefriend->BuildIndex("evtnum");
   output->Write();
}


void testAndScan(TTree *chain, TChain *chainfriend, const char *friendfilename, const char *option, const char *text) 
{
   chainfriend->Merge(friendfilename,option);
   
   TFile *f = TFile::Open(friendfilename);
   TTree *mergedfriend; f->GetObject("treefriend",mergedfriend);
   
   chain->AddFriend(mergedfriend);
   cout << "Scanning using " << text << "\n";
   chain->Scan("evtnum:treefriend.evtnum");   
   chain->RemoveFriend(mergedfriend);
   
   delete f;
}

void readchain(int order = 1, bool debug = false, const char *filename = "missingindex.root") {
   if (gSystem->AccessPathName(filename)) {
      writechain(filename);
   }
   
   TChain *chain = new TChain("tree");
   chain->Add(filename);
   TChain *chainfriend = new TChain("treefriend");
   if (order == 0) {
      chainfriend->Add(filename);
      for (int i = 1; i < (21/3); ++i) {
         chainfriend->Add(TString::Format("%s/treefriend%d",filename,i));
      }
      chainfriend->GetListOfFiles()->ls();
   } else {
      for (int i = (20/3); i > 0 ; --i) {
         chainfriend->Add(TString::Format("%s/treefriend%d",filename,i));
      }
      chainfriend->Add(filename);
      // chainfriend->GetListOfFiles()->ls();
   }
   chain->AddFriend(chainfriend);
   cout << "Scanning using the original friend\n";
   chain->Scan("evtnum:treefriend.evtnum");
   
   cout << "Scanning using the original friend after rebuilding the index\n";
   chainfriend->BuildIndex("evtnum");
   chain->Scan("evtnum:treefriend.evtnum");
   chain->RemoveFriend(chainfriend);
   
   TTree *treefriend = chainfriend->CloneTree(-1);
   treefriend->SetName("treefriend");
   
   chain->AddFriend(treefriend);
   cout << "Scanning using the cloned friend\n";
   chain->Scan("evtnum:treefriend.evtnum");
   chain->RemoveFriend(treefriend);

   testAndScan(chain, chainfriend, "missingindexfriend.root", "fast ", "the merged friend");
   testAndScan(chain, chainfriend, "droppedindexfriend.root", "fast dropindex", "tthe merged friend dropping the indices");
   testAndScan(chain, chainfriend, "noindexfriend.root",      "fast noindex", "the merged friend with no indices");
   testAndScan(chain, chainfriend, "buildindexfriend.root",   "fast buildindex", "the merged friend with index building");
   testAndScan(chain, chainfriend, "keepindexfriend.root",    "fast AsIsIndex", "the merged friend keeping the index as is (old default)");

   testAndScan(chain, chainfriend, "missingindexfriend.root", "", "the merged friend");
   testAndScan(chain, chainfriend, "droppedindexfriend.root", "dropindex", "tthe merged friend dropping the indices");
   testAndScan(chain, chainfriend, "noindexfriend.root",      "noindex", "the merged friend with no indices");
   testAndScan(chain, chainfriend, "buildindexfriend.root",   "buildindex", "the merged friend with index building");
   testAndScan(chain, chainfriend, "keepindexfriend.root",    "AsIsIndex", "the merged friend keeping the index as is (old default)");
}

int execmissingIndex(int order = 1, bool debug = false, const char *filename = "missingindex.root") {
   readchain(order,debug,filename);
   return 0;
}

