#include "TFile.h"
#include "TChain.h"
#include "TROOT.h"

/*Find object in collection by using iterator instead of FindObject member function*/
Bool_t FindObject_loop(TCollection *col, TObject *obj) {
   // col->ls();
	TIter iter(col);
	for (TObject *item; (item = iter.Next()); ) {
		if (item == obj) {
			return kTRUE;
		}
	}
	return kFALSE;
}


void execTChainCleanup(const char *tmp_filename = "tmp2.root") {

   if(1) {
      TChain *chain = new TChain("my_chain");
      printf("created TChain.\n");
      printf("   %s in ListOfCleanups using member function FindObject()\n", gROOT->GetListOfCleanups()->FindObject(chain) ? "found" : "NOT found");
      printf("   %s in ListOfCleanups while iterating over collection\n", FindObject_loop(gROOT->GetListOfCleanups(), chain) ? "found" : "NOT found");
      
      TFile *file = new TFile(tmp_filename, "RECREATE");
      chain->Write();
      delete file;
      printf("TChain written to file.\n");
      
      delete chain;
      printf("deleted TChain.\n");
      printf("   %s in ListOfCleanups while iterating over collection\n", FindObject_loop(gROOT->GetListOfCleanups(), chain) ? "found" : "NOT found");
   }
   if (1) {
      TFile *file = new TFile(tmp_filename);
      TChain *chain = dynamic_cast<TChain*>(file->Get("my_chain"));
      if (chain != NULL) {
         printf("read TChain from file.\n");
         printf("   %s in ListOfCleanups using member function FindObject()\n", gROOT->GetListOfCleanups()->FindObject(chain) ? "found" : "NOT found");
         printf("   %s in ListOfCleanups while iterating over collection\n", FindObject_loop(gROOT->GetListOfCleanups(), chain) ? "found" : "NOT found");

         delete chain;
         printf("deleted TChain.\n");
         printf("   %s in ListOfCleanups while iterating over collection\n", FindObject_loop(gROOT->GetListOfCleanups(), chain) ? "found" : "NOT found");
      }

      delete file;
   }
}


