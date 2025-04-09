#if defined(__CLING__) && !defined(__MAKECINT__) && !defined(ClingWorkAroundMissingSmartInclude)
#include "classes.h+"
#else
#include "classes.h"
#endif
#include "TFile.h"
#include "TTree.h"

void write(const char *filename = "inherit.root")
{
   fprintf(stdout,"Testing Directory\n");
   TClass::GetClass("Marker")->GetStreamerInfo(); // To synchronize the input of write.C and read.C
   TFile *file = TFile::Open(filename,"RECREATE");
   Holder *h = new Holder;
   h->Init();
   file->WriteObject(h,"holder");
   delete h;
   fprintf(stdout,"Testing TTree\n");
   h = new Holder;
   h->Init();
   TTree *tree = new TTree("tree","tree testing inheritance");
   tree->Branch("holder",&h);
   
   tree->Fill();
   file->Write();
   delete file;
   delete h;
}