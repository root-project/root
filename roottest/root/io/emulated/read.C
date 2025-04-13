#include "TFile.h"
#include "TTree.h"
#ifdef ClingWorkAroundMissingSmartInclude
#include "marker.h"
#else
#include "marker.h+"
#endif

void read(const char *filename = "inherit.root")
{
   TFile *file = TFile::Open(filename,"READ");
   fprintf(stdout,"Testing Directory\n");
   void *h = file->Get("holder");
   TClass *cl = TClass::GetClass("Holder");
   cl->Destructor(h);
   
   fprintf(stdout,"Testing TTree\n");
   TTree *tree; file->GetObject("tree",tree);
   tree->GetEntry(0);
   delete tree;
}

