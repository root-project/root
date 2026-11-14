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
   fprintf(stdout, "Testing Directory\n");
   TClass *cl = TClass::GetClass("Holder");
   if (!cl) {
      std::cerr << "Error: could not find the Holder TClass\n";
      return;
   }
   void *h = file->GetObjectChecked("holder", cl);
   if (!h) {
      std::cerr << "Error: could not read the Holder object from the file\n";
      return;
   }
   cl->Destructor(h);
   
   fprintf(stdout,"Testing TTree\n");
   TTree *tree; file->GetObject("tree",tree);
   tree->GetEntry(0);
   delete tree;
}

