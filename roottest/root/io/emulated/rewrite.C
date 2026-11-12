#include "TFile.h"
#include "TTree.h"
#ifdef ClingWorkAroundMissingSmartInclude
#include "marker.h"
#else
#include "marker.h+"
#endif

void rewrite(const char *filename = "inherit.root")
{
   TFile *file = TFile::Open(filename,"READ");
   TFile *out = TFile::Open("inherit-2.root","RECREATE");
   
   fprintf(stdout,"Testing Directory\n");
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

   // gDebug = 7;
   fprintf(stdout,"Testing Writing\n");
   out->WriteObjectAny(h,"Holder","holdercopy");

   fprintf(stdout, "Delete original\n");
   cl->Destructor(h);
   
   fprintf(stdout,"Testing the re-wrote object\n");
   h = out->GetObjectChecked("holdercopy", cl);
   if (!h) {
      std::cerr << "Error: could not read the copy of the Holder object from the file\n";
      return;
   }
   fprintf(stdout,"Delete copy\n");
   cl->Destructor(h);
   
//   fprintf(stdout,"Testing TTree\n");
//   TTree *tree; file->GetObject("tree",tree);
//   tree->GetEntry(0);
//   delete tree;
}
