#include "TFile.h"
#include "TTree.h"
#include "Marker.h+"

void rewrite(const char *filename = "inherit.root")
{
   TFile *file = TFile::Open(filename,"READ");
   TFile *out = TFile::Open("inherit-2.root","RECREATE");
   
   fprintf(stdout,"Testing Directory\n");
   void *h = file->Get("holder");
   // gDebug = 7;
   fprintf(stdout,"Testing Writing\n");
   out->WriteObject(h,"holdercopy");
   
   fprintf(stdout,"Delete original\n");
   TClass *cl = TClass::GetClass("Holder");
   cl->Destructor(h);
   
   fprintf(stdout,"Testing the re-wrote object\n");
   h = out->Get("holdercopy");
   fprintf(stdout,"Delete copy\n");
   cl->Destructor(h);
   
//   fprintf(stdout,"Testing TTree\n");
//   TTree *tree; file->GetObject("tree",tree);
//   tree->GetEntry(0);
//   delete tree;
}