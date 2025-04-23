#include "colClass2.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranchElement.h"
#include "TMath.h"

bool testBranch(TTree *tree, const char *name)
{
   TBranchElement *br = dynamic_cast<TBranchElement*>(tree->GetBranch(name));
   if (br == 0) {
      fprintf(stdout,"Error: %s is missing from the TTree\n",name);
      return false;
   }
   if (br->GetObject() != 0) {
      fprintf(stdout,"Error: %s is pointing to a (missing) memory address\n",name);
      return false;
   }
   return true;
}

int execMissingCollection() {
   colClass *obj = 0;
   TFile *f = TFile::Open("missingCollection.root");
   if (!f || f->IsZombie()) {
      fprintf(stdout,"Error: Missing file missingCollection.root\n");
      return 1;
   }
   TTree *tree; f->GetObject("T",tree);
   if (!tree) {
      fprintf(stdout,"Error: Missing TTree object\n");
      return 1;
   }
   tree->SetBranchAddress("obj.",&obj);
   tree->GetEntry(0);
   if (!obj) {
      fprintf(stdout,"Error: Object not read in\n");
      return 2;
   }
   if (obj->fMeans.size() == 0) {
      fprintf(stdout,"Error: fMeans not read in\n");
      return 3;
   }
   if (obj->fMeans.size() != obj->fValues.size()) {
      fprintf(stdout,"Error: fValues (%ld) and fMeans (%ld) are not the same size\n",(long)obj->fValues.size(),(long)obj->fMeans.size());
   }
   for(unsigned int i = 0; i < obj->fMeans.size(); ++i) {
      if ( TMath::Abs( (i/2.0) - obj->fMeans[i] ) > 0.001 ) {
         fprintf(stdout,"Error, fMeans[i]; (%g) is not as expected (%g)\n", obj->fMeans[i], i/2.0);
         return 4;
      }
   }
   // if (!testBranch(tree,"obj.fObjects.fValue")) {
   //    return 5;
   // }
   if (!testBranch(tree,"obj.fObjects.fSub.fNestedValue")) {
      return 6;
   }
   if (!testBranch(tree,"obj.fObjects.fSub.fOther.fMoreNestedValue")) {
      return 7;
   }
   return 0;
}
      
