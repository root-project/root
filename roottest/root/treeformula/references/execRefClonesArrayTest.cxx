#include "execRefClonesArrayTest.h"

#include "TFile.h"
#include "TTree.h"

const char *filename = "reftca.root";

int write() 
{
  fprintf(stdout,"Writing %s\n",filename);
  Top* atop = new Top();
  TFile afile(filename, "recreate");
  TTree *tree = new TTree("tree", "tree");
 
  tree->BranchRef();
  tree->Branch("top", atop);

  for (size_t i=0;i<10;i++) {
    ObjA* a = static_cast<ObjA*>(atop->fObjAArray->New(i));
    ObjB* b = static_cast<ObjB*>(atop->fObjBArray->New(i));
    a->fObjAVal = i*100; 
    b->fObjBVal = i; 
    a->fObjB = b;
    atop->fLastB = b;
  }
  tree->Fill();
  tree->Write();
  afile.Close();
  tree = 0;
  fprintf(stdout,"Done writing %s\n",filename);
     
  return 0;
}

int read()
{
   fprintf(stdout,"Reading %s\n",filename);
   TFile *afile = TFile::Open(filename);
   if (afile == 0) {
      printf("Error: Missing the file %s.\n",filename);
      return 1;
   }
   TTree *tree = (TTree*) afile->Get("tree");
   if (tree == 0) {
      printf("Error: The file %s is missing the tree\n",filename);
      return 2;
   }

   const char *works_one = "fLastB.fObjBVal";
   const char *works_two = "fObjAArray.fObjAVal";
   const char *does_not_work = "fObjAArray.fObjB.fObjBVal";

   fprintf(stdout,"Scannning the tree\n");
   tree->Scan(works_one);
   tree->Scan(works_two);
   tree->Scan(does_not_work);
   return 0;
}

int execRefClonesArrayTest() 
{
   int res = write();
   if (res == 0) {
      res = read();
   } 
   return res;
}
   
