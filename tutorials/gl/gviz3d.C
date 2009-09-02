// Simple TStructViewer tutorial 

#include <TRandom.h>
#include <TList.h>
#include <TROOT.h>

// Function creating elements of lists
void MakeCrazy(TList *list, Int_t maxDepth, Int_t maxObjects, Float_t pList) {
   const Int_t ncl =12;
   const char *clnames[ncl] = {"TH1F","TGraph","TGraphErrors","TF1","TPaveText",
      "TAxis","TF2","TH2D","TLatex","TText","TCutG","THnSparseF"};
      Int_t nobj = gRandom->Uniform(0,maxObjects);
      for (Int_t i=0;i<nobj;i++) {
         if (maxDepth && gRandom->Rndm() < pList) {
            TList *slist = new TList();
            slist->SetName(Form("list_%d_%d",maxDepth,i));
            list->Add(slist);
            MakeCrazy(slist,maxDepth-1,maxObjects,pList);
         } else {
            Int_t icl = (Int_t)gRandom->Uniform(0,ncl);
            TNamed *named = (TNamed*)gROOT->ProcessLine(Form("new %s;",clnames[icl]));
            named->SetName(Form("%s_%d_%d",clnames[icl],maxDepth,i));
            list->Add(named);
         }
      }
}

// function creating a hierachy of objects to test the TStructViewer
TList *crazy(Int_t maxDepth=5, Int_t maxObjects=20, Float_t pList=0.2) {
   TList *list = new TList();
   list->SetName("SuperList");
   MakeCrazy(list,maxDepth,maxObjects,pList);
   gROOT->GetListOfTasks()->Add(list);
   return list;
}
// function adding colors to viewer
void FillColorsMap(TStructViewer* sv)
{
   // Fills list fColors with TStructNodeProperty
  const Int_t ncl =12;
  const char *clnames[ncl] = {"TH1F","TGraph","TGraphErrors","TF1","TPaveText",
    "TAxis","TF2","TH2D","TLatex","TText","TCutG","THnSparseF"};

  for (int i = 0; i < ncl ; i++) {
    sv->SetColor(clnames[i], (Int_t)gRandom->Integer(8)+2);
  }
}


void gviz3d()
{
   // Loading library needed by TStructViewer
   gSystem->Load("libGviz3d");
   // Creating a pointer to list
   TList* pointer = crazy(2,10);
   // Creating a viewer
   TStructViewer* sv = new TStructViewer(pointer);
   // adding colors
   FillColorsMap(sv);
   sv->Draw();
}
