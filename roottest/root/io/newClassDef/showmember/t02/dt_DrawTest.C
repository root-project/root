#include "TTree.h"
#include "TFile.h"
#include "TList.h"
#include <TBenchmark.h>

int gHasLibrary = kFALSE;
TList gSkipped;

void DrawSkippable(TTree* tree, const char* what, const char* where, Bool_t skip) {
  //cerr << "Doing " << what << " which is " << skip << endl;
  if (skip) gSkipped.Add(new TNamed(where,where));
  else {
    TString cut = what;
    cut.Append(">>");
    cut.Append(where);
    tree->Draw(cut.Data(),"","goff");
  }
};

void DrawSkippable(TTree* tree, const char* what, const char* cond,
                   const char* where, Bool_t skip) {
  //cerr << "Doing " << what << " which is " << skip << endl;
  if (skip) gSkipped.Add(new TNamed(where,where));
  else {
    TString cut = what;
    cut.Append(">>");
    cut.Append(where);
    tree->Draw(cut.Data(),cond,"goff");
  }
};

// Rootmarks for fcdflnx1 is 153.4
void DrawMarks() {

  // The base is currently: RunDrawTest.C++("Event.old.split.root",0)
  Float_t rt_base = 2.33;
  Float_t cp_base = 2.34;

  Float_t rt = gBenchmark->GetRealTime("DrawTest");
  Float_t ct = gBenchmark->GetCpuTime("DrawTest");

  // gBenchmark->Print("DrawTest");
  
  Float_t rootmarks = 200*(rt_base + cp_base)/(rt + ct);
  
  printf("*  ROOTMARKS =%6.1f   *  Root%-8s  %d/%d\n",rootmarks,gROOT->GetVersion(),gROOT->GetVersionDate(),gROOT->GetVersionTime());
 
}


//_______________________________________________________________
TDirectory* GenerateDrawHist(TTree *tree,int level = 1)
{
// Test selections via TreeFormula
// tree is a TTree when called by stress9
// tree is a TChain when called from stres11
// This is a quite complex test checking the results of TTree::Draw
// or TChain::Draw with an explicit loop on events.
// Also a good test for the interpreter

   gROOT->cd();
   TDirectory *hfile = gDirectory;

   gBenchmark = new TBenchmark();
   gBenchmark->Start("DrawTest");
   
   // Each tree->Draw generates an histogram
   DrawSkippable(tree,"GetNtrack()","hGetNtrack",!(level>0 && gHasLibrary));

   //gBenchmark->Show("DrawTest");  gBenchmark->Start("DrawTest");

   tree->Draw("fNtrack>>hNtrack",    "","goff");
   tree->Draw("fNseg>>hNseg",        "","goff");
   tree->Draw("fTemperature>>hTemp", "","goff");

   tree->Draw("fH.GetMean()>>hHmean","","goff");
   if (level>0) tree->Draw("fH.fXaxis.fXmax>>hHAxisMax","","goff");
   if (level>0) tree->Draw("fH.fXaxis.GetXmax()>>hHAxisGetMax","","goff");
   DrawSkippable(tree,"fH.GetXaxis().GetXmax()","hHGetAxisGetMax",!(level>0));
   DrawSkippable(tree,"fH.GetXaxis().fXmax","hHGetAxisMax",!(level>0));
   DrawSkippable(tree,"GetHistogram().GetXaxis().GetXmax()","hGetHGetAxisMax",
                 !(level>0&&gHasLibrary));
   DrawSkippable(tree,"event.GetHistogram().GetXaxis().GetXmax()",
                 "hGetRefHGetAxisMax",!(level>0&&gHasLibrary));

   tree->Draw("fTracks.fPx>>hPx","fEvtHdr.fEvtNum%10 == 0","goff");
   tree->Draw("fTracks.fPy>>hPy","fEvtHdr.fEvtNum%10 == 0","goff");
   tree->Draw("fTracks.fPz>>hPz","fEvtHdr.fEvtNum%10 == 0","goff");
   tree->Draw("fRandom>>hRandom","fEvtHdr.fEvtNum%10 == 1","goff");
   tree->Draw("fMass2>>hMass2",  "fEvtHdr.fEvtNum%10 == 1","goff");
   tree->Draw("fBx>>hBx",        "fEvtHdr.fEvtNum%10 == 1","goff");
   tree->Draw("fBy>>hBy",        "fEvtHdr.fEvtNum%10 == 1","goff");
   tree->Draw("fXfirst>>hXfirst","fEvtHdr.fEvtNum%10 == 2","goff");
   tree->Draw("fYfirst>>hYfirst","fEvtHdr.fEvtNum%10 == 2","goff");
   tree->Draw("fZfirst>>hZfirst","fEvtHdr.fEvtNum%10 == 2","goff");
   tree->Draw("fXlast>>hXlast",  "fEvtHdr.fEvtNum%10 == 3","goff");
   tree->Draw("fYlast>>hYlast",  "fEvtHdr.fEvtNum%10 == 3","goff");
   tree->Draw("fZlast>>hZlast",  "fEvtHdr.fEvtNum%10 == 3","goff");
   tree->Draw("fCharge>>hCharge","fPx < 0","goff");
   tree->Draw("fNpoint>>hNpoint","fPx < 0","goff");
   tree->Draw("fValid>>hValid",  "fPx < 0","goff");

   tree->Draw("fMatrix>>hFullMatrix","","goff");
   tree->Draw("fMatrix[][0]>>hColMatrix","","goff");
   tree->Draw("fMatrix[1][]>>hRowMatrix","","goff");
   tree->Draw("fMatrix[2][2]>>hCellMatrix","","goff");

   tree->Draw("fMatrix - fVertex>>hFullOper","","goff");
   tree->Draw("fMatrix[2][1] - fVertex[5][1]>>hCellOper","","goff");
   tree->Draw("fMatrix[][1]  - fVertex[5][1]>>hColOper","","goff");
   tree->Draw("fMatrix[2][]  - fVertex[5][2]>>hRowOper","","goff");
   tree->Draw("fMatrix[2][]  - fVertex[5][]>>hMatchRowOper","","goff");
   tree->Draw("fMatrix[][2]  - fVertex[][1]>>hMatchColOper","","goff");
   tree->Draw("fMatrix[][2]  - fVertex[][]>>hRowMatOper","","goff");
   tree->Draw("fMatrix[][2]  - fVertex[5][]>>hMatchDiffOper","","goff");
   tree->Draw("fMatrix[][]   - fVertex[][]>>hFullOper2","","goff");

   // Test on variable arrays
   tree->Draw("fClosestDistance>>hClosestDistance","","goff");
   tree->Draw("fClosestDistance[2]>>hClosestDistance2","","goff");
   tree->Draw("fClosestDistance[9]>>hClosestDistance9","","goff");

   // Test variable indexing
   DrawSkippable(tree,"fClosestDistance[fNvertex/2]","hClosestDistanceIndex",
                 !(level>0));
   DrawSkippable(tree,"fPx:fPy[fNpoint]","fPy[fNpoint]>0","hPxInd",!(level>0));

   // Test of simple function calls
   DrawSkippable(tree,"sqrt(fNtrack)","hSqrtNtrack",!(level>0));   

   // Test string operations
   DrawSkippable(tree,"fEvtHdr.fEvtNum","fType==\"type1\" ","hString",!(level>0));
   DrawSkippable(tree,"fEvtHdr.fEvtNum","strstr(fType,\"1\") ","+hString",!(level>0));

   // Test binary operators
   DrawSkippable(tree,"fValid<<4","hShiftValid",!(level>0));
   DrawSkippable(tree,"((fValid<<4)>>2)","+hShiftValid",!(level>0));
   DrawSkippable(tree,"fValid&0x1","(fNvertex>10) && (fNseg<=6000)"
                 ,"hAndValid",!(level>0));

   // Test weight
   DrawSkippable(tree,"fPx","(fBx>.4) || (fBy<=-.4)","hPxBx",!(level>0));
   DrawSkippable(tree,"fPx","fBx*fBx*(fBx>.4) + fBy*fBy*(fBy<=-.4)",
                 "hPxBxWeight",!(level>0));
  
   gBenchmark->Show("DrawTest");  gBenchmark->Start("DrawTest");

   return hfile;

}
