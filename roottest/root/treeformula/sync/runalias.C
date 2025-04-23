/* 
 This is an example of a bug that occurs with TTree Draw in a special case.
 It occurs for some vector content only (?)
 When I use an alias to draw the vector multiplied with a non-vector
 And I have a cut on the vector
 
 TTree draw then picks up the first value of the non-vector and uses that for
 all data.
 
 This macro shows the problem on my machine:
 SUSE 11, root v5-25-01-alice
 
 No problem:
 .L example.C+
 create_tree(kFALSE)
 draw_tree()
 
 Problem:
 create_tree(kTRUE)
 draw_tree()   
 */

#include <TObject.h>
#include <TVectorD.h>
#include <TFile.h>
#include <TTree.h>
#include <TRandom.h>
#include <TCanvas.h>
#include "TH1.h"

const Int_t nHits = 10;

class MyClass:public TObject {
public:
   MyClass() : fOffset(0), fX(nHits), fY(nHits) {};
   
   Double_t  fOffset;
   TVectorD  fX;
   TVectorD  fY;
   
   ClassDefOverride(MyClass, 1)
};

ClassImp(MyClass)

void create_tree();
void draw_tree();

//________________________________________________________________________
void create_tree()
{
   TFile* file = new TFile("example.root", "RECREATE");
   // create a TTree   
   TTree *tree = new TTree("tree","example tree");
   MyClass* m = new MyClass;                               
   
   // create a branch with energy                      
   tree->Branch("MyClass", &m);                           
   
   // fill some events with random numbers             
   Int_t nevent=5;                                 
   for (Int_t iev=0;iev<nevent;iev++) {                
      
      m->fOffset = iev+1;
      for(Int_t i = 0; i < nHits; i++) {
         
         m->fY[i] = gRandom->Gaus(iev, 0.1);
         m->fX[i] = gRandom->Gaus(i, 0.1);
      }
      
      tree->Fill();  // fill the tree with the current event
   }  
   
   file->Write();
   file->Close();
}

int PlotAndCheck(TTree *tree, TCanvas *c1, int index, const char *selection) 
{
   c1->cd(index);
   tree->Draw("fOffset*fX.fElements",selection);
   TH1 *href = (TH1*)gPad->GetListOfPrimitives()->FindObject("htemp");
   
   c1->cd(index+1);
   tree->Draw("mult", selection);
   TH1 *halias = (TH1*)gPad->GetListOfPrimitives()->FindObject("htemp");
   
   if ( href->GetEntries() != halias->GetEntries() 
       || fabs(href->GetMean()-halias->GetMean()) > 0.000001  ){
      fprintf(stdout, "For %s\n",selection);
      fprintf(stdout, "The direct histogram and the alias histogram are different!\n");
      fprintf(stdout, "Entries: %f vs %f\n",href->GetEntries(),halias->GetEntries());
      fprintf(stdout, "Mean   : %f vs %f\n",href->GetMean(),halias->GetMean());
   }
   return index+2;
}

//________________________________________________________________________
void draw_tree()
{
   TFile* file = TFile::Open("example.root");
   // create a TTree   
   TTree *tree = (TTree*)file->Get("tree");
   
   tree->SetAlias("mult", "fOffset*fX.fElements");
   
   TCanvas* c1 = new TCanvas("c1");
   c1->Divide(2,3);
   c1->cd(1);
   tree->Draw("fOffset*fX.fElements");
   
   c1->cd(2);
   tree->Draw("mult");
   
   PlotAndCheck(tree,c1,3,"fX.fElements > 1.0");
   PlotAndCheck(tree,c1,5,"fY.fElements > 1.0");
   
}

void runalias() {
   create_tree();
   draw_tree();
}
