/// \file
/// \ingroup tutorial_tree
/// \notebook -nodraw
/// Script illustrating the use of the TParalleCoord class
///
/// \macro_image
/// \macro_code
///
/// \author  Bastien Dallapiazza

#include "TFile.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TRandom.h"
#include "TNtuple.h"
#include "TParallelCoord.h"
#include "TParallelCoordVar.h"
#include "TParallelCoordRange.h"

Double_t r1,r2,r3,r4,r5,r6,r7,r8,r9;
Double_t dr = 3.5;
TRandom *r;

void generate_random(Int_t i) {
   r1 = (2*dr*r->Rndm(i))-dr;
   r2 = (2*dr*r->Rndm(i))-dr;
   r7 = (2*dr*r->Rndm(i))-dr;
   r9 = (2*dr*r->Rndm(i))-dr;
   r4 = (2*dr*r->Rndm(i))-dr;
   r3 = (2*dr*r->Rndm(i))-dr;
   r5 = (2*dr*r->Rndm(i))-dr;
   r6 = (2*dr*r->Rndm(i))-dr;
   r8 = (2*dr*r->Rndm(i))-dr;
}

void parallelcoord() {

   TNtuple *nt = NULL;

   Double_t s1x, s1y, s1z;
   Double_t s2x, s2y, s2z;
   Double_t s3x, s3y, s3z;
   r = new TRandom();;

   new TCanvas("c1", "c1",0,0,800,700);

   nt = new TNtuple("nt","Demo ntuple","x:y:z:u:v:w");

   for (Int_t i=0; i<20000; i++) {
      r->Sphere(s1x, s1y, s1z, 0.1);
      r->Sphere(s2x, s2y, s2z, 0.2);
      r->Sphere(s3x, s3y, s3z, 0.05);

      generate_random(i);
      nt->Fill(r1, r2, r3, r4, r5, r6);

      generate_random(i);
      nt->Fill(s1x, s1y, s1z, s2x, s2y, s2z);

      generate_random(i);
      nt->Fill(r1, r2, r3, r4, r5, r6);

      generate_random(i);
      nt->Fill(s2x-1, s2y-1, s2z, s1x+.5, s1y+.5, s1z+.5);

      generate_random(i);
      nt->Fill(r1, r2, r3, r4, r5, r6);

      generate_random(i);
      nt->Fill(s1x+1, s1y+1, s1z+1, s3x-2, s3y-2, s3z-2);

      generate_random(i);
      nt->Fill(r1, r2, r3, r4, r5, r6);
   }
   nt->Draw("x:y:z:u:v:w","","para",5000);
   TParallelCoord* para = (TParallelCoord*)gPad->GetListOfPrimitives()->FindObject("ParaCoord");
   para->SetDotsSpacing(5);
   TParallelCoordVar* firstaxis = (TParallelCoordVar*)para->GetVarList()->FindObject("x");
   firstaxis->AddRange(new TParallelCoordRange(firstaxis,0.846018,1.158469));
   para->AddSelection("violet");
   para->GetCurrentSelection()->SetLineColor(kViolet);
   firstaxis->AddRange(new TParallelCoordRange(firstaxis,-0.169447,0.169042));
   para->AddSelection("Orange");
   para->GetCurrentSelection()->SetLineColor(kOrange+9);
   firstaxis->AddRange(new TParallelCoordRange(firstaxis,-1.263024,-0.755292));
}
