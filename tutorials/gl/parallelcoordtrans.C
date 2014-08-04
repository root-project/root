// Script illustrating the use of transparency with ||-Coord.
// It displays the same data set twice. The first time without transparency and
// the second time with transparency. On the second plot, several clusters
// appear.

//Authors: by Timur Pocheptsov, based on macro by Olivier Couet.


//All these includes are (only) to make the macro
//ACLiCable.
#include <cassert>

#include "TParallelCoordVar.h"
#include "TParallelCoord.h"
#include "TNtuple.h"
#include "TCanvas.h"
#include "TRandom.h"
#include "TColor.h"
#include "TStyle.h"
#include "TError.h"
#include "TList.h"
#include "TROOT.h"

namespace ROOT {
namespace GLTutorials {

Double_t r1, r2, r3, r4, r5, r6, r7, r8, r9;
TRandom r;

//______________________________________________________________________
void generate_random(Int_t i)
{
   const Double_t dr = 3.5;

   r.Rannor(r1, r4);
   r.Rannor(r7, r9);

   r2 = (2 * dr * r.Rndm(i)) - dr;
   r3 = (2 * dr * r.Rndm(i)) - dr;
   r5 = (2 * dr * r.Rndm(i)) - dr;
   r6 = (2 * dr * r.Rndm(i)) - dr;
   r8 = (2 * dr * r.Rndm(i)) - dr;
}

}//GLTutorials
}//ROOT

void parallelcoordtrans()
{
   //This macro shows how to use parallel coords and semi-transparent lines
   //(the system color is updated with alpha == 0.01 (1% opaque).

   using namespace ROOT::GLTutorials;

   Double_t s1x = 0., s1y = 0., s1z = 0.;
   Double_t s2x = 0., s2y = 0., s2z = 0.;
   Double_t s3x = 0., s3y = 0., s3z = 0.;

   gStyle->SetCanvasPreferGL(kTRUE);
   TCanvas *c1 = new TCanvas("parallel coors", "parallel coords", 0, 0, 900, 1000);

   TNtuple * const nt = new TNtuple("nt", "Demo ntuple", "x:y:z:u:v:w:a:b:c");

   for (Int_t i = 0; i < 1500; ++i) {
      r.Sphere(s1x, s1y, s1z, 0.1);
      r.Sphere(s2x, s2y, s2z, 0.2);
      r.Sphere(s3x, s3y, s3z, 0.05);

      generate_random(i);
      nt->Fill(r1, r2, r3, r4, r5, r6, r7, r8, r9);

      generate_random(i);
      nt->Fill(s1x, s1y, s1z, s2x, s2y, s2z, r7, r8, r9);

      generate_random(i);
      nt->Fill(r1, r2, r3, r4, r5, r6, r7, s3y, r9);

      generate_random(i);
      nt->Fill(s2x - 1, s2y - 1, s2z, s1x + .5, s1y + .5, s1z + .5, r7, r8, r9);

      generate_random(i);
      nt->Fill(r1, r2, r3, r4, r5, r6, r7, r8, r9);

      generate_random(i);
      nt->Fill(s1x + 1, s1y + 1, s1z + 1, s3x - 2, s3y - 2, s3z - 2, r7, r8, r9);

      generate_random(i);
      nt->Fill(r1, r2, r3, r4, r5, r6, s3x, r8, s3z);
   }

   c1->Divide(1, 2);
   c1->cd(1);

   // ||-Coord plot without transparency
   nt->Draw("x:y:z:u:v:w:a:b:c", "", "para");
   TParallelCoord * const para1 = (TParallelCoord*)gPad->GetListOfPrimitives()->FindObject("ParaCoord");
   assert(para1 != 0 && "parallelcoordtrans, 'ParaCoord' is null");

   para1->SetLineColor(25);
   TParallelCoordVar *pcv = (TParallelCoordVar*)para1->GetVarList()->FindObject("x");
   pcv->SetHistogramHeight(0.);

   pcv = (TParallelCoordVar*)para1->GetVarList()->FindObject("y");
   pcv->SetHistogramHeight(0.);

   pcv = (TParallelCoordVar*)para1->GetVarList()->FindObject("z");
   pcv->SetHistogramHeight(0.);

   pcv = (TParallelCoordVar*)para1->GetVarList()->FindObject("a");
   pcv->SetHistogramHeight(0.);

   pcv = (TParallelCoordVar*)para1->GetVarList()->FindObject("b");
   pcv->SetHistogramHeight(0.);

   pcv = (TParallelCoordVar*)para1->GetVarList()->FindObject("c");
   pcv->SetHistogramHeight(0.);

   pcv = (TParallelCoordVar*)para1->GetVarList()->FindObject("u");
   pcv->SetHistogramHeight(0.);

   pcv = (TParallelCoordVar*)para1->GetVarList()->FindObject("v");
   pcv->SetHistogramHeight(0.);

   pcv = (TParallelCoordVar*)para1->GetVarList()->FindObject("w");
   pcv->SetHistogramHeight(0.);

   // ||-Coord plot with transparency
   // We modify a 'system' color! You'll probably
   // have to restart ROOT or reset this color later.
   TColor * const col26 = gROOT->GetColor(26);
   assert(col26 != 0 && "parallelcoordtrans, color with index 26 not found");

   col26->SetAlpha(0.01);

   c1->cd(2);
   nt->Draw("x:y:z:u:v:w:a:b:c","","para");
   TParallelCoord * const para2 = (TParallelCoord*)gPad->GetListOfPrimitives()->FindObject("ParaCoord");
   assert(para2 != 0 && "parallelcoordtrans, 'ParaCoord' is null");

   para2->SetLineColor(26);

   pcv = (TParallelCoordVar*)para2->GetVarList()->FindObject("x");
   pcv->SetHistogramHeight(0.);

   pcv = (TParallelCoordVar*)para2->GetVarList()->FindObject("y");
   pcv->SetHistogramHeight(0.);

   pcv = (TParallelCoordVar*)para2->GetVarList()->FindObject("z");
   pcv->SetHistogramHeight(0.);

   pcv = (TParallelCoordVar*)para2->GetVarList()->FindObject("a");
   pcv->SetHistogramHeight(0.);

   pcv = (TParallelCoordVar*)para2->GetVarList()->FindObject("b");
   pcv->SetHistogramHeight(0.);

   pcv = (TParallelCoordVar*)para2->GetVarList()->FindObject("c");
   pcv->SetHistogramHeight(0.);

   pcv = (TParallelCoordVar*)para2->GetVarList()->FindObject("u");
   pcv->SetHistogramHeight(0.);

   pcv = (TParallelCoordVar*)para2->GetVarList()->FindObject("v");
   pcv->SetHistogramHeight(0.);

   pcv = (TParallelCoordVar*)para2->GetVarList()->FindObject("w");
   pcv->SetHistogramHeight(0.);
}
