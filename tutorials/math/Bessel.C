/// \file
/// \ingroup tutorial_math
/// \notebook
/// Show the different kinds of Bessel functions available in ROOT
/// To execute the macro type in:
///
/// ~~~{.cpp}
/// root[0] .x Bessel.C
/// ~~~
///
/// It will create one canvas with the representation
/// of the  cylindrical and spherical Bessel functions
/// regular and modified
///
/// \macro_image
/// \macro_code
///
/// \author Magdalena Slawinska

#include "TMath.h"
#include "TF1.h"
#include "TCanvas.h"

#include <Riostream.h>
#include "TLegend.h"
#include "TLegendEntry.h"

#include "Math/IFunction.h"
#include <cmath>
#include "TSystem.h"
#include "TAxis.h"
#include "TPaveLabel.h"

void Bessel()
{
   R__LOAD_LIBRARY(libMathMore);

   TCanvas *DistCanvas = new TCanvas("DistCanvas", "Bessel functions example", 10, 10, 800, 600);
   DistCanvas->SetFillColor(17);
   DistCanvas->Divide(2, 2);
   DistCanvas->cd(1);
   gPad->SetGrid();
   gPad->SetFrameFillColor(19);
   TLegend *leg = new TLegend(0.75, 0.7, 0.89, 0.89);

   int n = 5; //number of functions in each pad
   //drawing the set of Bessel J functions
   TF1* JBessel[5];
   for(int nu = 0; nu < n; nu++)
   {
      JBessel[nu]= new TF1("J_0", "ROOT::Math::cyl_bessel_j([0],x)", 0, 10);
      JBessel[nu]->SetParameters(nu, 0.0);
      JBessel[nu]->SetTitle(""); //Bessel J functions");
      JBessel[nu]->SetLineStyle(1);
      JBessel[nu]->SetLineWidth(3);
      JBessel[nu]->SetLineColor(nu+1);
   }
   JBessel[0]->TF1::GetXaxis()->SetTitle("x");
   JBessel[0]->GetXaxis()->SetTitleSize(0.06);
   JBessel[0]->GetXaxis()->SetTitleOffset(.7);

   //setting the title in a label style
   TPaveLabel *p1 = new TPaveLabel(.0,.90 , (.0+.50),(.90+.10) , "Bessel J functions", "NDC");
   p1->SetFillColor(0);
   p1->SetTextFont(22);
   p1->SetTextColor(kBlack);

   //setting the legend
   leg->AddEntry(JBessel[0]->DrawCopy(), " J_0(x)", "l");
   leg->AddEntry(JBessel[1]->DrawCopy("same"), " J_1(x)", "l");
   leg->AddEntry(JBessel[2]->DrawCopy("same"), " J_2(x)", "l");
   leg->AddEntry(JBessel[3]->DrawCopy("same"), " J_3(x)", "l");
   leg->AddEntry(JBessel[4]->DrawCopy("same"), " J_4(x)", "l");

   leg->Draw();
   p1->Draw();

   //------------------------------------------------
   DistCanvas->cd(2);
   gPad->SetGrid();
   gPad->SetFrameFillColor(19);

   TLegend *leg2 = new TLegend(0.75, 0.7, 0.89, 0.89);
   //------------------------------------------------
   //Drawing Bessel k
   TF1* KBessel[5];
   for(int nu = 0; nu < n; nu++){
      KBessel[nu]= new TF1("J_0", "ROOT::Math::cyl_bessel_k([0],x)", 0, 10);
      KBessel[nu]->SetParameters(nu, 0.0);
      KBessel[nu]->SetTitle("Bessel K functions");
      KBessel[nu]->SetLineStyle(1);
      KBessel[nu]->SetLineWidth(3);
      KBessel[nu]->SetLineColor(nu+1);
   }
   KBessel[0]->GetXaxis()->SetTitle("x");
   KBessel[0]->GetXaxis()->SetTitleSize(0.06);
   KBessel[0]->GetXaxis()->SetTitleOffset(.7);

   //setting title
   TPaveLabel *p2 = new TPaveLabel(.0,.90 , (.0+.50),(.90+.10) , "Bessel K functions", "NDC");
   p2->SetFillColor(0);
   p2->SetTextFont(22);
   p2->SetTextColor(kBlack);

   //setting legend
   leg2->AddEntry(KBessel[0]->DrawCopy(), " K_0(x)", "l");
   leg2->AddEntry(KBessel[1]->DrawCopy("same"), " K_1(x)", "l");
   leg2->AddEntry(KBessel[2]->DrawCopy("same"), " K_2(x)", "l");
   leg2->AddEntry(KBessel[3]->DrawCopy("same"), " K_3(x)", "l");
   leg2->AddEntry(KBessel[4]->DrawCopy("same"), " K_4(x)", "l");
   leg2->Draw();
   p2->Draw();
   //------------------------------------------------
   DistCanvas->cd(3);
   gPad->SetGrid();
   gPad->SetFrameFillColor(19);
   TLegend *leg3 = new TLegend(0.75, 0.7, 0.89, 0.89);
   //------------------------------------------------
   //Drawing Bessel i
   TF1* iBessel[5];
   for(int nu = 0; nu <= 4; nu++){
      iBessel[nu]= new TF1("J_0", "ROOT::Math::cyl_bessel_i([0],x)", 0, 10);
      iBessel[nu]->SetParameters(nu, 0.0);
      iBessel[nu]->SetTitle("Bessel I functions");
      iBessel[nu]->SetLineStyle(1);
      iBessel[nu]->SetLineWidth(3);
      iBessel[nu]->SetLineColor(nu+1);
   }

   iBessel[0]->GetXaxis()->SetTitle("x");
   iBessel[0]->GetXaxis()->SetTitleSize(0.06);
   iBessel[0]->GetXaxis()->SetTitleOffset(.7);

   //setting title
   TPaveLabel *p3 = new TPaveLabel(.0,.90 , (.0+.50),(.90+.10) ,"Bessel I functions", "NDC");
   p3->SetFillColor(0);
   p3->SetTextFont(22);
   p3->SetTextColor(kBlack);

   //setting legend
   leg3->AddEntry(iBessel[0]->DrawCopy(), " I_0", "l");
   leg3->AddEntry(iBessel[1]->DrawCopy("same"), " I_1(x)", "l");
   leg3->AddEntry(iBessel[2]->DrawCopy("same"), " I_2(x)", "l");
   leg3->AddEntry(iBessel[3]->DrawCopy("same"), " I_3(x)", "l");
   leg3->AddEntry(iBessel[4]->DrawCopy("same"), " I_4(x)", "l");
   leg3->Draw();
   p3->Draw();
   //------------------------------------------------
   DistCanvas->cd(4);
   gPad->SetGrid();
   gPad->SetFrameFillColor(19);
   TLegend *leg4 = new TLegend(0.75, 0.7, 0.89, 0.89);
   //------------------------------------------------
   //Drawing sph_bessel
   TF1* jBessel[5];
   for(int nu = 0; nu <= 4; nu++){
      jBessel[nu]= new TF1("J_0", "ROOT::Math::sph_bessel([0],x)", 0, 10);
      jBessel[nu]->SetParameters(nu, 0.0);
      jBessel[nu]->SetTitle("Bessel j functions");
      jBessel[nu]->SetLineStyle(1);
      jBessel[nu]->SetLineWidth(3);
      jBessel[nu]->SetLineColor(nu+1);
   }
   jBessel[0]->GetXaxis()->SetTitle("x");
   jBessel[0]->GetXaxis()->SetTitleSize(0.06);
   jBessel[0]->GetXaxis()->SetTitleOffset(.7);

   //setting title
   TPaveLabel *p4 = new TPaveLabel(.0,.90 , (.0+.50),(.90+.10) ,"Bessel j functions", "NDC");
   p4->SetFillColor(0);
   p4->SetTextFont(22);
   p4->SetTextColor(kBlack);

   //setting legend

   leg4->AddEntry(jBessel[0]->DrawCopy(), " j_0(x)", "l");
   leg4->AddEntry(jBessel[1]->DrawCopy("same"), " j_1(x)", "l");
   leg4->AddEntry(jBessel[2]->DrawCopy("same"), " j_2(x)", "l");
   leg4->AddEntry(jBessel[3]->DrawCopy("same"), " j_3(x)", "l");
   leg4->AddEntry(jBessel[4]->DrawCopy("same"), " j_4(x)", "l");

   leg4->Draw();
   p4->Draw();

   DistCanvas->cd();
}
