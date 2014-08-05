// Example describing the usage of different kinds of Associate Legendre Polynomials
// To execute the macro type in:
//
// root[0]: .x LegendreAssoc.C
//
// It draws common graphs for first 5
// Associate Legendre Polynomials
// and Spherical Associate Legendre Polynomials
// Their integrals on the range [-1, 1] are calculated
//
// Author: Magdalena Slawinska

#if defined(__CINT__) && !defined(__MAKECINT__)
{
   gSystem->CompileMacro("LegendreAssoc.C", "k");
   LegendreAssoc();
}
#else

#include "TMath.h"
#include "TF1.h"
#include "TCanvas.h"

#include <Riostream.h>
#include "TLegend.h"
#include "TLegendEntry.h"

#include "Math/IFunction.h"
#include <cmath>
#include "TSystem.h"

void LegendreAssoc()
{

  //const int n=5;
  gSystem->Load("libMathMore");

  std::cout <<"Drawing associate Legendre Polynomials.." << std::endl;
  TCanvas *Canvas = new TCanvas("DistCanvas", "Associate Legendre polynomials", 10, 10, 1000, 600);
  Canvas->SetFillColor(17);
  Canvas->Divide(2,1);
  Canvas->SetFrameFillColor(19);
  TLegend *leg1 = new TLegend(0.5, 0.7, 0.8, 0.89);
  TLegend *leg2 = new TLegend(0.5, 0.7, 0.8, 0.89);
  //leg->TLegend::SetNDC();
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//drawing the set of Legendre functions
  TF1* L[5];

    L[0]= new TF1("L_0", "ROOT::Math::assoc_legendre(1, 0,x)", -1, 1);
    L[1]= new TF1("L_1", "ROOT::Math::assoc_legendre(1, 1,x)", -1, 1);
    L[2]= new TF1("L_2", "ROOT::Math::assoc_legendre(2, 0,x)", -1, 1);
    L[3]= new TF1("L_3", "ROOT::Math::assoc_legendre(2, 1,x)", -1, 1);
    L[4]= new TF1("L_4", "ROOT::Math::assoc_legendre(2, 2,x)", -1, 1);


  TF1* SL[5];

    SL[0]= new TF1("SL_0", "ROOT::Math::sph_legendre(1, 0,x)", -TMath::Pi(), TMath::Pi());
    SL[1]= new TF1("SL_1", "ROOT::Math::sph_legendre(1, 1,x)", -TMath::Pi(), TMath::Pi());
    SL[2]= new TF1("SL_2", "ROOT::Math::sph_legendre(2, 0,x)", -TMath::Pi(), TMath::Pi());
    SL[3]= new TF1("SL_3", "ROOT::Math::sph_legendre(2, 1,x)", -TMath::Pi(), TMath::Pi());
    SL[4]= new TF1("SL_4", "ROOT::Math::sph_legendre(2, 2,x)", -TMath::Pi(), TMath::Pi() );


    Canvas->cd(1);
    gPad->SetGrid();
    gPad->SetFillColor(kWhite);
    L[0]->SetMaximum(3);
    L[0]->SetMinimum(-2);
    L[0]->SetTitle("Associate Legendre Polynomials");
    for(int nu = 0; nu < 5; nu++)
    {
      //L[nu]->SetTitle("Legendre polynomials");

      L[nu]->SetLineStyle(1);
      L[nu]->SetLineWidth(2);
      L[nu]->SetLineColor(nu+1);
    }

    leg1->AddEntry(L[0]->DrawCopy(), " P^{1}_{0}(x)", "l");
    leg1->AddEntry(L[1]->DrawCopy("same"), " P^{1}_{1}(x)", "l");
    leg1->AddEntry(L[2]->DrawCopy("same"), " P^{2}_{0}(x)", "l");
    leg1->AddEntry(L[3]->DrawCopy("same"), " P^{2}_{1}(x)", "l");
    leg1->AddEntry(L[4]->DrawCopy("same"), " P^{2}_{2}(x)", "l");
    leg1->Draw();

    Canvas->cd(2);
    gPad->SetGrid();
    gPad->SetFillColor(kWhite);
    SL[0]->SetMaximum(1);
    SL[0]->SetMinimum(-1);
    SL[0]->SetTitle("Spherical Legendre Polynomials");
    for(int nu = 0; nu < 5; nu++)
    {
      //L[nu]->SetTitle("Legendre polynomials");

      SL[nu]->SetLineStyle(1);
      SL[nu]->SetLineWidth(2);
      SL[nu]->SetLineColor(nu+1);
    }

    leg2->AddEntry(SL[0]->DrawCopy(), " P^{1}_{0}(x)", "l");
    leg2->AddEntry(SL[1]->DrawCopy("same"), " P^{1}_{1}(x)", "l");
    leg2->AddEntry(SL[2]->DrawCopy("same"), " P^{2}_{0}(x)", "l");
    leg2->AddEntry(SL[3]->DrawCopy("same"), " P^{2}_{1}(x)", "l");
    leg2->AddEntry(SL[4]->DrawCopy("same"), " P^{2}_{2}(x)", "l");
    leg2->Draw();


    //integration

    std::cout << "Calculating integrals of Associate Legendre Polynomials on [-1, 1]" << std::endl;
    double integral[5];
    for(int nu = 0; nu < 5; nu++)
    {
      integral[nu] = L[nu]->Integral(-1.0, 1.0);
      std::cout <<"Integral [-1,1] for Associated Legendre Polynomial of Degree " << nu << "\t = \t" << integral[nu] <<  std::endl;
    }



}

#endif

