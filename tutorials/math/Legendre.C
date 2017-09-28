/// \file
/// \ingroup tutorial_math
/// \notebook
/// Example of first few Legendre Polynomials
///
/// Draws a graph.
///
/// \macro_image
/// \macro_code
///
/// \author Lorenzo Moneta


#include "TMath.h"
#include "TF1.h"
#include "TCanvas.h"

#include <Riostream.h>
#include "TLegend.h"
#include "TLegendEntry.h"

#include "Math/IFunction.h"
#include <cmath>
#include "TSystem.h"


void Legendre()
{
   R__LOAD_LIBRARY(libMathMore);

   TCanvas *Canvas = new TCanvas("DistCanvas", "Legendre polynomials example", 10, 10, 750, 600);
   Canvas->SetGrid();
   TLegend *leg = new TLegend(0.5, 0.7, 0.4, 0.89);
   //drawing the set of Legendre functions
   TF1* L[5];
   for(int nu = 0; nu <= 4; nu++)
   {
         L[nu]= new TF1("L_0", "ROOT::Math::legendre([0],x)", -1, 1);
         L[nu]->SetParameters(nu, 0.0);
         L[nu]->SetLineStyle(1);
         L[nu]->SetLineWidth(2);
         L[nu]->SetLineColor(nu+1);
   }
   L[0]->SetMaximum(1);
   L[0]->SetMinimum(-1);
   L[0]->SetTitle("Legendre polynomials");
   leg->AddEntry(L[0]->DrawCopy(), " L_{0}(x)", "l");
   leg->AddEntry(L[1]->DrawCopy("same"), " L_{1}(x)", "l");
   leg->AddEntry(L[2]->DrawCopy("same"), " L_{2}(x)", "l");
   leg->AddEntry(L[3]->DrawCopy("same"), " L_{3}(x)", "l");
   leg->AddEntry(L[4]->DrawCopy("same"), " L_{4}(x)", "l");
   leg->Draw();

   Canvas->cd();
}

