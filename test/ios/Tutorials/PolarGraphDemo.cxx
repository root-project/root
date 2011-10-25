#include "TGraphPolar.h"
#include "IOSPad.h"
#include "TFrame.h"
#include "TMath.h"

#include "PolarGraphDemo.h"

namespace ROOT {
namespace iOS {
namespace Demos {

namespace {

//______________________________________________________________________________
Double_t FPlot(Double_t x)
{
   return TMath::Cos(2. * x) * TMath::Cos(20. * x);
}

}

//______________________________________________________________________________
PolarGraphDemo::PolarGraphDemo()
{
   const Double_t xMin = 0;
   const Double_t xMax = TMath::TwoPi();
   const Double_t xStep = (xMax - xMin) / kNPointsAFL;

   Double_t xAFL[kNPointsAFL];
   Double_t yAFL[kNPointsAFL];
  
   for (UInt_t i = 0; i < kNPointsAFL; ++i) {
      xAFL[i] = i * xStep + xMin;
      yAFL[i] = FPlot(xAFL[i]);
   }

   fPolarAFL.reset(new TGraphPolar(kNPointsAFL, xAFL, yAFL));
   fPolarAFL->SetLineColor(2);
   fPolarAFL->SetLineWidth(0.2);
   fPolarAFL->SetFillStyle(3012);
   fPolarAFL->SetFillColor(2);
   
   Double_t xCP[kNPointsCP];
   Double_t yCP[kNPointsCP];

   for (Int_t i = 0; i < kNPointsCP; ++i) {
      xCP[i] = xAFL[kNPointsAFL / kNPointsCP * i];
      yCP[i] = yAFL[kNPointsAFL / kNPointsCP * i];
   }

   fPolarCP.reset(new TGraphPolar(kNPointsCP, xCP, yCP));
   fPolarCP->SetMarkerStyle(kFullStar);

   fPolarCP->SetMarkerSize(2);
   fPolarCP->SetMarkerColor(4);
   fPolarCP->SetLineColor(4);
}

//______________________________________________________________________________
PolarGraphDemo::~PolarGraphDemo()
{
   //Just for std::auto_ptr's dtor.
}

//______________________________________________________________________________
void PolarGraphDemo::AdjustPad(Pad *pad)
{
   pad->SetFillColor(0);
   pad->GetFrame()->SetFillColor(0);
   pad->GetFrame()->SetBorderSize(1);
   pad->GetFrame()->SetBorderMode(0);
}

//______________________________________________________________________________
void PolarGraphDemo::PresentDemo()
{
   fPolarAFL->Draw("AFL");
   fPolarCP->Draw("CP");
}

}
}
}
