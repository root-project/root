//macro illustrating how to animate a picture using a Timer
//Author: Rene Brun

#include "TStyle.h"
#include "TCanvas.h"
#include "TF2.h"
#include "TTimer.h"

Double_t pi;
TF2 *f2;
Float_t t = 0;
Float_t phi = 30;
void anim()
{
   gStyle->SetCanvasPreferGL(true);
   gStyle->SetFrameFillColor(42);
   TCanvas *c1 = new TCanvas("c1");
   c1->SetFillColor(17);
   pi = TMath::Pi();
   f2 = new TF2("f2","sin(2*x)*sin(2*y)*[0]",0,pi,0,pi);
   f2->SetParameter(0,1);
   f2->SetNpx(15);
   f2->SetNpy(15);
   f2->SetMaximum(1);
   f2->SetMinimum(-1);
   f2->Draw("glsurf1");
   TTimer *timer = new TTimer(20);
   timer->SetCommand("Animate()");
   timer->TurnOn();
}
void Animate()
{
   //just in case the canvas has been deleted
   if (!gROOT->GetListOfCanvases()->FindObject("c1")) return;
   t += 0.05*pi;
   f2->SetParameter(0,TMath::Cos(t));
   phi += 2;
   gPad->SetPhi(phi);
   gPad->Modified();
   gPad->Update();
}
