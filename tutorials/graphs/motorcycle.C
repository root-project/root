#include "TString.h"
#include "TInterpreter.h"
#include <fstream>
#include "TH1.h"
#include "TGraphSmooth.h"
#include "TCanvas.h"
#include "TSystem.h"


TCanvas *vC1;
TGraph *grin, *grout;

void DrawSmooth(Int_t pad, const char *title, const char *xt, const char *yt)
{
   vC1->cd(pad);
   TH1F *vFrame = gPad->DrawFrame(0,-130,60,70);
   vFrame->SetTitle(title);
   vFrame->SetTitleSize(0.2);
   vFrame->SetXTitle(xt);
   vFrame->SetYTitle(yt);
   grin->Draw("P");
   grout->DrawClone("LPX");
}


void motorcycle()
{
/******************************************************************************
* Author: Christian Stratowa, Vienna, Austria.                                *
* Created: 26 Aug 2001                            Last modified: 29 Sep 2001  *
******************************************************************************/

// Macro to test scatterplot smoothers: ksmooth, lowess, supsmu
// as described in:
//    Modern Applied Statistics with S-Plus, 3rd Edition
//    W.N. Venables and B.D. Ripley
//    Chapter 9: Smooth Regression, Figure 9.1
//
// Example is a set of data on 133 observations of acceleration against time
// for a simulated motorcycle accident, taken from Silverman (1985).


// data taken from R library MASS: mcycle.txt
   TString dir = gSystem->UnixPathName(gInterpreter->GetCurrentMacroName());
   dir.ReplaceAll("motorcycle.C","");
   dir.ReplaceAll("/./","/");

// read file and add to fit object
   Double_t *x = new Double_t[133];
   Double_t *y = new Double_t[133];
   Double_t vX, vY;
   Int_t vNData = 0;
   ifstream vInput;
   vInput.open(Form("%smotorcycle.dat",dir.Data()));
   while (1) {
      vInput >> vX >> vY;
      if (!vInput.good()) break;
      x[vNData] = vX;
      y[vNData] = vY;
      vNData++;
   }//while
   vInput.close();
   grin = new TGraph(vNData,x,y);
   
// draw graph
   vC1 = new TCanvas("vC1","Smooth Regression",200,10,900,700);
   vC1->Divide(2,3);

// Kernel Smoother
// create new kernel smoother and smooth data with bandwidth = 2.0
   TGraphSmooth *gs = new TGraphSmooth("normal");
   grout = gs->SmoothKern(grin,"normal",2.0);
   DrawSmooth(1,"Kernel Smoother: bandwidth = 2.0","times","accel");

// redraw ksmooth with bandwidth = 5.0
   grout = gs->SmoothKern(grin,"normal",5.0);
   DrawSmooth(2,"Kernel Smoother: bandwidth = 5.0","","");

// Lowess Smoother
// create new lowess smoother and smooth data with fraction f = 2/3
   grout = gs->SmoothLowess(grin,"",0.67);
   DrawSmooth(3,"Lowess: f = 2/3","","");

// redraw lowess with fraction f = 0.2
   grout = gs->SmoothLowess(grin,"",0.2);
   DrawSmooth(4,"Lowess: f = 0.2","","");

// Super Smoother
// create new super smoother and smooth data with default bass = 0 and span = 0
   grout = gs->SmoothSuper(grin,"",0,0);
   DrawSmooth(5,"Super Smoother: bass = 0","","");

// redraw supsmu with bass = 3 (smoother curve)
   grout = gs->SmoothSuper(grin,"",3);
   DrawSmooth(6,"Super Smoother: bass = 3","","");

// cleanup
   delete [] x;
   delete [] y;
   delete gs;
}

