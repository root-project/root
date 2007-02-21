// @(#)root/graf:$Name:  $:$Id: TGraphPolar.cxx,v 1.10 2007/02/15 15:04:40 brun Exp $
// Author: Sebastian Boser, Mathieu Demaret 02/02/06

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//______________________________________________________________________________
//
//  TGraphPolar creates a polar graph (including error bars). A TGraphPolar is
//  a TGraphErrors represented in polar coordinates.
//  It uses the class TGraphPolargram to draw the polar axis.
//  "r" is used as radial parameter, and "theta" as polar paramater
//
// Example:
//
// {
//    TCanvas * CPol = new TCanvas("CPol","TGraphPolar Examples",600,600);
//
//    Double_t rmin=0;
//    Double_t rmax=TMath::Pi()*2;
//    Double_t r[1000];
//    Double_t theta[1000];
//
//    TF1 * fp1 = new TF1("fplot","cos(x)",rmin,rmax);
//    for (Int_t ipt = 0; ipt < 1000; ipt++) {
//       r[ipt] = ipt*(rmax-rmin)/1000+rmin;
//       theta[ipt] = fp1->Eval(r[ipt]);
//    }
//    TGraphPolar * grP1 = new TGraphPolar(1000,r,theta);
//    grP1->SetLineColor(2);
//    grP1->Draw("AOL");
// }
//Begin_Html
/*
<img src="gif/graphpol.gif">
*/
//End_Html
//


#include "TGaxis.h"
#include "TGraphPolar.h"
#include "THLimitsFinder.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TVirtualPad.h"
#include "TROOT.h"
#include "TLatex.h"
#include "TEllipse.h"
#include "TH1.h"
#include "TMath.h"


ClassImp(TGraphPolar);


//______________________________________________________________________________
TGraphPolar::TGraphPolar() : TGraphErrors()
{
   // TGraphPolar default constructor.
}


//______________________________________________________________________________
TGraphPolar::TGraphPolar(Int_t n, const Double_t* r, const Double_t* theta,
                                  const Double_t *er, const Double_t* etheta)
  : TGraphErrors(n,r,theta,er,etheta)
{
   // TGraphPolar constructor.
   //
   // n      : number of points.
   // r      : radial values.
   // theta  : angular values.
   // er     : errors on radial values.
   // etheta : errors on angular values.

   fXpol       = 0;
   fYpol       = 0;
   fPolargram  = 0;
   fOptionAxis = kFALSE;
   SetEditable(kFALSE);
}


//______________________________________________________________________________
TGraphPolar::~TGraphPolar()

{
   // TGraphPolar destructor.

   if (fXpol) delete fXpol;
   if (fYpol) delete fYpol;
}


//______________________________________________________________________________
Int_t TGraphPolar::DistancetoPrimitive(Int_t px, Int_t py)

{
   // Return the distance in pixel between the GraphPolar and (px,py).

   // Swap Polar and orthognal coordinates.
   Double_t* xold = fX; fX = fXpol;
   Double_t* yold = fY; fY = fYpol;

   // Now get proper distance.
   Int_t dist = TGraphErrors::DistancetoPrimitive(px,py);

   // Swap back.
   fX = xold;
   fY = yold;
   return dist;
}


//______________________________________________________________________________
void TGraphPolar::Draw(Option_t* options)

{
   // Draw TGraphPolar.

   // Process options
   TString opt = options;
   opt.ToUpper();

   // Ignore same
   opt.ReplaceAll("SAME","");

   // ReDraw polargram if required by options
   if (opt.Contains("A")) fOptionAxis = kTRUE;
   opt.ReplaceAll("A","");

   AppendPad(opt);
}


//______________________________________________________________________________
void TGraphPolar::ExecuteEvent(Int_t /*event*/, Int_t /*px*/, Int_t /*py*/)

{
   //Indicate that there is something to click here

   gPad->SetCursor(kHand);
}


//______________________________________________________________________________
void TGraphPolar::Paint(Option_t* options)

{
   // Paint TGraphPolar.
   //
   // "options" can have the following values:
   //    - "O" Polar labels are paint orthogonally to the polargram radius.
   //    - "P" Polymarker are paint at each point position.
   //    - "E" Paint error bars.
   //    - "F" Paint fill area (closed polygon).
   //    - "A" Force axis redrawing even if a polagram already exists.

   Int_t ipt, i;
   Double_t rwrmin, rwrmax, rwtmin, rwtmax;
   if (fNpoints<1) return;
   TString opt = options;
   opt.ToUpper();

   // Check for existing TGraphPolargram in the Pad
   if (gPad) {
      // Existing polargram
      if (fPolargram)
         if (!gPad->FindObject(fPolargram->GetName())) fPolargram=0;
      if (!fPolargram) {
         // Find any other Polargram in the Pad
         TListIter padObjIter(gPad->GetListOfPrimitives());
         while (TObject* AnyObj = padObjIter.Next()) {
            if (TString(AnyObj->ClassName()).CompareTo("TGraphPolargram",
                                                      TString::kExact)==0)
            fPolargram = (TGraphPolargram*)AnyObj;
         }
      }
   }

   // Get new polargram range if necessary.
   if (!fPolargram) {

      // Get range, initialize with first/last value
      rwrmin = fY[0]; rwrmax = fY[fNpoints-1];
      rwtmin = fX[0]; rwtmax = fX[fNpoints-1];

      for (ipt = 0; ipt < fNpoints; ipt++) {
         // Check for errors if available
         if (fEX) {
            if (fX[ipt] -fEX[ipt] < rwtmin) rwtmin = fX[ipt]-fEX[ipt];
            if (fX[ipt] +fEX[ipt] > rwtmax) rwtmax = fX[ipt]+fEX[ipt];
         } else {
            if (fX[ipt] < rwtmin) rwtmin=fX[ipt];
            if (fX[ipt] > rwtmax) rwtmax=fX[ipt];
         }
         if (fEY) {
            if (fY[ipt] -fEY[ipt] < rwrmin) rwrmin = fY[ipt]-fEY[ipt];
            if (fY[ipt] +fEY[ipt] > rwrmax) rwrmax = fY[ipt]+fEY[ipt];
         } else {
            if (fY[ipt] < rwrmin) rwrmin=fY[ipt];
            if (fY[ipt] > rwrmax) rwrmax=fY[ipt];
         }
      }

      // Add radial and Polar margins.
      if (rwrmin == rwrmax) rwrmax += 1.;
      if (rwtmin == rwtmax) rwtmax += 1.;
      Double_t dr = (rwrmax-rwrmin);
      Double_t dt = (rwtmax-rwtmin);
      rwrmax += 0.1*dr;
      rwrmin -= 0.1*dr;

      // Assume equaly spaced points for full 2*Pi.
      rwtmax += dt/fNpoints;
   } else {
      rwrmin = fPolargram->GetRMin();
      rwrmax = fPolargram->GetRMax();
      rwtmin = fPolargram->GetTMin();
      rwtmax = fPolargram->GetTMax();
   }

   if ((!fPolargram)||fOptionAxis) {
      // Draw Polar coord system
      fPolargram = new TGraphPolargram("Polargram",rwrmin,rwrmax,rwtmin,rwtmax);
      if (opt.Contains("O")) fPolargram->SetBit(TGraphPolargram::kLabelOrtho);
      else fPolargram->ResetBit(TGraphPolargram::kLabelOrtho);
      fPolargram->Draw(opt);
      fOptionAxis = kFALSE;   //Prevent redrawing
   }

   // Convert points to polar.
   if (fXpol) delete fXpol;
   if (fYpol) delete fYpol;
   fXpol = new Double_t[fNpoints];
   fYpol = new Double_t[fNpoints];

   // Project theta in [0,2*Pi] and radius in [0,1].
   Double_t radiusNDC = rwrmax-rwrmin;
   Double_t thetaNDC  = (rwtmax-rwtmin)/(2*TMath::Pi());

   // Draw the error bars.
   // Y errors are lines, but X errors are pieces of circles.
   if (opt.Contains("E")) {
      if (fEY) {
         for (i=0; i<fNpoints; i++) {
            Double_t eymin, eymax, exmin,exmax;
            exmin = (fY[i]-fEY[i]-rwrmin)/radiusNDC*
                     TMath::Cos((fX[i]-rwtmin)/thetaNDC);
            eymin = (fY[i]-fEY[i]-rwrmin)/radiusNDC*
                     TMath::Sin((fX[i]-rwtmin)/thetaNDC);
            exmax = (fY[i]+fEY[i]-rwrmin)/radiusNDC*
                     TMath::Cos((fX[i]-rwtmin)/thetaNDC);
            eymax = (fY[i]+fEY[i]-rwrmin)/radiusNDC*
                     TMath::Sin((fX[i]-rwtmin)/thetaNDC);
            TAttLine::Modify();
            gPad->PaintLine(exmin,eymin,exmax,eymax);
         }
      }
      if (fEX) {
         for (i=0; i<fNpoints; i++) {
            Double_t rad = (fY[i]-rwrmin)/radiusNDC;
            Double_t phimin = (fX[i]-fEX[i]-rwtmin)/thetaNDC*180/TMath::Pi();
            Double_t phimax = (fX[i]+fEX[i]-rwtmin)/thetaNDC*180/TMath::Pi();
            TAttLine::Modify();
            fPolargram->PaintCircle(0,0,rad,phimin,phimax,0);
         }
      }
   }

   // Draw the graph itself.
   if (!(gPad->GetLogx()) && !(gPad->GetLogy())) {
      Double_t a, b, c=1, x1, x2, y1, y2, discr, norm1, norm2, xts, yts;
      Bool_t previouspointin = kFALSE;
      Double_t norm = 0;
      Double_t xt   = 0;
      Double_t yt   = 0 ;
      Int_t j       = -1;
      for (i=0; i<fNpoints; i++) {
         if (fPolargram->fRadian) {c=1;}
         if (fPolargram->fDegree) {c=180/TMath::Pi();}
         if (fPolargram->fGrad) {c=100/TMath::Pi();}
         xts  = xt;
         yts  = yt;
         xt   = (fY[i]-rwrmin)/radiusNDC*TMath::Cos(c*(fX[i]-rwtmin)/thetaNDC);
         yt   = (fY[i]-rwrmin)/radiusNDC*TMath::Sin(c*(fX[i]-rwtmin)/thetaNDC);
         norm = sqrt(xt*xt+yt*yt);
         // Check if points are in the main circle.
         if ( norm <= 1) {
            // We check that the previous point was in the circle too.
            // We record new point position.
            if (!previouspointin) {
               j++;
               fXpol[j] = xt;
               fYpol[j] = yt;
            } else {
               a = (yt-yts)/(xt-xts);
               b = yts-a*xts;
               discr = 4*(a*a-b*b+1);
               x1 = (-2*a*b+sqrt(discr))/(2*(a*a+1));
               x2 = (-2*a*b-sqrt(discr))/(2*(a*a+1));
               y1 = a*x1+b;
               y2 = a*x2+b;
               norm1 = sqrt((x1-xt)*(x1-xt)+(y1-yt)*(y1-yt));
               norm2 = sqrt((x2-xt)*(x2-xt)+(y2-yt)*(y2-yt));
               previouspointin = kFALSE;
               j = 0;
               if (norm1 < norm2) {
                  fXpol[j] = x1;
                  fYpol[j] = y1;
               } else {
                  fXpol[j] = x2;
                  fYpol[j] = y2;
               }
               j++;
               fXpol[j] = xt;
               fYpol[j] = yt;
               TGraph::PaintGraph(j+1, fXpol, fYpol, opt);
            }
         } else {
            // We check that the previous point was in the circle.
            // We record new point position
            if (j>=1 && !previouspointin) {
               a = (yt-fYpol[j])/(xt-fXpol[j]);
               b = fYpol[j]-a*fXpol[j];
               previouspointin = kTRUE;
               discr = 4*(a*a-b*b+1);
               x1 = (-2*a*b+sqrt(discr))/(2*(a*a+1));
               x2 = (-2*a*b-sqrt(discr))/(2*(a*a+1));
               y1 = a*x1+b;
               y2 = a*x2+b;
               norm1 = sqrt((x1-xt)*(x1-xt)+(y1-yt)*(y1-yt));
               norm2 = sqrt((x2-xt)*(x2-xt)+(y2-yt)*(y2-yt));
               j++;
               if (norm1 < norm2) {
                  fXpol[j] = x1;
                  fYpol[j] = y1;
               } else {
                  fXpol[j] = x2;
                  fYpol[j] = y2;
               }
               TGraph::PaintGraph(j+1, fXpol, fYpol, opt);
            }
            j=-1;
         }
      }
      if (j>=1) {
         // If the last point is in the circle, we draw the last serie of point.
         TGraph::PaintGraph(j+1, fXpol, fYpol, opt);
      }
   } else {
      for (i=0; i<fNpoints; i++) {
         fXpol[i] = TMath::Abs((fY[i]-rwrmin)/radiusNDC*TMath::Cos((fX[i]-rwtmin)/thetaNDC)+1);
         fYpol[i] = TMath::Abs((fY[i]-rwrmin)/radiusNDC*TMath::Sin((fX[i]-rwtmin)/thetaNDC)+1);
      }
      TGraph::PaintGraph(fNpoints, fXpol, fYpol,opt);
   }
   PaintTitle();
}


//______________________________________________________________________________
void TGraphPolar::PaintTitle()

{
   // Paint the title.

   if (TestBit(TH1::kNoTitle)) return;
   Int_t nt = strlen(GetTitle());
   TPaveText *title = 0;
   TObject *obj;
   TIter next(gPad->GetListOfPrimitives());
   while ((obj = next())) {
      if (!obj->InheritsFrom(TPaveText::Class())) continue;
      title = (TPaveText*)obj;
      if (strcmp(title->GetName(),"title")) {title = 0; continue;}
      break;
   }
   if (nt == 0 || gStyle->GetOptTitle() <= 0) {
      if (title) delete title;
      return;
   }
   Double_t ht = gStyle->GetTitleH();
   Double_t wt = gStyle->GetTitleW();
   if (ht <= 0) ht = 1.1*gStyle->GetTitleFontSize();
   if (ht <= 0) ht = 0.05;
   if (wt <= 0) {
      TLatex l;
      l.SetTextSize(ht);
      l.SetTitle(GetTitle());
      // Adjustment in case the title has several lines (#splitline)
      ht = TMath::Max(ht, 1.2*l.GetYsize()/(gPad->GetY2() - gPad->GetY1()));
      Double_t wndc = l.GetXsize()/(gPad->GetX2() - gPad->GetX1());
      wt = TMath::Min(0.7, 0.02+wndc);
   }
   if (title) {
      TText *t0 = (TText*)title->GetLine(0);
      if (t0) {
         if (!strcmp(t0->GetTitle(),GetTitle())) return;
         t0->SetTitle(GetTitle());
         if (wt > 0) title->SetX2NDC(title->GetX1NDC()+wt);
      }
      return;
   }

   Int_t talh = gStyle->GetTitleAlign()/10;
   if (talh < 1) talh = 1; if (talh > 3) talh = 3;
   Int_t talv = gStyle->GetTitleAlign()%10;
   if (talv < 1) talv = 1; if (talv > 3) talv = 3;

   Double_t xpos, ypos;
   xpos = gStyle->GetTitleX();
   ypos = gStyle->GetTitleY();

   if (talh == 2) xpos = xpos-wt/2.;
   if (talh == 3) xpos = xpos-wt;
   if (talv == 2) ypos = ypos+ht/2.;
   if (talv == 1) ypos = ypos+ht;

   TPaveText *ptitle = new TPaveText(xpos, ypos-ht, xpos+wt, ypos,"blNDC");

   // Box with the histogram title.
   ptitle->SetFillColor(gStyle->GetTitleFillColor());
   ptitle->SetFillStyle(gStyle->GetTitleStyle());
   ptitle->SetName("title");
   ptitle->SetBorderSize(gStyle->GetTitleBorderSize());
   ptitle->SetTextColor(gStyle->GetTitleTextColor());
   ptitle->SetTextFont(gStyle->GetTitleFont(""));
   if (gStyle->GetTitleFont("")%10 > 2)
   ptitle->SetTextSize(gStyle->GetTitleFontSize());
   ptitle->AddText(GetTitle());
   ptitle->SetBit(kCanDelete);
   ptitle->Draw();
   ptitle->Paint();
}


//______________________________________________________________________________
void TGraphPolar::SetMaxPolar(Double_t maximum)
{
   // Set maximum Polar.

   if (fPolargram) fPolargram->ChangeRangePolar(fPolargram->GetTMin(),maximum);
}


//______________________________________________________________________________
void TGraphPolar::SetMaxRadial(Double_t maximum)
{
   // Set maximum radial at the intersection of the positive X axis part and
   // the circle.

   if (fPolargram) fPolargram->SetRangeRadial(fPolargram->GetRMin(),maximum);
}


//______________________________________________________________________________
void TGraphPolar::SetMinPolar(Double_t minimum)
{
   // Set minimum Polar.

   if (fPolargram) fPolargram->ChangeRangePolar(minimum, fPolargram->GetTMax());
}


//______________________________________________________________________________
void TGraphPolar::SetMinRadial(Double_t minimum)
{
   // Set minimum radial in the center of the circle.

   if (fPolargram) fPolargram->SetRangeRadial(minimum, fPolargram->GetRMax());
}


ClassImp(TGraphPolargram);


//______________________________________________________________________________
TGraphPolargram::TGraphPolargram(const char* name, Double_t rmin, Double_t rmax,
                                 Double_t tmin, Double_t tmax):
                                 TNamed(name,"Polargram")
{
   // TGraphPolargram Constructor.

   fAxisAngle        = 0;
   fCutRadial        = 0;
   fDegree           = kFALSE;
   fGrad             = kFALSE;
   fLineStyle        = 3;
   fNdivRad          = 508;
   fNdivPol          = 508;
   fPolarLabelColor  = 1;
   fPolarLabelFont   = 62;
   fPolarOffset      = 0.04;
   fPolarTextSize    = 0.04;
   fRadialOffset     = 0.025;
   fRadian           = kTRUE;
   fRadialLabelColor = 1;
   fRadialLabelFont  = 62;
   fRadialTextSize   = 0.035;
   fRwrmax           = rmax;
   fRwrmin           = rmin;
   fRwtmin           = tmin;
   fRwtmax           = tmax;
   fTickpolarSize    = 0.02;
}


//______________________________________________________________________________
TGraphPolargram::~TGraphPolargram()
{
   // TGraphPolargram destructor.
}


//______________________________________________________________________________
void TGraphPolargram::ChangeRangePolar(Double_t tmin, Double_t tmax)
{
   // Set the Polar range.
   // tmin is the start number.
   // tmax is the end number.

   if (tmin < tmax) {
      fRwtmin = tmin;
      fRwtmax = tmax;
   }
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
Int_t TGraphPolargram::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Everything within the circle belongs to the TGraphPolargram.

   Int_t i;
   Double_t x = gPad->AbsPixeltoX(px);
   Double_t y = gPad->AbsPixeltoY(py);

   // Check if close to a (major) radial line.
   Double_t rad = TMath::Sqrt(x*x+y*y);
   Int_t div    = (Int_t)rad*(fNdivRad%100);
   Double_t dr  = TMath::Min(TMath::Abs(rad-div*1./(fNdivRad%100)),
                             TMath::Abs(rad-(div+1)*1./(fNdivRad%100)));
   Int_t drad   = gPad->XtoPixel(dr)-gPad->XtoPixel(0);

   // Check if close to a (major) Polar line.
   // This is not a proper calculation, but rather fast.
   Int_t dt = kMaxPixel;
   for (i=0; i<(fNdivPol%100); i++) {
      Double_t theta = i*2*TMath::Pi()/(fNdivPol%100);

      // Attention: px,py in pixel units, line given in user coordinates.
      Int_t dthis = DistancetoLine(px,py,0.,0.,TMath::Cos(theta),
                                               TMath::Sin(theta));

      // Fails if we are outside box discribed by the line.
      // (i.e for all hor/vert lines)
      if (dthis==9999) {

         // Outside -> Get distance to endpoint of line.
         if (rad>1) {
            dthis = (Int_t)TMath::Sqrt(
                    TMath::Power(px-gPad->XtoPixel(TMath::Cos(theta)),2)+
                    TMath::Power(py-gPad->YtoPixel(TMath::Sin(theta)),2));
         } else {

            // Check for horizontal line.
            if (((TMath::Abs(theta-TMath::Pi())<0.1) &&
                ((px-gPad->XtoPixel(0))<0))          ||
                ((TMath::Abs(theta)<0.1)             &&
                ((px-gPad->XtoPixel(0))>0))) {
               dthis = TMath::Abs(py-gPad->YtoPixel(0.));
            }

            //Check for vertical line.
            if (((TMath::Abs(theta-TMath::PiOver2())<0.1)   &&
               ((py-gPad->YtoPixel(0))>0))                 ||
                ((TMath::Abs(theta-3*TMath::PiOver2())<0.1) &&
                                  (py-gPad->YtoPixel(0))<0)) {
               dthis = TMath::Abs(px-gPad->XtoPixel(0.));
            }
            if (dthis==9999) {

               // Inside, but out of box for nonorthogonal line ->
               // get distance to start point.
               dthis = (Int_t)TMath::Sqrt(
                       TMath::Power(px-gPad->XtoPixel(0.),2)+
                       TMath::Power(py-gPad->YtoPixel(0.),2));
            }
         }
      }

      // Take distance to closes line.
      dt = TMath::Min(dthis,dt);
   }
   return TMath::Min(drad, dt);
}


//______________________________________________________________________________
void TGraphPolargram::Draw(Option_t* options)
{
   // Draw Polargram.

   Paint(options);
   AppendPad(options);
}


//______________________________________________________________________________
void TGraphPolargram::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Indicate that there is something to click here.

   Int_t kMaxDiff = 20;
   static Int_t d1, d2, d3, px1, py1, px3, py3, px4, py4;
   static Bool_t p1, p2, p3, p4, p5, p6, p7, p8;
   Double_t px2, py2;
   p2 = p3 = p4 = p5 = p6 = p7 = p8 = kFALSE;
   if (!gPad->IsEditable()) return;
   switch (event) {
      case kMouseMotion:
         px1 = gPad->XtoAbsPixel(TMath::Cos(GetAngle()));
         py1 = gPad->YtoAbsPixel(TMath::Sin(GetAngle()));
         d1  = TMath::Abs(px1 - px) + TMath::Abs(py1-py); //simply take sum of pixels differences
         p1  = kFALSE;
         px2 = gPad->XtoAbsPixel(-1);
         py2 = gPad->YtoAbsPixel(1);
         d2  = (Int_t)(TMath::Abs(px2 - px) + TMath::Abs(py2 - py)) ;
         px3 = gPad->XtoAbsPixel(-1);
         py3 = gPad->YtoAbsPixel(-1);
         d3  = TMath::Abs(px3 - px) + TMath::Abs(py3 - py) ; //simply take sum of pixels differences
         // check if point is close to the radial axis
         if (d1 < kMaxDiff) {
            gPad->SetCursor(kMove);
            p1 = kTRUE;
         }
         // check if point is close to the left high axis
         if ( d2 < kMaxDiff) {
            gPad->SetCursor(kHand);
            p7 = kTRUE;
         }
         // check if point is close to the left down axis
         if ( d3 < kMaxDiff) {
            gPad->SetCursor(kHand);
            p8 = kTRUE;
         }
         // check if point is close to a main circle
         if (!p1 && !p7 ) {
            p6 = kTRUE;
            gPad->SetCursor(kHand);
         }
         break;

      case kButton1Down:
         // Record initial coordinates
         px4 = px;
         py4 = py;

      case kButton1Motion:
         if (p1) {
            px2 = gPad->AbsPixeltoX(px);
            py2 = gPad->AbsPixeltoY(py);
            if ( px2 < 0 && py2 < 0)  {p2 = kTRUE;};
            if ( px2 < 0 && py2 > 0 ) {p3 = kTRUE;};
            if ( px2 > 0 && py2 > 0 ) {p4 = kTRUE;};
            if ( px2 > 0 && py2 < 0 ) {p5 = kTRUE;};
            px2 = TMath::ACos(TMath::Abs(px2));
            py2 = TMath::ASin(TMath::Abs(py2));
            if (p2) {
               fAxisAngle = TMath::Pi()+(px2+py2)/2;
               p2 = kFALSE;
            };
            if (p3) {
               fAxisAngle = TMath::Pi()-(px2+py2)/2;
               p3 = kFALSE;
            };
            if (p4) {
               fAxisAngle = (px2+py2)/2;
               p4 = kFALSE;
            };
            if (p5) {
               fAxisAngle = -(px2+py2)/2;
               p5 = kFALSE;
            };
         }
         break;

      case kButton1Up:
         Paint();
   }
}


//______________________________________________________________________________
void TGraphPolargram::Paint(Option_t * /*chopt*/)
{
   // Paint TGraphPolargram

   PaintRadialDivisions();
   PaintPolarDivisions();
}


//______________________________________________________________________________
void TGraphPolargram::PaintCircle(Double_t x1, Double_t y1, Double_t r,
                            Double_t phimin, Double_t phimax, Double_t theta)
{
   // This is simplifed from TEllipse::PaintEllipse.
   // Draw this ellipse with new coordinates.

   Int_t i;
   const Int_t np = 200; // Number of point to draw circle
   static Double_t x[np+3], y[np+3];

   // Set number of points approximatively proportional to the ellipse
   // circumference.

   Double_t circ = TMath::Pi()*2*r*(phimax-phimin)/36;
   Int_t n = (Int_t)(np*circ/((gPad->GetX2()-gPad->GetX1())+
                              (gPad->GetY2()-gPad->GetY1())));
   if (n < 8) n  = 8;
   if (n > np) n = np;
   Double_t angle,dx,dy;
   Double_t dphi = (phimax-phimin)*TMath::Pi()/(180*n);
   Double_t ct   = TMath::Cos(TMath::Pi()*theta/180);
   Double_t st   = TMath::Sin(TMath::Pi()*theta/180);
   for (i=0; i<=n; i++) {
      angle = phimin*TMath::Pi()/180 + Double_t(i)*dphi;
      dx    = r*TMath::Cos(angle);
      dy    = r*TMath::Sin(angle);
      x[i]  =  x1 + dx*ct - dy*st;
      y[i]  =  y1 + dx*st + dy*ct;
   }
   gPad->PaintPolyLine(n+1,x,y);
}


//______________________________________________________________________________
void TGraphPolargram::PaintPolarDivisions()
{
   // Draw Polar divisions.
   // Check for editable pad or create default.

   Int_t i, j, rnum, rden, x=0, first, last;
   if (!gPad) return ;
   if (!gPad->IsEditable()) gROOT->MakeDefCanvas();

   gPad->RangeAxis(-1,-1,1,1);
   gPad->Range(-1.25,-1.25,1.25,1.25);
   Int_t ndivMajor = fNdivPol%100;
   Int_t ndivMinor = fNdivPol/100;

   if (!gPad->GetLogy()) {
      for (i=0; i<ndivMajor; i++) {
         Double_t txtval    = fRwtmin + i*(fRwtmax-fRwtmin)/ndivMajor;
         Double_t theta     = i*2*TMath::Pi()/ndivMajor;
         Double_t costheta  = TMath::Cos(theta);
         Double_t sintheta  = TMath::Sin(theta);
         Double_t tantheta  = TMath::Tan(theta);
         Double_t costhetas = (1+fPolarOffset)*costheta;
         Double_t sinthetas = (1+fPolarOffset)*sintheta;
         Double_t angle     = 0;
         Double_t offset    = 0;

         TLatex *textangular = new TLatex();
         textangular->SetTextColor(GetPolarColorLabel());
         textangular->SetTextFont(GetPolarLabelFont());

         char* form = " ";
         TGaxis axis;

         if (TestBit(TGraphPolargram::kLabelOrtho)) {
            // Polar numbers are aligned with their axis.
            if (theta <= TMath::PiOver2()) angle = theta*180/TMath::Pi();
            if (theta > TMath::PiOver2() && theta < 3*TMath::PiOver2())
               angle = -180+theta*180/TMath::Pi();
            if (theta >= 3*TMath::PiOver2() && theta < 2*TMath::Pi())
               angle = -360+theta*180/TMath::Pi();
            if (fRadian) {
               // Radian case.
               ReduceFraction(2*i, ndivMajor, rnum, rden); // Reduces the fraction.
               if (theta <= TMath::PiOver2()) x=12;
               if (theta > TMath::PiOver2() && theta < 3*TMath::PiOver2()) x = 32;
               if (theta >= 3*TMath::PiOver2() && theta < 2*TMath::Pi())   x = 2;
               if (rnum == 0)                       form = Form("%d",rnum);
               if (rnum == 1 && rden == 1)          form = Form("#pi");
               if (rnum == 1 && rden != 1)          form = Form("#frac{#pi}{%d}",rden);
               if (rnum != 1 && rden == 1 && i !=0) form= Form("%d#pi",rnum);
               if (rnum != 1 && rden != 1)          form = Form("#frac{%d#pi}{%d}",rnum,rden);
               textangular->SetTextAlign(x);
               textangular->PaintLatex(costhetas,
                                       sinthetas, angle, GetPolarLabelSize(), form);
            } else {
               // Any other cases: numbers are aligned with their axis.
               if ((theta <= TMath::PiOver2()) || (theta >= 3*TMath::PiOver2()
                                            && theta < 2*TMath::Pi())) x = 12;
               if (theta > TMath::PiOver2() && theta < 3*TMath::PiOver2()) x = 32;
               offset = 1.005;
               form = Form("%5.3g",txtval);
               axis.LabelsLimits(form,first,last);
               TString s = Form("%s",form);
               if (first != 0) s.Remove(0, first);
               textangular->SetTextAlign(x);
               textangular->PaintLatex(offset*costhetas,
                                     offset*sinthetas, angle, GetPolarLabelSize(), s);
            }
         } else {
       // Polar numbers are shown horizontaly.
            if (fRadian) {
          // Radian case
               ReduceFraction(2*i, ndivMajor, rnum, rden);
               if (theta == 0) x=02;
               if (theta > 0 && theta < TMath::PiOver4()) {
                  x         = 3;
                  costhetas = costhetas+0.005;
                  sinthetas = sinthetas+0.08;
               }
               if (theta >= TMath::PiOver4() && theta <= 3*TMath::PiOver4()) x=20;
               if (theta > 3*TMath::PiOver4() && theta < 5*TMath::PiOver4()) {
                  x         = 32;
                  costhetas = costhetas-0.005;
               }
               if (theta >= 5*TMath::Pi()/ 4 && theta <3*TMath::PiOver2()) {
                  x         = 32;
                  costhetas = costhetas+0.03;
                  sinthetas = sinthetas-0.07;
               }
               if (theta == 3*TMath::PiOver2() ) x=23;
               if (theta > 3*TMath::PiOver2() && theta <= 7*TMath::PiOver4()) {
                  x         = 23;
                  costhetas = costhetas+0.01;
                  sinthetas = sinthetas-0.01;
               }
               if (theta > 7*TMath::PiOver4()  && theta < 2*TMath::Pi()) {
                  x         = 2;
                  costhetas = costhetas+0.005;
                  sinthetas = sinthetas-0.01;
               }
               if (rnum == 0) form = Form("%d",rnum);
               if (rnum == 1 && rden == 1)          form = Form("#pi");
               if (rnum == 1 && rden != 1)          form = Form("#frac{#pi}{%d}",rden);
               if (rnum != 1 && rden == 1 && i !=0) form = Form("%d#pi",rnum);
               if (rnum != 1 && rden != 1)          form = Form("#frac{%d#pi}{%d}",rnum,rden);
               textangular->SetTextAlign(x);
               textangular->PaintLatex(costhetas,sinthetas,angle,GetPolarLabelSize(),form);
            } else {
          // Any other cases where numbers are shown horizontaly.
               x = 2;
               if (theta < TMath::PiOver4()) offset=1;
               if (theta >= TMath::PiOver4() && theta<TMath::PiOver2()) offset=1.02;
               if (theta == TMath::PiOver2()) {
                  x = 22;
                  offset = 1.03;
               }
               if (theta > TMath::PiOver2() && theta<3*TMath::PiOver4()) {
                  x = 32;
                  offset = 1.02;
               }
               if (theta >= 3*TMath::PiOver4() && theta < 3*TMath::PiOver2()) {
                  x = 32;
                  offset = 1;
               }
               if (theta == 3*TMath::PiOver2()) {
                  x = 23;
                  offset = 1;
               }
               if (theta > 3*TMath::PiOver2() && theta < 2*TMath::Pi()) offset = 1;
               form = Form("%5.3g",txtval);
               axis.LabelsLimits(form,first,last);
               TString s = Form("%s",form);
               if (first != 0) s.Remove(0, first);
               textangular->SetTextAlign(x);
               textangular->PaintLatex(offset*costhetas,
                                       offset*sinthetas,angle,GetPolarLabelSize(),s);
            }
         }
         TAttLine::Modify();
       //Check if SetTickpolar is actived, and draw Tickmarcks
         Bool_t issettickpolar = gPad->GetTicky();

         if (issettickpolar) {
            if (theta != 0 && theta !=TMath::Pi()) {
               gPad->PaintLine((sintheta-GetTickpolarSize())/tantheta,sintheta-GetTickpolarSize(),
               (sintheta+GetTickpolarSize())/tantheta,sintheta+GetTickpolarSize());
            }
            if (theta == 0 || theta ==TMath::Pi()) {
               gPad->PaintLine(1-GetTickpolarSize(),0,1+GetTickpolarSize(),0);
               gPad->PaintLine(-1+GetTickpolarSize(),0,-1-GetTickpolarSize(),0);
            }
         }
         gPad->PaintLine(0.,0.,costheta,sintheta);
         delete textangular;
       // Add minor lines w/o text.
         Int_t oldLineStyle = GetLineStyle();
         TAttLine::SetLineStyle(2);  //Minor lines always in this style.
         TAttLine::Modify();  //Changes line attributes apart from style.
         for (j=1; j<ndivMinor; j++) {
            Double_t thetamin = theta+j*2*TMath::Pi()/(ndivMajor*ndivMinor);
            gPad->PaintLine(0.,0.,TMath::Cos(thetamin),TMath::Sin(thetamin));
         }
         SetLineStyle(oldLineStyle);
      }
   } else {
         Int_t big = (Int_t)fRwtmax;
         Int_t test= 1;
         while (big >= 10) {
            big = big/10;
            test++;
         }
         for (i=1; i<=test; i++) {
         Double_t txtval    = pow((double)10,(double)(i-1));
         Double_t theta     = (i-1)*2*TMath::Pi()/(double)(test);
         Double_t costheta  = TMath::Cos(theta);
         Double_t sintheta  = TMath::Sin(theta);
         Double_t tantheta  = TMath::Tan(theta);
         Double_t costhetas = (1+fPolarOffset)*costheta;
         Double_t sinthetas = (1+fPolarOffset)*sintheta;
         Double_t angle     = 0;
         Double_t offset    = 0;

         TLatex *textangular = new TLatex();
         textangular->SetTextColor(GetPolarColorLabel());
         textangular->SetTextFont(GetPolarLabelFont());

         char* form = " ";
         TGaxis axis;

         if (TestBit(TGraphPolargram::kLabelOrtho)) {
            // Polar numbers are aligned with their axis.
               if ((theta <= TMath::PiOver2()) || (theta >= 3*TMath::PiOver2()
                                            && theta < 2*TMath::Pi())) x = 12;
               if (theta > TMath::PiOver2() && theta < 3*TMath::PiOver2()) x = 32;
               offset = 1.005;
               form = Form("%5.3g",txtval);
               axis.LabelsLimits(form,first,last);
               TString s = Form("%s",form);
               if (first != 0) s.Remove(0, first);
               textangular->SetTextAlign(x);
               textangular->PaintLatex(offset*costhetas,
                                     offset*sinthetas, angle, GetPolarLabelSize(), s);

         } else {
            // Polar numbers are shown horizontaly.
               x = 2;
               if (theta < TMath::PiOver4()) offset=1;
               if (theta >= TMath::PiOver4() && theta<TMath::PiOver2()) offset=1.02;
               if (theta == TMath::PiOver2()) {
                  x = 22;
                  offset = 1.03;
               }
               if (theta > TMath::PiOver2() && theta<3*TMath::PiOver4()) {
                  x = 32;
                  offset = 1.02;
               }
               if (theta >= 3*TMath::PiOver4() && theta < 3*TMath::PiOver2()) {
                  x = 32;
                  offset = 1;
               }
               if (theta == 3*TMath::PiOver2()) {
                  x = 23;
                  offset = 1;
               }
               if (theta > 3*TMath::PiOver2() && theta < 2*TMath::Pi()) offset = 1;
               form = Form("%5.3g",txtval);
               axis.LabelsLimits(form,first,last);
               TString s = Form("%s",form);
               if (first != 0) s.Remove(0, first);
               textangular->SetTextAlign(x);
               textangular->PaintLatex(offset*costhetas,
                                       offset*sinthetas,angle,GetPolarLabelSize(),s);
            }

         TAttLine::Modify();
         //Check if SetTickpolar is actived, and draw Tickmarcks
         Bool_t issettickpolar = gPad->GetTicky();
         if (issettickpolar) {
            if (theta != 0 && theta !=TMath::Pi()) {
               gPad->PaintLine((sintheta-GetTickpolarSize())/tantheta,sintheta-GetTickpolarSize(),
               (sintheta+GetTickpolarSize())/tantheta,sintheta+GetTickpolarSize());
            }
            if (theta == 0 || theta ==TMath::Pi()) {
               gPad->PaintLine(1-GetTickpolarSize(),0,1+GetTickpolarSize(),0);
               gPad->PaintLine(-1+GetTickpolarSize(),0,-1-GetTickpolarSize(),0);
            }
         }
         gPad->PaintLine(0.,0.,costheta,sintheta);
         delete textangular;
         // Add minor lines w/o text.
         Int_t oldLineStyle = GetLineStyle();
         TAttLine::SetLineStyle(2);  //Minor lines always in this style.
         TAttLine::Modify();  //Changes line attributes apart from style.
         Double_t a=0;
         Double_t b,c,d;
         b = TMath::Log(10)*test;
         d= 2*TMath::Pi()/(double)test;
         for (j=1; j<9; j++) {
            a=TMath::Log(j+1)-TMath::Log(j)+a;
            c=a/b*6.28+d*(i-1);
            gPad->PaintLine(0.,0.,TMath::Cos(c),TMath::Sin(c));
         }
         SetLineStyle(oldLineStyle);
      }
   }
}


//______________________________________________________________________________
void TGraphPolargram::PaintRadialDivisions()
{
   // Paint radial divisions.
   // Check for editable pad or create default.

   static char chopt[8] = "";
   Int_t i,j;
   Int_t ndiv      = TMath::Abs(fNdivRad);
   Int_t ndivMajor = ndiv%100;
   Int_t ndivMinor = ndiv/100;
   Int_t ndivmajor;
   Double_t frwrmin, frwrmax, binWidth = 0;

   THLimitsFinder::Optimize(fRwrmin,fRwrmax,ndivMajor,frwrmin,
                               frwrmax, ndivmajor,binWidth,"");

   if (!gPad) return ;
   if (!gPad->IsEditable()) gROOT->MakeDefCanvas();
   if (!gPad->GetLogx()) {
      gPad->RangeAxis(-1,-1,1,1);
      gPad->Range(-1.25,-1.25,1.25,1.25);
      Double_t umin  = fRwrmin;
      Double_t umax  = fRwrmax;
      Double_t rmajmin  = (frwrmin-fRwrmin)/(fRwrmax-fRwrmin);
      Double_t rmajmax  = (frwrmax-fRwrmin)/(fRwrmax-fRwrmin);
      Double_t dist  = (rmajmax-rmajmin)/ndivmajor;
      Int_t ndivminor;

      chopt[0] = 0;
      strcat(chopt, "SDH");
      if (fNdivRad < 0) strcat(chopt, "N");
      // Paint axis.
      TGaxis axis;
      axis.SetLabelSize(GetRadialLabelSize());
      axis.SetLabelColor(GetRadialColorLabel());
      axis.SetLabelFont(GetRadialLabelFont());
      axis.SetLabelOffset(GetRadialOffset());
      axis.PaintAxis(0, 0, TMath::Cos(GetAngle()), TMath::Sin(GetAngle()),
                                     umin, umax,  ndiv, chopt, 0., kFALSE);

      // Paint Circles.
      // First paint main circle.
      PaintCircle(0.,0.,1,0.,360,0);

      // Optimised case.
      if (fNdivRad>0 ) {
         Double_t frwrmini, frwrmaxi, binWidth2 =0;
         THLimitsFinder::Optimize(frwrmin,frwrmin+binWidth,ndivMinor,frwrmini,
                                  frwrmaxi, ndivminor,binWidth2,"");
         Double_t dist2 = dist/(ndivminor);
         // Paint major circles.
         for (i=1; i<=ndivmajor+2; i++) {
            TAttLine::SetLineStyle(1);
            TAttLine::Modify();
            PaintCircle(0.,0.,rmajmin,0.,360,0);

            //Paint minor circles.
            TAttLine::SetLineStyle(2);
            TAttLine::Modify();
            for (j=1; j<ndivminor+1; j++) {
               if (rmajmin+j*dist2<=1) PaintCircle(0.,0.,rmajmin+j*dist2,0.,360,0);
            }
            rmajmin = (frwrmin-fRwrmin)/(fRwrmax-fRwrmin)+(i-1)*dist;
         }
      // Non-optimized case.
      } else {

         // Paint major circles.
         for (i=1; i<=ndivMajor; i++) {
            TAttLine::SetLineStyle(1);
            TAttLine::Modify();
            Double_t rmaj = i*1./ndivMajor;
            PaintCircle(0.,0.,rmaj,0.,360,0);

            // Paint minor circles.
            for (j=1; j<ndivMinor; j++) {
               TAttLine::SetLineStyle(2);
               TAttLine::Modify();
               PaintCircle(0.,0.,rmaj- j*1./(ndivMajor*ndivMinor),0.,360,0);
            }
         }
      }
   } else {
   // Draw Log scale on radial axis if option activated.
      Int_t big = (Int_t)fRwrmax;
      Int_t test= 0;
      while (big >= 10) {
         big = big/10;
         test++;
      }
      for (i=1; i<=test; i++) {
         TAttLine::SetLineStyle(1);
         TAttLine::Modify();
         Double_t ecart;
         ecart = ((double) i)/ ((double) test);
         PaintCircle(0.,0.,ecart,0,360,0);
         TAttLine::SetLineStyle(GetLineStyle());
         TAttLine::Modify();
         Double_t a=0;
         Double_t b,c,d;
         b = TMath::Log(10)*test;
         d = 1/(double)test;
         for (j=1; j<9; j++) {
            a = TMath::Log(j+1)-TMath::Log(j)+a;
            c = a/b+d*(i-1);
            PaintCircle(0,0.,c,0.,360,0);
         }
      }
   }
   TAttLine::SetLineStyle(1);
   TAttLine::Modify();
}


//______________________________________________________________________________
void TGraphPolargram::ReduceFraction(Int_t num, Int_t den, Int_t &rnum, Int_t &rden)
{
   // Reduce fractions.

   Int_t a = 0;
   Int_t b = 0;
   Int_t i = 0;
   Int_t j = 0;
   a = den;
   b = num;
   if (b > a) {
      j = b;
   } else {
      j = a;
   }
   for (i=j; i > 1; i--) {
      if ((a % i == 0) && (b % i == 0)) {
         a = a/i;
         b = b/i;
      }
   }
   rden = a;
   rnum = b;
}


//______________________________________________________________________________
void TGraphPolargram::SetAxisAngle(Double_t angle)
{
   // Set axis angle.

   fAxisAngle = angle/180*TMath::Pi();
}


//______________________________________________________________________________
void TGraphPolargram::SetNdivPolar(Int_t ndiv)
{
   // Set the number of Polar divisions: enter a number ij with 0<i<99
   //                                                           0<j<99
   // i sets the major Polar divisions.
   // j sets the minor Polar divisions.

   if (ndiv > 0) fNdivPol = ndiv;
   if (gPad) gPad->Modified();
}


//_____________________________________________________________________________
void TGraphPolargram::SetNdivRadial(Int_t ndiv)
{
  // Set the number of radial divisions: enter a number ij with 0<i<99
  //                                                            0<j<99
  // i sets the major radial divisions.
  // j sets the minor radial divisions.

   fNdivRad = ndiv;
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TGraphPolargram::SetPolarLabelColor(Color_t tcolorangular )
{
   // Set Polar labels color.

   fPolarLabelColor = tcolorangular;
}


//______________________________________________________________________________
void TGraphPolargram::SetPolarLabelFont(Font_t tfontangular)
{
   // Set Polar label font.

   fPolarLabelFont = tfontangular;
}


//______________________________________________________________________________
void TGraphPolargram::SetPolarLabelSize(Double_t angularsize )
{
   // Set angular labels size.

   fPolarTextSize = angularsize;
}


//______________________________________________________________________________
void TGraphPolargram::SetPolarOffset(Double_t angularOffset)
{
   // Set the labels offset.

   fPolarOffset = angularOffset;
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TGraphPolargram::SetRadialLabelColor(Color_t tcolorradial )
{
   // Set radial labels color.

   fRadialLabelColor = tcolorradial;
}


//______________________________________________________________________________
void TGraphPolargram::SetRadialLabelFont(Font_t tfontradial)
{
   // Set radial label font.

   fRadialLabelFont = tfontradial;
}


//______________________________________________________________________________
void TGraphPolargram::SetRadialLabelSize(Double_t radialsize )
{
   // Set radial labels size.

   fRadialTextSize = radialsize;
}


//______________________________________________________________________________
void TGraphPolargram::SetRadialOffset(Double_t radialOffset)
{
   // Set the labels offset.

   fRadialOffset = radialOffset;
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TGraphPolargram::SetRangePolar(Double_t tmin, Double_t tmax)
{
   // Allows to change range Polar.
   // tmin is the start number.
   // tmax is the end number.

   fDegree = kFALSE;
   fGrad   = kFALSE;
   fRadian = kFALSE;

   if (tmin < tmax) {
      fRwtmin = tmin;
      fRwtmax = tmax;
   }
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TGraphPolargram::SetRangeRadial(Double_t rmin, Double_t rmax)
{
   // Set the radial range.
   // rmin is at center of the circle.
   // rmax is at the intersection of the right X axis part and the circle.

   if (rmin < rmax) {
      fRwrmin = rmin;
      fRwrmax = rmax;
   }
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TGraphPolargram::SetTickpolarSize(Double_t tickpolarsize)
{
   // Set polar ticks size.

   fTickpolarSize = tickpolarsize;
}


//______________________________________________________________________________
void TGraphPolargram::SetToDegree()
{
   // The Polar circle is labelled using degree.

   fDegree = kTRUE;
   fGrad   = kFALSE;
   fRadian = kFALSE;
   ChangeRangePolar(0,360);
}


//______________________________________________________________________________
void TGraphPolargram::SetToGrad()
{
   // The Polar circle is labelled using gradian.

   fGrad   = kTRUE;
   fRadian = kFALSE;
   fDegree = kFALSE;
   ChangeRangePolar(0,200);
}


//______________________________________________________________________________
void TGraphPolargram::SetToRadian()
{
   // The Polar circle is labelled using radian.

   fRadian = kTRUE;
   fGrad   = kFALSE;
   fDegree = kFALSE;
   ChangeRangePolar(0,2*TMath::Pi());
}


//______________________________________________________________________________
void TGraphPolargram::SetTwoPi()
{
   //set range from 0 to 2*pi
   SetRangePolar(0,2*TMath::Pi());
}
