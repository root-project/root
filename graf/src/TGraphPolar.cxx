// @(#)root/graf:$Name:  $:$Id: TGraphPolar.cxx,v 1.1 2006/05/18 16:12:09 couet Exp $
// Author: Sebastian Boser, 02/02/06

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//______________________________________________________________________________
//
//  TPolarGraph creates a polar graph (including error bars). 
//  It uses the class TGraphPolargram to draw the polar axis.
//

#include "TGraphPolar.h"

ClassImp(TGraphPolar);
  

//______________________________________________________________________________
TGraphPolar::TGraphPolar(Int_t n, 
   const Double_t* x, const Double_t* y, const Double_t *ex, const Double_t* ey)
  : TGraphErrors(n,x,y,ex,ey)
{
   // TGraphPolar constructor.
   
   fXpol       = 0;
   fYpol       = 0;
   fPolargram  = 0;
   fOptionAxis = kFALSE;
   SetEditable(kFALSE);
};
 

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
   // Return the disatnce in pixel between the GraphPloar and (px,py).

   // Swap polar and orthognal coordinates.
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
void TGraphPolar::ExecuteEvent(Int_t /*event*/, Int_t /*px*/, Int_t /*py*/)
{
   //Indicate that there is something to click here

   gPad->SetCursor(kHand);
}


//______________________________________________________________________________
void TGraphPolar::SetMaxRadial(Double_t maximum)
{
   // Set maximum radial.
   
   if (fPolargram) fPolargram->SetRangeRadial(fPolargram->GetRMin(),maximum);
}


//______________________________________________________________________________
void TGraphPolar::SetMinRadial(Double_t minimum)
{
   // Set minimum radial.
   
   if (fPolargram) fPolargram->SetRangeRadial(minimum, fPolargram->GetRMax());
}


//______________________________________________________________________________
void TGraphPolar::SetMaxPolar(Double_t maximum)
{
   // Set maximum polar.
   
   if (fPolargram) fPolargram->SetRangePolar(fPolargram->GetTMin(),maximum);
}


//______________________________________________________________________________
void TGraphPolar::SetMinPolar(Double_t minimum)
{
   // Set minimum polar.

   if (fPolargram) fPolargram->SetRangePolar(minimum, fPolargram->GetTMax());
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
void TGraphPolar::Paint(Option_t* options)
{
   // Paint TGraphPolar.

   Int_t ipt, i;

   if (fNpoints<1) return;
   
   TString opt = options;
   opt.ToUpper();

   Double_t rwrmin, rwrmax, rwtmin, rwtmax;
   
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
   
   // Get new polargram range if necessary
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
    
      // Add radial and polar margins
      if (rwrmin == rwrmax) rwrmax += 1.;
      if (rwtmin == rwtmax) rwtmax += 1.;
      Double_t dr = (rwrmax-rwrmin);
      Double_t dt = (rwtmax-rwtmin);
      rwrmax += 0.1*dr;
      rwrmin -= 0.1*dr;

      // Assume equaly spaced points for full 2*Pi
      rwtmax += dt/fNpoints;
   } else { 
      rwrmin = fPolargram->GetRMin();
      rwrmax = fPolargram->GetRMax();
      rwtmin = fPolargram->GetTMin();
      rwtmax = fPolargram->GetTMax();
   }

   if ((!fPolargram)||fOptionAxis) {
      // Draw polar coord system
      fPolargram = new TGraphPolargram("Polargram",rwrmin,rwrmax,rwtmin,rwtmax);
      fPolargram->Draw(opt);
      fOptionAxis = kFALSE;   //Prevent redrawing
   }
   
   // Convert points to polar
   if (fXpol) delete fXpol;
   if (fYpol) delete fYpol;
   fXpol = new Double_t[fNpoints];
   fYpol = new Double_t[fNpoints];

   // Project theta in [0,2*Pi] and radius in [0,1]
   Double_t radiusNDC = rwrmax-rwrmin; 
   Double_t thetaNDC  = (rwtmax-rwtmin)/(2*kPi);
   
   // First draw the error bars
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
            Double_t phimin = (fX[i]-fEX[i]-rwtmin)/thetaNDC*180/kPi;
            Double_t phimax = (fX[i]+fEX[i]-rwtmin)/thetaNDC*180/kPi;

            // Use TGraphPolargram::PaintCircle
            TAttLine::Modify();
            fPolargram->PaintCircle(0,0,rad,phimin,phimax,0);
         }
      }
   }

   // Then draw the graph itself
   for (i=0; i<fNpoints; i++) {
      fXpol[i] = (fY[i]-rwrmin)/radiusNDC*TMath::Cos((fX[i]-rwtmin)/thetaNDC);
      fYpol[i] = (fY[i]-rwrmin)/radiusNDC*TMath::Sin((fX[i]-rwtmin)/thetaNDC);
   }
   TGraph::PaintGraph(fNpoints, fXpol, fYpol,opt);
}


ClassImp(TGraphPolargram);


//______________________________________________________________________________
TGraphPolargram::TGraphPolargram(const char* name, Double_t rmin, Double_t rmax,
                                 Double_t tmin, Double_t tmax):
                                 TNamed(name,"Polargram")
{
   // TGraphPolargram Constructor.

   fRwrmin      = rmin;
   fRwrmax      = rmax;
   fRwtmin      = tmin;
   fRwtmax      = tmax;
   fNdivRad     = 502;
   fNdivPol     = 304;
   fLabelOffset = 0.03;
}


//______________________________________________________________________________
TGraphPolargram::~TGraphPolargram()
{
   // TGraphPolargram destructor.
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
   
   // Check if close to a (major) polar line.
   // This is not a proper calculation, but rather fast.
   Int_t dt = kMaxPixel;
   for (i=0; i<(fNdivPol%100); i++) {
      Double_t theta = i*2*kPi/(fNdivPol%100);
      
      // Attention: px,py in pixel units, line given in user coordinates
      Int_t dthis = DistancetoLine(px,py,0.,0.,TMath::Cos(theta),
                                               TMath::Sin(theta));

      // Fails if we are outside box discribed by the line
      // (i.e for all hor/vert lines)
      if (dthis==9999) {

         // Outside -> Get distance to endpoint of line
         if (rad>1) {
            dthis = (Int_t)TMath::Sqrt(
                    TMath::Power(px-gPad->XtoPixel(TMath::Cos(theta)),2)+
                    TMath::Power(py-gPad->YtoPixel(TMath::Sin(theta)),2));
         } else {

            // Check for horizontal line 
            if (((TMath::Abs(theta-kPi)<0.1)&&((px-gPad->XtoPixel(0))<0))
              ||((TMath::Abs(theta)<0.1)&&((px-gPad->XtoPixel(0))>0))) {
               dthis = abs(py-gPad->YtoPixel(0.));
            }

            //Check for vertical line
            if (((TMath::Abs(theta-kPi/2)<0.1)&&((py-gPad->YtoPixel(0))>0))
              ||((TMath::Abs(theta-3*kPi/2)<0.1)&&(py-gPad->YtoPixel(0))<0)) {
               dthis = abs(px-gPad->XtoPixel(0.));
            }
            if (dthis==9999) {

               // Inside, but out of box for nonorthogonal line ->
               // get distance to start point
               dthis = (Int_t)TMath::Sqrt(
                       TMath::Power(px-gPad->XtoPixel(0.),2)+
                       TMath::Power(py-gPad->YtoPixel(0.),2));
            }
         }
      }

      // Take distance to closes line
      dt = TMath::Min(dthis,dt);
   }
   return TMath::Min(drad, dt);
}


//______________________________________________________________________________
void TGraphPolargram::ExecuteEvent(Int_t /*event*/, Int_t /*px*/, Int_t /*py*/)
{
   // Indicate that there is something to click here.

   gPad->SetCursor(kHand);
}


//______________________________________________________________________________
void TGraphPolargram::SetNdivRadial(Int_t ndiv)
{
   // Set the number of radial divisions
   
   if (ndiv > 0) fNdivRad = ndiv;
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TGraphPolargram::SetNdivPolar(Int_t ndiv)
{
   // Set the number of polar divisions

   if (ndiv > 0) fNdivPol = ndiv;
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TGraphPolargram::SetLabelOffset(Double_t labelOffset)
{
   // Set the labels offset.

   fLabelOffset = labelOffset;
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TGraphPolargram::SetRangeRadial(Double_t rmin, Double_t rmax)
{
   // Set the radial range.

   if (rmin < rmax) {
      fRwrmin = rmin;
      fRwrmax = rmax;
   }
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TGraphPolargram::SetRangePolar(Double_t tmin, Double_t tmax)
{
   // Set the polar range.

   if (tmin < tmax) {
      fRwtmin = tmin;
      fRwtmax = tmax;
   }
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TGraphPolargram::Draw(Option_t* options)
{
   // Draw Polargram.

   TGraphPolargram::Paint(options);
   AppendPad(options);
}


//______________________________________________________________________________
void TGraphPolargram::PaintCircle(Double_t x1, Double_t y1, Double_t r,
                            Double_t phimin, Double_t phimax, Double_t theta)
{
   // This is simplified from TEllipse::PaintEllipse
   // Draw this ellipse with new coordinates.

   Int_t i;
   const Int_t np = 200;
   static Double_t x[np+3], y[np+3];

   // Set number of points approximatively proportional to the ellipse
   // circumference
   Double_t circ = kPi*2*r*(phimax-phimin)/360;
   Int_t n = (Int_t)(np*circ/((gPad->GetX2()-gPad->GetX1())+
                              (gPad->GetY2()-gPad->GetY1())));
   if (n < 8) n= 8;
   if (n > np) n = np;
   Double_t angle,dx,dy;
   Double_t dphi = (phimax-phimin)*kPi/(180*n);
   Double_t ct   = TMath::Cos(kPi*theta/180);
   Double_t st   = TMath::Sin(kPi*theta/180);
   for (i=0;i<=n;i++) {
      angle = phimin*kPi/180 + Double_t(i)*dphi;
      dx    = r*TMath::Cos(angle);
      dy    = r*TMath::Sin(angle);
      x[i]  = gPad->XtoPad(x1 + dx*ct - dy*st);
      y[i]  = gPad->YtoPad(y1 + dx*st + dy*ct);
   }
   gPad->PaintPolyLine(n+1,x,y);
}


//______________________________________________________________________________
void TGraphPolargram::Paint(Option_t* /* options */)
{
   // Check for editable pad or create default.

   Int_t i,j;
   if (!gPad) return ;
   if (!gPad->IsEditable()) gROOT->GetMakeDefCanvas();
    
   gPad->RangeAxis(-1,-1,1,1);
   gPad->Range(-1.25,-1.25,1.25,1.25);

   // Draw radial divisions.
   Int_t ndivMajor = fNdivRad%100;
   Int_t ndivMinor = fNdivRad/100;
   for (i=1; i<=ndivMajor; i++) {
      TAttLine::Modify();  //Change line attributes apart from style
      Double_t rmaj = i*1./ndivMajor;
      PaintCircle(0.,0.,rmaj,0.,360,0);
      Double_t txtval = fRwrmin+i*(fRwrmax-fRwrmin)/ndivMajor;
      SetTextAlign(23);
      TAttText::Modify();
      gPad->PaintText(rmaj,-fLabelOffset,Form("%5.3g",txtval));
      Int_t oldLineStyle = GetLineStyle();
      TAttLine::SetLineStyle(2);  //Minor lines allways in this style
      TAttLine::Modify();  //Change line attributes apart from style
      for (j=1; j<ndivMinor; j++) {
         PaintCircle(0.,0.,rmaj- j*1./(ndivMajor*ndivMinor),0.,360,0);
      }
      SetLineStyle(oldLineStyle);
   }

   // Draw polar divisions.
   ndivMajor = fNdivPol%100;
   ndivMinor = fNdivPol/100;
   for (i=0; i<ndivMajor; i++) {
      Double_t txtval = fRwtmin + i*(fRwtmax-fRwtmin)/ndivMajor;
      Double_t theta = i*2*kPi/ndivMajor;
      TAttLine::Modify();  
      gPad->PaintLine(0.,0.,TMath::Cos(theta),TMath::Sin(theta));

      // Adjust text right/left in left/right half of circle.
      if ((theta < kPi/2)||(theta > 3*kPi/2)) SetTextAlign(12);
      else SetTextAlign(32);

      // Adjust text centerted top/bottom at bottom/top.
      if (TMath::Abs(theta - kPi/2)<0.2) SetTextAlign(21);
      if (TMath::Abs(theta - 3*kPi/2)<0.2) SetTextAlign(23);
      TAttText::Modify(); //Get current text settings if changed
      gPad->PaintText((1+fLabelOffset)*TMath::Cos(theta),
                      (1+fLabelOffset)*TMath::Sin(theta),
      Form("%5.3g",txtval));

      // Add minor lines w/o text.
      Int_t oldLineStyle = GetLineStyle();
      TAttLine::SetLineStyle(2);  //Minor lines allways in this style
      TAttLine::Modify();  //Change line attributes apart from style
      for (j=1; j<ndivMinor; j++) {
         Double_t thetamin = theta+j*2*kPi/(ndivMajor*ndivMinor);
         gPad->PaintLine(0.,0.,TMath::Cos(thetamin),TMath::Sin(thetamin));
      }
      SetLineStyle(oldLineStyle);
   }
}
