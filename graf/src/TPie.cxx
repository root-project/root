// @(#)root/graf:$Name:  $:$Id: TPie.cxx,v 1.3 2006/11/22 15:42:22 couet Exp $
// Author: Guido Volpi, Olivier Couet 03/11/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TPie.h"

#include <Riostream.h>
#include <TROOT.h>
#include <TPad.h>
#include <TArc.h>
#include <TMath.h>
#include <TStyle.h>
#include <TLatex.h>
#include <TPaveText.h>
#include <TH1.h>

ClassImp(TPie)

//______________________________________________________________________________
//
// Draw a Pie Chart
//
// The macro $ROOTSYS/tutorials/piechart.C produces the following picture:
//Begin_Html
/*
<img src="gif/piechart.gif">
*/
//End_Html
//

static Double_t gX             = 0; // temporary pie X position
static Double_t gY             = 0; // temporary pie Y position
static Double_t gRadius        = 0; // temporary pie Radius of the TPie
static Double_t gRadiusOffset  = 0; // temporary slice's radial offset
static Double_t gAngularOffset = 0; // temporary slice's angular offset
static Bool_t   gIsUpt         = kFALSE; // True is the TPie should be updated


//______________________________________________________________________________
TPie::TPie() : TNamed()
{
   // Default constructor.

   Init(1, 0, 0.5, 0.5, 0.4);
}


//______________________________________________________________________________
TPie::TPie(const char *name, const char *title, Int_t npoints) :
   TNamed(name,title)
{
   // This constructor creates a pie chart when only the number of
   // the slices is known. The number of slices is fixed.

   Init(npoints, 0, 0.5, 0.5, 0.4);
}


//______________________________________________________________________________
TPie::TPie(const char *name,
               const char *title,
               Int_t npoints, Double_t *vals,
               Int_t *colors, const char *lbls[]) : TNamed(name,title)
{
   // Normal constructor. The 1st and 2nd parameters are the name of the object
   // and its title.
   //
   // The number of points passed at this point is used to allocate the memory.
   //
   // Slices values are given as Double_t.
   //
   // The 4th elements is an array containing, in double precision format,
   // the value of each slice. It is also possible to specify the filled color
   // of each slice. If the color array is not specfied the slices are colored
   // using a color sequence in the standard palette.

   Init(npoints, 0, 0.5, 0.5, 0.4);
   for (Int_t i=0; i<fNvals; ++i) fVals[i] = vals[i];

   SetFillColors(colors);
   SetLabels(lbls);
}


//______________________________________________________________________________
TPie::TPie(const char *name,
               const char *title,
               Int_t npoints, Float_t *vals,
               Int_t *colors, const char *lbls[]) : TNamed(name,title)
{
   // Normal constructor (Float_t).

   Init(npoints, 0, 0.5, 0.5, 0.4);
   for (Int_t i=0; i<fNvals; ++i) fVals[i] = vals[i];

   SetFillColors(colors);
   SetLabels(lbls);
}


//______________________________________________________________________________
TPie::TPie(const TH1 *h) : TNamed(h->GetName(),h->GetTitle())
{
   // Constructor from a TH1

   Init(h->GetNbinsX(), 0, 0.5, 0.5, 0.4);
   for (Int_t i=0; i<fNvals; ++i) fVals[i] = h->GetBin(i);
}


//______________________________________________________________________________
TPie::TPie(const TPie &cpy) : TNamed(cpy), TAttText(cpy)
{
   // Copy constructor.

   Init(cpy.fNvals, cpy.fAngularOffset, cpy.fX, cpy.fY, cpy.fRadius);

   for (Int_t i=0;i<fNvals;++i) {
      fVals[i]          = cpy.fVals[i];
      fLineColors[i]    = cpy.fLineColors[i];
      fLineStyles[i]    = cpy.fLineStyles[i];
      fLineWidths[i]    = cpy.fLineWidths[i];
      fFillColors[i]    = cpy.fFillColors[i];
      fFillStyles[i]    = cpy.fFillStyles[i];
      fRadiusOffsets[i] = cpy.fRadiusOffsets[i];
   }
}

//______________________________________________________________________________
TPie::~TPie()
{
   // Destructor.

   if (fNvals>0) {
      delete [] fVals;
      delete [] fLabels;
      delete [] fLineColors;
      delete [] fLineStyles;
      delete [] fLineWidths;
      delete [] fFillColors;
      delete [] fFillStyles;
      delete [] fRadiusOffsets;
   }

   if (fSlices) delete [] fSlices;
}


//______________________________________________________________________________
Int_t TPie::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Evaluate the distance to the chart in gPad.

   if ( (fCurrent_slice = DistancetoSlice(px,py))>=0 ) {
      if (fCurrent_rad<=fRadius) {
         return 0;
      } else {
         Int_t tmpx = gPad->XtoPixel((fCurrent_rad-fRadius)*
                                     TMath::Cos(fCurrent_ang));
         Int_t tmpy = gPad->YtoPixel((fCurrent_rad-fRadius)*
                                     TMath::Sin(fCurrent_ang));
         return (Int_t) TMath::Sqrt(tmpx*tmpx+tmpy*tmpy);
      }
   } else {
      return 9999;
   }
}


//______________________________________________________________________________
Int_t TPie::DistancetoSlice(Int_t px, Int_t py)
{
   // Returns the slice number at the pixel position (px,py).
   // Returns -1 if no slice is picked.
   //
   // Used by DistancetoPrimitive.

   MakeSlices();

   Int_t result(-1);

   // coordinates
   Double_t xx = gPad->PadtoX(gPad->AbsPixeltoX(px));
   Double_t yy = gPad->PadtoY(gPad->AbsPixeltoY(py));

   Double_t phimin;
   Double_t cphi;
   Double_t phimax;
   for (Int_t i=0;i<fNvals;++i) {
      if (gIsUpt && fCurrent_slice!=i) continue;

      // Angles' values for this slice
      phimin = fSlices[2*i  ]*TMath::Pi()/180.;
      cphi   = fSlices[2*i+1]*TMath::Pi()/180.;
      phimax = fSlices[2*i+2]*TMath::Pi()/180.;

      // Recalculate slice's vertex' coordinates.
      Double_t x, y, rad, radOffset;
      if (gIsUpt) {
         x         = gX;
         y         = gY;
         rad       = gRadius,
         radOffset = gRadiusOffset;
      }
      else {
         x         = fX;
         y         = fY;
         rad       = fRadius;
         radOffset = fRadiusOffsets[i];
      }

      Double_t dx  = xx-x-radOffset*TMath::Cos(cphi);
      Double_t dy  = yy-y-radOffset*TMath::Sin(cphi);
      Double_t ang = TMath::ATan2(dy,dx);
      if (ang<0) ang += TMath::TwoPi();

      Double_t dist = TMath::Sqrt(dx*dx+dy*dy);

      if ( ((ang>=phimin && ang <= phimax) || (phimax>TMath::TwoPi() &&
             ang+TMath::TwoPi()>=phimin && ang+TMath::TwoPi()<phimax)) &&
             dist<=rad) { // if true the pointer is in the slice region
         fCurrent_x    = dx;
         fCurrent_y    = dy;
         fCurrent_ang  = ang;
         fCurrent_phi1 = phimin;
         fCurrent_phi2 = phimax;
         fCurrent_rad  = dist;
         result        = i;
         break;
      }
   }
   return result;
}


//______________________________________________________________________________
void TPie::Draw(Option_t *option)
{
   // Draw the pie chart.
   //
   // The possible options are listed in the TPie::Paint() method.

   TString soption(option);
   soption.ToLower();

   if (soption.Length()==0) soption = "l";

   if (gPad) {
      if (!gPad->IsEditable()) (gROOT->GetMakeDefCanvas())();
      if (!soption.Contains("same")) {
         gPad->Clear();
         gPad->Range(0.,0.,1.,1.);
      }
   }

   AppendPad(option);
}


//______________________________________________________________________________
void TPie::DrawGhost()
{
   // This method is for internal use. It is used by Execute event to draw the
   // outline of "this" TPie. Used when the opaque movements are not permitted.

   MakeSlices();

   // Loop over slices
   for (Int_t i=0;i<fNvals;++i) {
      Float_t minphi = (fSlices[i*2]+gAngularOffset+.5)*TMath::Pi()/180.;
      Float_t avgphi = fSlices[i*2+1]*TMath::Pi()/180.;
      Float_t maxphi = (fSlices[i*2+2]-.5)*TMath::Pi()/180;

      Double_t radOffset = (i == fCurrent_slice ? gRadiusOffset : fRadiusOffsets[i]);
      Double_t x0 = gX+radOffset*TMath::Cos(avgphi);
      Double_t y0 = gY+radOffset*TMath::Sin(avgphi);

      gVirtualX->DrawLine( gPad->XtoAbsPixel(x0), gPad->YtoAbsPixel(y0),
                           gPad->XtoAbsPixel(x0+gRadius*TMath::Cos(minphi)),
                           gPad->YtoAbsPixel(y0+gRadius*TMath::Sin(minphi)) );

      Int_t ndiv = 10;
      Double_t dphi = (maxphi-minphi)/ndiv;

      if (dphi>.15) ndiv = (Int_t) ((maxphi-minphi)/.15);
      dphi = (maxphi-minphi)/ndiv;

      // Loop to draw the arc
      for (Int_t j=1;j<ndiv;++j) {
         Double_t phi = minphi+dphi*j;
         gVirtualX->DrawLine( gPad->XtoAbsPixel(x0+gRadius*TMath::Cos(phi)),
                              gPad->YtoAbsPixel(y0+gRadius*TMath::Sin(phi)),
                              gPad->XtoAbsPixel(x0+gRadius*TMath::Cos(phi+dphi)),
                              gPad->YtoAbsPixel(y0+gRadius*TMath::Sin(phi+dphi)));
      }

      gVirtualX->DrawLine( gPad->XtoAbsPixel(x0+gRadius*TMath::Cos(maxphi)),
                           gPad->YtoAbsPixel(y0+gRadius*TMath::Sin(maxphi)),
                           gPad->XtoAbsPixel(x0), gPad->YtoAbsPixel(y0) );
   }
}


//______________________________________________________________________________
void TPie::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Execute the mouse events.

   if (!gPad) return;
   if (!gPad->IsEditable() && event != kMouseEnter) return;

   MakeSlices();

   static bool isMovingPie(kFALSE);
   static bool isMovingSlice(kFALSE);
   static bool isResizing(kFALSE);
   static bool isRotating(kFALSE);
   static bool onBorder(kFALSE);
   bool isRedrawing(kFALSE);
   static Int_t prev_event(-1);
   static Int_t oldpx, oldpy;

   // Portion of pie considered as "border"
   static Double_t dr     = gPad->PixeltoX(3);
   static Double_t minRad = gPad->PixeltoX(10);

   // Angular divisions in radial direction
   static Double_t angstep1 = 0.5*TMath::PiOver4();
   static Double_t angstep2 = 1.5*TMath::PiOver4();
   static Double_t angstep3 = 2.5*TMath::PiOver4();
   static Double_t angstep4 = 3.5*TMath::PiOver4();
   static Double_t angstep5 = 4.5*TMath::PiOver4();
   static Double_t angstep6 = 5.5*TMath::PiOver4();
   static Double_t angstep7 = 6.5*TMath::PiOver4();
   static Double_t angstep8 = 7.5*TMath::PiOver4();

   Int_t dx, dy;
   Double_t mdx, mdy;

   switch(event) {
      case kButton1Down:
         // Change cursor to show pie's movement.
         gVirtualX->SetLineColor(1);
         gVirtualX->SetLineWidth(2);

         // Current center and radius.
         gX             = fX;
         gY             = fY;
         gRadius        = fRadius;
         gRadiusOffset  = fRadiusOffsets[fCurrent_slice];
         gAngularOffset = 0;
         gIsUpt         = kTRUE;

         prev_event = kButton1Down;

      case kMouseMotion:
         if (fCurrent_rad>=fRadius-2.*dr && fCurrent_rad<=fRadius+dr
               && !isMovingPie && !isMovingSlice && !isResizing) {
            if (fCurrent_ang>=angstep8 || fCurrent_ang<angstep1)
               gPad->SetCursor(kRightSide);
            else if (fCurrent_ang>=angstep1 && fCurrent_ang<angstep2)
               gPad->SetCursor(kTopRight);
            else if (fCurrent_ang>=angstep2 && fCurrent_ang<angstep3)
               gPad->SetCursor(kTopSide);
            else if (fCurrent_ang>=angstep3 && fCurrent_ang<angstep4)
               gPad->SetCursor(kTopLeft);
            else if (fCurrent_ang>=angstep4 && fCurrent_ang<=angstep5)
               gPad->SetCursor(kLeftSide);
            else if (fCurrent_ang>=angstep5 && fCurrent_ang<angstep6)
               gPad->SetCursor(kBottomLeft);
            else if (fCurrent_ang>=angstep6 && fCurrent_ang<angstep7)
               gPad->SetCursor(kBottomSide);
            else if (fCurrent_ang>=angstep7 && fCurrent_ang<angstep8)
               gPad->SetCursor(kBottomRight);
            onBorder = kTRUE;
         } else {
            onBorder = kFALSE;
            if (fCurrent_rad>fRadius*.6) {
               gPad->SetCursor(kPointer);
            } else if (fCurrent_rad<=fRadius*.3) {
               gPad->SetCursor(kHand);
            } else if (fCurrent_rad<=fRadius*.6 && fCurrent_rad>=fRadius*.3) {
               gPad->SetCursor(kRotate);
            }
         }
         oldpx = px;
         oldpy = py;
         if (isMovingPie || isMovingSlice) gPad->SetCursor(kMove);
         break;

      case kButton1Motion:
         if (!isMovingSlice || !isMovingPie || !isResizing) {
            if (prev_event==kButton1Down) {
               if (onBorder) {
                  isResizing = kTRUE;
               } else if (fCurrent_rad>=fRadius*.6 && fCurrent_slice>=0) {
                  isMovingSlice = kTRUE;
               } else if (fCurrent_rad<=fRadius*.3) {
                  isMovingPie = kTRUE;
               } else if (fCurrent_rad<fRadius*.6 && fCurrent_rad>fRadius*.3) {
                  isRotating = kTRUE;
               }
            }
         }

         dx = px-oldpx;
         dy = py-oldpy;

         mdx = gPad->PixeltoX(dx);
         mdy = gPad->PixeltoY(dy);

         if (isMovingPie || isMovingSlice) {
            gPad->SetCursor(kMove);
            if (isMovingSlice) {
               Float_t avgphi = fSlices[fCurrent_slice*2+1]*TMath::Pi()/180.;

               if (!gPad->OpaqueMoving()) DrawGhost();

               gRadiusOffset += TMath::Cos(avgphi)*mdx +TMath::Sin(avgphi)*mdy;
               if (gRadiusOffset<0) gRadiusOffset = .0;

               if (!gPad->OpaqueMoving()) DrawGhost();
            } else {
               if (!gPad->OpaqueMoving()) DrawGhost();

               gX += mdx;
               gY += mdy;

               if (!gPad->OpaqueMoving()) DrawGhost();
            }
         } else if (isResizing) {
            if (!gPad->OpaqueResizing()) DrawGhost();

            Float_t dr = mdx*TMath::Cos(fCurrent_ang)+mdy*TMath::Sin(fCurrent_ang);
            if (gRadius+dr>=minRad) {
               gRadius += dr;
            } else {
               gRadius = minRad;
            }

            if (!gPad->OpaqueResizing()) DrawGhost();
         } else if (isRotating) {
            if (!gPad->OpaqueMoving()) DrawGhost();

            if (py>=gPad->YtoAbsPixel(fY)) {
               if (px<oldpx) gAngularOffset--;
               else          gAngularOffset++;
            } else {
               if (px<oldpx) gAngularOffset++;
               else          gAngularOffset--;
            }
            if (gAngularOffset>=360) gAngularOffset = 0;
            if (!gPad->OpaqueMoving()) DrawGhost();
         }

         oldpx = px;
         oldpy = py;
         if ( ((isMovingPie || isMovingSlice) && gPad->OpaqueMoving()) ||
               (isResizing && gPad->OpaqueResizing()) ||
               (isRotating && gPad->OpaqueMoving()) ) isRedrawing = kTRUE;
         else break;

      case kButton1Up:
         if (!isRedrawing) {
            prev_event = kButton1Up;
            gIsUpt = kFALSE;
         }

         if (gROOT->IsEscaped()) {
            gROOT->SetEscape(kFALSE);
            gIsUpt = kFALSE;
            break;
         }

         fX      = gX;
         fY      = gY;
         fRadius = gRadius;
         fRadiusOffsets[fCurrent_slice] = gRadiusOffset;
         SetAngularOffset(fAngularOffset+gAngularOffset);

         gPad->Modified(kTRUE);

         gVirtualX->SetLineColor(-1);
         gVirtualX->SetLineWidth(-1);

         if (isMovingPie)   isMovingPie   = kFALSE;
         if (isMovingSlice) isMovingSlice = kFALSE;
         if (isResizing)    isResizing    = kFALSE;
         if (isRotating)    isResizing    = kFALSE;

         gIsUpt = kFALSE;
         break;

      case kMouseEnter:
         break;

      case kButton1Locate:
         break;

      default:
         // unknown event
         break;
   }
}


//______________________________________________________________________________
const char* TPie::GetEntryLabel(Int_t i)
{
   // Returns the label of the entry number "i".

   return fLabels[i].Data();
}


//______________________________________________________________________________
Int_t TPie::GetEntryFillColor(Int_t i)
{
   // Return the color of the slice number "i".

   return fFillColors[i];
}


//______________________________________________________________________________
Int_t TPie::GetEntryFillStyle(Int_t i)
{
   // Return the style use to fill the slice number "i".

   return fFillStyles[i];
}


//______________________________________________________________________________
Double_t TPie::GetEntryRadiusOffset(Int_t i)
{
   // Return the radial offset's value for the slice number "i".

   return fRadiusOffsets[i];
}


//______________________________________________________________________________
Double_t TPie::GetEntryVal(Int_t i)
{
   // Return the value associated with the slice number "i".

   return fVals[i];
}


//______________________________________________________________________________
void TPie::Init(Int_t np, Double_t ao, Double_t x, Double_t y, Double_t r)
{
   // Common initialization for all constructors.
   // This is a private function called to allocate the memory.

   gIsUpt = kFALSE;

   fAngularOffset = ao;
   fX             = x;
   fY             = y;
   fRadius        = r;
   fNvals         = np;
   fSum           = 0.;
   fSlices        = 0;

   fLabelsOffset = gStyle->GetLabelOffset();

   fLineColors = new Int_t[fNvals];
   fLineStyles = new Int_t[fNvals];
   fLineWidths = new Int_t[fNvals];

   fFillColors = new Int_t[fNvals];
   fFillStyles = new Int_t[fNvals];

   fLabels        = new TString[fNvals];
   fRadiusOffsets = new Double_t[fNvals];
   fVals          = new Double_t[fNvals];

   for (Int_t i=0;i<fNvals;++i) {
      fVals[i]          = 0.;
      fLabels[i]        = "Slice";
      fLabels[i]       += i;
      fLineColors[i]    = 1;
      fLineStyles[i]    = 1;
      fLineWidths[i]    = 1;
      fFillColors[i]    = gStyle->GetColorPalette(i);
      fFillStyles[i]    = 1001;
      fRadiusOffsets[i] = 0.;
   }

   fLabelFormat    = "%txt";
   fFractionFormat = "%3.2f";
   fValFormat      = "%4.2f";
   fPercentFormat  = "%3.1f";
}


//______________________________________________________________________________
void TPie::Paint(Option_t *option)
{
   // Paint a Pie chart in a canvas.
   // The possible option are:
   //
   // "R"  Print the labels along the central "R"adius of slices.
   // "T"  Print the label in a direction "T"angent to circle that describes
   //      the TPie.
   //
   // Other options changing the labels' format are described in
   // TPie::SetLabelFormat().

   MakeSlices();

   TString soption(option);

   bool optionSame(kFALSE);

   // For the label orientation there are 3 possibilities:
   //   0: horizontal
   //   1: radial
   //   2: tangent
   Int_t lblor(0);

   // Parse the options
   Int_t idx;
   // Paint the TPie in an existing canvas
   if ( (idx=soption.Index("same"))>=0 ) {
      optionSame = kTRUE;
      soption.Remove(idx,4);
   }

   // Seek if have to draw the labels around the pie chart
   if ( (idx=soption.Index("t"))>=0 ) {
      lblor = 2;
      soption.Remove(idx,1);
   }

   // Seek if have to paint the labels along the radii
   if ( (idx=soption.Index("r"))>=0 ) {
      lblor = 1;
      soption.Remove(idx,1);
   }

   if (fNvals<=0) {
      Warning("Paint","No vals");
      return;
   }

   if (!fVals) {
      Warning("TPie","No valid arrays of values");
      return;
   }

   // Objects useful to draw labels and slices
   TLatex *textlabel = new TLatex();
   TArc *arc = new TArc();

   // Draw the slices.
   for (Int_t i=0;i<fNvals;++i) {
      // Set the color of the next slice
      arc->SetFillColor(fFillColors[i]);
      arc->SetFillStyle(fFillStyles[i]);
      arc->SetLineColor(fLineColors[i]);
      arc->SetLineStyle(fLineStyles[i]);
      arc->SetLineWidth(fLineWidths[i]);

      // Paint the slice
      Float_t aphi = fSlices[2*i+1]*TMath::Pi()/180.;
      Double_t ax = fX+TMath::Cos(aphi)*fRadiusOffsets[i];
      Double_t ay = fY+TMath::Sin(aphi)*fRadiusOffsets[i];
      arc->PaintEllipse(ax, ay, fRadius, fRadius, fSlices[2*i],
                                                  fSlices[2*i+2], 0.);
   }


   // Loop to place the labels.
   for (Int_t i=0;i<fNvals;++i) {
      Float_t aphi = fSlices[2*i+1]*TMath::Pi()/180.;
      Float_t label_off = fLabelsOffset;

      // Set the font
      textlabel->SetTextFont(GetTextFont());
      textlabel->SetTextSize(GetTextSize());
      textlabel->SetTextColor(GetTextColor());

      textlabel->SetTextAngle(GetTextAngle());

      // Paint the text in the pad
      TString tmptxt  = fLabelFormat;

      tmptxt.ReplaceAll("%txt",fLabels[i]);
      tmptxt.ReplaceAll("%val",Form(fValFormat.Data(),fVals[i]));
      tmptxt.ReplaceAll("%frac",Form(fFractionFormat.Data(),fVals[i]/fSum));
      tmptxt.ReplaceAll("%perc",Form("%3.1f %s",(fVals[i]/fSum)*100,"%"));

      textlabel->SetTitle(tmptxt.Data());
      Double_t h = textlabel->GetYsize();
      Double_t w = textlabel->GetXsize();

      Float_t lx = fX+(fRadius+fRadiusOffsets[i]+label_off)*TMath::Cos(aphi);
      Float_t ly = fY+(fRadius+fRadiusOffsets[i]+label_off)*TMath::Sin(aphi);

      Double_t lblang = GetTextAngle();

      if (lblor==1) {
         lblang += aphi;

         lx += h/2.*TMath::Sin(lblang);
         ly -= h/2.*TMath::Cos(lblang);

         // This control prevent text up-side
         if (aphi>TMath::PiOver2() && aphi<=3.*TMath::PiOver2()) {
            lx     += w*TMath::Cos(lblang)-h*TMath::Sin(lblang);
            ly     += w*TMath::Sin(lblang)+h*TMath::Cos(lblang);
            lblang -= TMath::Pi();
         }
      } else if (lblor==2) {
         lblang += aphi-TMath::PiOver2();
         if (lblang<0) lblang+=TMath::TwoPi();

         lx -= w/2.*TMath::Cos(lblang);
         ly -= w/2.*TMath::Sin(lblang);

         if (aphi>TMath::Pi() && aphi<TMath::TwoPi()) {
            lx     += w*TMath::Cos(lblang)-h*TMath::Sin(lblang);
            ly     += w*TMath::Sin(lblang)+h*TMath::Cos(lblang);
            lblang -= TMath::Pi();
         }
      } else {
         if (aphi>TMath::Pi()/2. && aphi<=TMath::Pi()*3./2.) lx -= w;
         if (aphi>=TMath::Pi() && aphi<TMath::TwoPi())       ly -= h;
      }

      textlabel->PaintLatex(lx,ly,
                            lblang*180/TMath::Pi()+GetTextAngle(),
                            GetTextSize(), tmptxt.Data());
   }

   delete arc;
   delete textlabel;

   if (optionSame) return;

   // Draw title
   TPaveText *title = 0;
   TObject *obj;
   if ((obj = gPad->GetListOfPrimitives()->FindObject("title"))) {
      title = dynamic_cast<TPaveText*>(obj);
   }

   // Check the OptTitle option
   if (strlen(GetTitle()) == 0 || gStyle->GetOptTitle() <= 0) {
      if (title) delete title;
      return;
   }

   // Height and width of the title
   Double_t ht = gStyle->GetTitleH();
   Double_t wt = gStyle->GetTitleW();
   if (ht<=0) ht = 1.1*gStyle->GetTitleFontSize();
   if (ht<=0) ht = 0.05; // minum height
   if (wt<=0) { // eval the width of the title
      TLatex l;
      l.SetTextSize(ht);
      l.SetTitle(GetTitle());
      // adjustment in case the title has several lines (#splitline)
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

   title = new TPaveText(xpos,ypos-ht,xpos+wt,ypos,"blNDC");
   title->SetFillColor(gStyle->GetTitleFillColor());
   title->SetFillStyle(gStyle->GetTitleStyle());
   title->SetName("title");

   title->SetBorderSize(gStyle->GetTitleBorderSize());
   title->SetTextColor(gStyle->GetTitleTextColor());
   title->SetTextFont(gStyle->GetTitleFont(""));
   if (gStyle->GetTitleFont("")%10 > 2)
      title->SetTextSize(gStyle->GetTitleFontSize());
   title->AddText(GetTitle());

   title->SetBit(kCanDelete);

   title->Draw();
   title->Paint();
}


//______________________________________________________________________________
void TPie::SavePrimitive(ostream &out, Option_t *)
{
   // Save primitive as a C++ statement(s) on output stream out

   out << "   " << endl;
   if (gROOT->ClassSaved(TPie::Class())) {
      out << "   ";
   } else {
      out << "   TPie *";
   }
   out << GetName() << " = new TPie(\"" << GetName() << "\", \"" << GetTitle()
       << "\", " << fNvals << ");" << endl;
   out << "   " << GetName() << "->SetCircle(" << fX << ", " << fY << ", "
       << fRadius << ");" << endl;

   SaveTextAttributes(out,GetName(),
                      GetTextAlign(), GetTextAngle(), GetTextColor(),
                      GetTextFont(), GetTextSize());

   // save the values fo the slices
   for (Int_t i=0;i<fNvals;++i) {
      out << "   " << GetName() << "->SetEntryLabel( " << i << ", \""
          << fLabels[i] << "\");" << endl;
      out << "   " << GetName() << "->SetEntryFillColor( " << i << ", " <<
          fFillColors[i] << ");" << endl;
      out << "   " << GetName() << "->SetEntryFillStyle( " << i << ", " <<
          fFillStyles[i] << ");" << endl;
      out << "   " << GetName() << "->SetEntryLineColor( " << i << ", " <<
          fLineColors[i] << ");" << endl;
      out << "   " << GetName() << "->SetEntryLineStyle( " << i << ", " <<
          fLineStyles[i] << ");" << endl;
      out << "   " << GetName() << "->SetEntryLineWidth( " << i << ", " <<
          fLineWidths[i] << ");" << endl;
      out << "   " << GetName() << "->SetEntryRadiusOffset( " << i << ", " <<
          fRadiusOffsets[i] << ");" << endl;
      out << "   " << GetName() << "->SetEntryVal( " << i << ", " << fVals[i]
          << ");" << endl;
   }

   out << "   " << GetName() << "->Draw();" << endl;
}


//______________________________________________________________________________
void TPie::SetCircle(Double_t x, Double_t y, Double_t rad)
{
   // Set the coordinates of the circle that describe the pie:
   // - the 1st and the 2nd arguments are the x and y center's coordinates.
   // - the 3rd value is the pie-chart's radius.
   //
   // All the coordinates are in NDC space.

   fX      = x;
   fY      = y;
   fRadius = rad;
}


//______________________________________________________________________________
void TPie::SetEntryLabel(Int_t i, const char *text)
{
   // Set slice number "i" label. The first parameter is the index of the slice,
   // the other is the label text.

   // Set the Label of a single slice
   if (i>=0 && i<fNvals) fLabels[i] = text;
}


//______________________________________________________________________________
void TPie::SetEntryLineColor(Int_t i, Int_t color)
{
   // Set the color for the slice's outline. "i" is the slice number.

   fLineColors[i] = color;
}


//______________________________________________________________________________
void TPie::SetEntryLineStyle(Int_t i, Int_t style)
{
   // Set the style for the slice's outline. "i" is the slice number.

   fLineStyles[i] = style;
}


//______________________________________________________________________________
void TPie::SetEntryLineWidth(Int_t i, Int_t width)
{
   // Set the width of the slice's outline. "i" is the slice number.

   fLineWidths[i] = width;
}


//______________________________________________________________________________
void TPie::SetEntryFillColor(Int_t i, Int_t color)
{
   // Set the color for the slice "i".

   fFillColors[i] = color;
}


//______________________________________________________________________________
void TPie::SetEntryFillStyle(Int_t i, Int_t style)
{
   // Set the fill style for the "i" slice

   fFillStyles[i] = style;
}


//______________________________________________________________________________
void TPie::SetEntryRadiusOffset(Int_t i, Double_t shift)
{
   // Set the distance, in the direction of the radius of the slice.

   fRadiusOffsets[i] = shift;
}


//______________________________________________________________________________
void TPie::SetEntryVal(Int_t i, Double_t val)
{
   // Set the value of a slice

   if (i>=0 && i<fNvals) fVals[i] = val;

   MakeSlices(kTRUE);
}


//______________________________________________________________________________
void TPie::SetFillColors(Int_t *colors)
{
   // Set the fill colors for all the TPie's slices.

   if (!colors) return;
   for (Int_t i=0;i<fNvals;++i) fFillColors[i] = colors[i];
}


//______________________________________________________________________________
void TPie::SetLabel(const char *txt)
{
   // Set the label for the slice under the mouse pointer,
   // this method is useful only when used in the GUI, not
   // in macros.

   if (fCurrent_slice>=0 && fCurrent_slice<fNvals)
      fLabels[fCurrent_slice] = txt;
}


//______________________________________________________________________________
void TPie::SetLabelFormat(const char *fmt)
{
   // This method is used to customize the label format. The format string
   // must contain one of these modifiers:
   //
   // - %txt  : to print the text label associated with the slice
   // - %val  : to print the numeric value of the slice
   // - %frac : to print the relative fraction of this slice
   // - %perc : to print the % of this slice
   //
   // ex. : mypie->SetLabelFormat("%txt (%frac)");

   fLabelFormat = fmt;
}


//______________________________________________________________________________
void TPie::SetFractionFormat(const char *fmt)
{
   // Set numeric format in the label, is used if the label format
   // there is the modifier %frac, in this case the value is printed
   // using this format.
   //
   // The numeric format use the standard C modifier used in stdio library:
   // %f, %2.1$, %e... for further documentation you can use the printf
   // mapage ("man 3 printf" on linux)
   //
   // ex. : mypie->SetLabelFormat("%txt (%frac)");
   //       mypie->SetFractionFormat("2.1f");

   fFractionFormat = fmt;
}


//______________________________________________________________________________
void TPie::SetLabels(const char *lbls[])
{
   // Set the labels for all the slices.

   if (!lbls) return;
   for (Int_t i=0;i<fNvals;++i) fLabels[i] = lbls[i];
}


//______________________________________________________________________________
void TPie::SetLabelsOffset(Float_t labelsoffset)
{
   // Set the distance between the label end the external line of the TPie.

   fLabelsOffset = labelsoffset;
}


//______________________________________________________________________________
void TPie::SetAngularOffset(Double_t offset)
{
   // Set the global angular offset for slices in degree [0,360]

   fAngularOffset = offset;

   while (fAngularOffset>=360.) fAngularOffset -= 360.;
   while (fAngularOffset<0.)    fAngularOffset += 360.;

   MakeSlices(kTRUE);
}


//______________________________________________________________________________
void TPie::SetPercentFormat(const char *fmt)
{
   // Set the numeric format for the percent value of a slice, default: %3.1f

   fPercentFormat = fmt;
}


//______________________________________________________________________________
void TPie::SetRadius(Double_t rad)
{
   // Set the pie chart's radius' value.

   if (rad>0) {
      fRadius = rad;
   } else {
      Warning("SetRadius",
              "It's not possible set the radius to a negative value");
   }
}


//______________________________________________________________________________
void TPie::SetValFormat(const char *fmt)
{
   // Set the numeric format the slices' values.
   // Used by %val (see SetLabelFormat()).

   fValFormat = fmt;
}


//______________________________________________________________________________
void TPie::SetX(Double_t x)
{
   // Set X value.

   fX = x;
}


//______________________________________________________________________________
void TPie::SetY(Double_t y)
{
   // Set Y value.

   fY = y;
}


//______________________________________________________________________________
void TPie::MakeSlices(Bool_t force)
{
   // Make the slices.
   // If they already exist it does nothing unless force=kTRUE.

   if (fSlices && !force) return;

   fSum = .0;

   for (Int_t i=0;i<fNvals;++i) {
      if (fVals[i]<0) {
         Warning("MakeSlices",
                 "Negative values in TPie, absolute value iwll be used");
         fVals[i] *= -1;
      }
      fSum += fVals[i];
   }

   if (fSum<=.0) {
      Warning("Paint","Sum of values is <= 0");
      return;
   }

   if (!fSlices) fSlices = new Float_t[2*fNvals+1];

   fSlices[0] = fAngularOffset;
   for (Int_t i=0;i<fNvals;++i) { // loop to place the labels
      // size of this slice
      Float_t dphi   = fVals[i]/fSum*360.;
      fSlices[2*i+1] = fSlices[2*i]+dphi/2.;
      fSlices[2*i+2] = fSlices[2*i]+dphi;
   }
}
