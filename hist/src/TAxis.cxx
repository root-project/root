// @(#)root/hist:$Name:  $:$Id: TAxis.cxx,v 1.6 2000/06/15 06:51:49 brun Exp $
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <iostream.h>
#include "TAxis.h"
#include "TVirtualPad.h"
#include "TVirtualX.h"
#include "TStyle.h"
#include "TView.h"
#include "TError.h"
#include "TH1.h"

ClassImp(TAxis)

//______________________________________________________________________________
//
// This class manages histogram axis. It is referenced by TH1 and TGraph.
// To make a graphical representation of an histogram axis, this class
// references the TGaxis class.
//
// TAxis supports axis with fixed or variable bin sizes.
// Labels may be associated to individual bins.
//
//    see examples of various axis representations drawn by class TGaxis.
//

//______________________________________________________________________________
TAxis::TAxis(): TNamed(), TAttAxis()
{
   fNbins   = 0;
   fXmin    = 0;
   fXmax    = 0;
   fFirst   = 0;
   fLast    = 0;
   fXlabels = 0;
   fParent  = 0;
   fTimeDisplay = 0;
}

//______________________________________________________________________________
TAxis::TAxis(Int_t nbins,Axis_t xlow,Axis_t xup): TNamed(), TAttAxis()
{
//*-*-*-*-*-*-*-*Axis constructor for axis with fix bin size*-*-*-*-*-*-*-*
//*-*            ===========================================

   fParent  = 0;
   Set(nbins,xlow,xup);
}

//______________________________________________________________________________
TAxis::TAxis(Int_t nbins,Axis_t *xbins): TNamed(), TAttAxis()
{
//*-*-*-*-*-*-*-*Axis constructor for variable bin size*-*-*-*-*-*-*-*-*-*-*
//*-*            ======================================

   fParent  = 0;
   Set(nbins,xbins);
}

//______________________________________________________________________________
TAxis::~TAxis()
{
   if (fXlabels) { delete [] fXlabels; fXlabels = 0; }
}

//______________________________________________________________________________
TAxis::TAxis(const TAxis &axis)
{
   ((TAxis&)axis).Copy(*this);
}

//______________________________________________________________________________
void TAxis::CenterTitle(Bool_t center)
{
//   if center = kTRUE axis title will be centered
//   default is right adjusted
   if (center) SetBit(kCenterTitle);
   else        ResetBit(kCenterTitle);
}

//______________________________________________________________________________
const char *TAxis::ChooseTimeFormat(Double_t axislength)
{
// Choose a reasonable time format from the coordinates in the active pad
// and the number of divisions in this axis
// If orientation = "X", the horizontal axis of the pad will be used for ref.
// If orientation = "Y", the vertical axis of the pad will be used for ref.

   const char *formatstr;
   Int_t reasformat = 0;
   Int_t ndiv,nx1,nx2,N;
   Double_t awidth;
   Double_t length;

   if (!axislength) {
      length = gPad->GetUxmax() - gPad->GetUxmin();
   } else {
      length = axislength;
   }

   ndiv = GetNdivisions();
   if (ndiv > 1000) {
      nx2   = ndiv/100;
      nx1   = TMath::Max(1, ndiv%100);
      ndiv = 100*nx2 + Int_t(Double_t(nx1)*gPad->GetAbsWNDC());
   }
   ndiv = TMath::Abs(ndiv);
   N = ndiv - (ndiv/100)*100;
   awidth = length/N;

//  width in seconds ?
   if (awidth>=.5) {
      reasformat = 1;
//  width in minutes ?
      if (awidth>=30) {
         awidth /= 60; reasformat = 2;
//   width in hours ?
         if (awidth>=30) {
            awidth /=60; reasformat = 3;
//   width in days ?
            if (awidth>=12) {
               awidth /= 24; reasformat = 4;
//   width in months ?
               if (awidth>=15.218425) {
                  awidth /= 30.43685; reasformat = 5;
//   width in years ?
                  if (awidth>=6) {
                     awidth /= 12; reasformat = 6;
                  }
               }
            }
         }
      }
   }
//   set reasonable format
   switch (reasformat) {
      case 0:
        formatstr = "%Ss";
        break;
      case 1:
        formatstr = "%Mm%S";
        break;
      case 2:
        formatstr = "%Hh%M";
        break;
      case 3:
        formatstr = "%d-%Hh";
        break;
      case 4:
        formatstr = "%d/%m";
        break;
      case 5:
        formatstr = "%d/%m/%y";
        break;
      case 6:
        formatstr = "%d/%m/%y";
        break;
      default:
        formatstr = "%H:%M:%S";
        break;
   }
   return formatstr;
}

//______________________________________________________________________________
void TAxis::Copy(TObject &obj)
{
//*-*-*-*-*-*-*-*-*-*Copy axis structure to another axis-*-*-*-*-*-*-*-*-*-*-*
//*-*                ===================================

   TNamed::Copy(obj);
   TAttAxis::Copy(((TAxis&)obj));
   ((TAxis&)obj).fNbins  = fNbins;
   ((TAxis&)obj).fXmin   = fXmin;
   ((TAxis&)obj).fXmax   = fXmax;
   ((TAxis&)obj).fFirst  = fFirst;
   ((TAxis&)obj).fLast   = fLast;
   ((TAxis&)obj).fXlabels= 0;
   fXbins.Copy(((TAxis&)obj).fXbins);
   ((TAxis&)obj).fTimeFormat   = fTimeFormat;
   ((TAxis&)obj).fTimeDisplay  = fTimeDisplay;
   ((TAxis&)obj).fParent       = fParent;
}

//______________________________________________________________________________
Int_t TAxis::DistancetoPrimitive(Int_t, Int_t)
{
//*-*-*-*-*-*-*-*-*-*-*Compute distance from point px,py to an axis*-*-*-*-*-*
//*-*                  ============================================

   return 9999;
}

//______________________________________________________________________________
void TAxis::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//  This member function is called when an axis is clicked with the locator
//
//  The axis range is set between the position where the mouse is pressed
//  and the position where it is released.
//  If the mouse position is outside the current axis range when it is released
//  the axis is unzoomed with the corresponding proportions.
//  Note that the mouse does not need to be in the pad or even canvas
//  when it is released.

   if (!gPad->IsEditable()) return;

   gPad->SetCursor(kHand);

   TView *view = gPad->GetView();
   static Int_t axisNumber;
   static Double_t ratio1, ratio2;
   static Int_t px1old, py1old, px2old, py2old;
   Int_t bin1, bin2, first, last;
   Double_t temp, xmin,xmax;

   switch (event) {

   case kButton1Down:
      axisNumber = 1;
      if (!strcmp(GetName(),"xaxis")) axisNumber = 1;
      if (!strcmp(GetName(),"yaxis")) axisNumber = 2;
      if (!axisNumber) return;
      if (view) {
         view->GetDistancetoAxis(axisNumber, px, py, ratio1);
      } else {
         if (axisNumber == 1) {
            ratio1 = (gPad->AbsPixeltoX(px) - gPad->GetUxmin())/(gPad->GetUxmax() - gPad->GetUxmin());
            px1old = gPad->XtoAbsPixel(gPad->GetUxmin()+ratio1*(gPad->GetUxmax() - gPad->GetUxmin()));
            py1old = gPad->YtoAbsPixel(gPad->GetUymin());
            px2old = px1old;
            py2old = gPad->YtoAbsPixel(gPad->GetUymax());
         } else {
            ratio1 = (gPad->AbsPixeltoY(py) - gPad->GetUymin())/(gPad->GetUymax() - gPad->GetUymin());
            py1old = gPad->YtoAbsPixel(gPad->GetUymin()+ratio1*(gPad->GetUymax() - gPad->GetUymin()));
            px1old = gPad->XtoAbsPixel(gPad->GetUxmin());
            px2old = gPad->XtoAbsPixel(gPad->GetUxmax());
            py2old = py1old;
         }
         gVirtualX->DrawBox(px1old, py1old, px2old, py2old, TVirtualX::kHollow);
      }
      gVirtualX->SetLineColor(-1);
      // No break !!!

   case kButton1Motion:
      if (view) {
         view->GetDistancetoAxis(axisNumber, px, py, ratio2);
      } else {
         gVirtualX->DrawBox(px1old, py1old, px2old, py2old, TVirtualX::kHollow);
         if (axisNumber == 1) {
            ratio2 = (gPad->AbsPixeltoX(px) - gPad->GetUxmin())/(gPad->GetUxmax() - gPad->GetUxmin());
            px2old = gPad->XtoAbsPixel(gPad->GetUxmin()+ratio2*(gPad->GetUxmax() - gPad->GetUxmin()));
         } else {
            ratio2 = (gPad->AbsPixeltoY(py) - gPad->GetUymin())/(gPad->GetUymax() - gPad->GetUymin());
            py2old = gPad->YtoAbsPixel(gPad->GetUymin()+ratio2*(gPad->GetUymax() - gPad->GetUymin()));
         }
         gVirtualX->DrawBox(px1old, py1old, px2old, py2old, TVirtualX::kHollow);
      }
   break;

   case kButton1Up:
      if (view) {
         view->GetDistancetoAxis(axisNumber, px, py, ratio2);
         if (ratio1 > ratio2) {
            temp   = ratio1;
            ratio1 = ratio2;
            ratio2 = temp;
         }
         if (ratio2 - ratio1 > 0.05) {
            if (fFirst > 0) first = fFirst;
            else            first = 1;
            if (fLast > 0) last = fLast;
            else           last = fNbins;
            bin1 = first + Int_t((last-first+1)*ratio1);
            bin2 = first + Int_t((last-first+1)*ratio2);
            SetRange(bin1, bin2);
            gPad->Modified(kTRUE);
         }
      } else {
         if (axisNumber == 1) {
            ratio2 = (gPad->AbsPixeltoX(px) - gPad->GetUxmin())/(gPad->GetUxmax() - gPad->GetUxmin());
            xmin = gPad->GetUxmin() +ratio1*(gPad->GetUxmax() - gPad->GetUxmin());
            xmax = gPad->GetUxmin() +ratio2*(gPad->GetUxmax() - gPad->GetUxmin());
            if (gPad->GetLogx()) {
               xmin = gPad->PadtoX(xmin);
               xmax = gPad->PadtoX(xmax);
            }
         } else {
            ratio2 = (gPad->AbsPixeltoY(py) - gPad->GetUymin())/(gPad->GetUymax() - gPad->GetUymin());
            xmin = gPad->GetUymin() +ratio1*(gPad->GetUymax() - gPad->GetUymin());
            xmax = gPad->GetUymin() +ratio2*(gPad->GetUymax() - gPad->GetUymin());
            if (gPad->GetLogy()) {
               xmin = gPad->PadtoY(xmin);
               xmax = gPad->PadtoY(xmax);
            }
         }
         if (xmin > xmax) {
            temp   = xmin;
            xmin   = xmax;
            xmax   = temp;
            temp   = ratio1;
            ratio1 = ratio2;
            ratio2 = temp;
         }
         if (ratio2 - ratio1 > 0.05) {
            TH1 *hobj = (TH1*)fParent;
            bin1 = FindFixBin(xmin);
            bin2 = FindFixBin(xmax);
            if (axisNumber == 1) SetRange(bin1,bin2);
            if (axisNumber == 2 && hobj) {
               if (hobj->GetDimension() == 1) {
                  hobj->SetMinimum(xmin);
                  hobj->SetMaximum(xmax);
               } else {
                  SetRange(bin1,bin2);
               }
            }
            gPad->Modified(kTRUE);
         }
      }
      gVirtualX->SetLineColor(-1);
      break;
   }
}

//______________________________________________________________________________
Int_t TAxis::FindBin(Axis_t x)
{
//*-*-*-*-*-*-*-*-*Find bin number corresponding to abscissa x-*-*-*-*-*-*
//*-*              ===========================================
//
// If x is underflow or overflow, attempt to rebin histogram

   Int_t bin;
   if (x < fXmin) {              //*-* underflow
      bin = 0;
      if (fParent == 0) return bin;
      if (!fParent->TestBit(TH1::kCanRebin)) return bin;
      ((TH1*)fParent)->RebinAxis(x,GetName());
      return FindFixBin(x);
   } else  if ( !(x < fXmax)) {     //*-* overflow  (note the way to catch NaN
      bin = fNbins+1;
      if (fParent == 0) return bin;
      if (!fParent->TestBit(TH1::kCanRebin)) return bin;
      ((TH1*)fParent)->RebinAxis(x,GetName());
      return FindFixBin(x);
   } else {
      if (!fXbins.fN) {        //*-* fix bins
         bin = 1 + int (fNbins*(x-fXmin)/(fXmax-fXmin) );
      } else {                  //*-* variable bin sizes
         for (bin =1; x >= fXbins.fArray[bin]; bin++);
      }
   }
   return bin;
}

//______________________________________________________________________________
Int_t TAxis::FindFixBin(Axis_t x)
{
//*-*-*-*-*-*-*-*-*Find bin number corresponding to abscissa x-*-*-*-*-*-*
//*-*              ===========================================
   Int_t bin;
   if (x < fXmin) {              //*-* underflow
      bin = 0;
   } else  if ( !(x < fXmax)) {     //*-* overflow  (note the way to catch NaN
      bin = fNbins+1;
   } else {
      if (!fXbins.fN) {        //*-* fix bins
         bin = 1 + int (fNbins*(x-fXmin)/(fXmax-fXmin) );
      } else {                  //*-* variable bin sizes
         for (bin =1; x >= fXbins.fArray[bin]; bin++);
      }
   }
   return bin;
}

//______________________________________________________________________________
Int_t TAxis::GetFirst()
{
//             return first bin on the axis
//       - 0 if no range defined

   if (!TestBit(kAxisRange)) return 1;
   return fFirst;
}

//______________________________________________________________________________
Int_t TAxis::GetLast()
{
//             return last bin on the axis
//       - fNbins if no range defined

   if (!TestBit(kAxisRange)) return fNbins;
   return fLast;
}

//______________________________________________________________________________
Axis_t TAxis::GetBinCenter(Int_t bin)
{
//*-*-*-*-*-*-*-*-*-*-*Return center of bin*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ====================
  Axis_t binwidth;
  if (!fXbins.fN || bin<1 || bin>fNbins) {
     binwidth = (fXmax - fXmin) / Axis_t(fNbins);
     return fXmin + (bin-1) * binwidth + 0.5*binwidth;
  } else {
     binwidth = fXbins.fArray[bin] - fXbins.fArray[bin-1];
     return fXbins.fArray[bin-1] + 0.5*binwidth;
  }
}

//______________________________________________________________________________
Axis_t TAxis::GetBinLowEdge(Int_t bin)
{
//*-*-*-*-*-*-*-*-*-*-*Return low edge of bin-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ======================

  if (fXbins.fN && bin > 0 && bin <=fNbins) return fXbins.fArray[bin-1];
  Axis_t binwidth = (fXmax - fXmin) / Axis_t(fNbins);
  return fXmin + (bin-1) * binwidth;
}

//______________________________________________________________________________
Axis_t TAxis::GetBinUpEdge(Int_t bin)
{
//*-*-*-*-*-*-*-*-*-*-*Return up edge of bin-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ======================

  Axis_t binwidth;
  if (!fXbins.fN || bin<1 || bin>fNbins) {
     binwidth = (fXmax - fXmin) / Axis_t(fNbins);
     return fXmin + bin*binwidth;
  } else {
     binwidth = fXbins.fArray[bin] - fXbins.fArray[bin-1];
     return fXbins.fArray[bin-1] + binwidth;
  }
}

//______________________________________________________________________________
Axis_t TAxis::GetBinWidth(Int_t bin)
{
//*-*-*-*-*-*-*-*-*-*-*Return bin width-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ================
  if (bin <1 ) bin = 1;
  if (bin >fNbins) bin = fNbins;
  if (!fXbins.fN)  return (fXmax - fXmin) / Axis_t(fNbins);
   return fXbins.fArray[bin] - fXbins.fArray[bin-1];
}


//______________________________________________________________________________
void TAxis::GetCenter(Axis_t *center)
{
//*-*-*-*-*-*-*-*-*Return an array with the center of all bins-*-*-*-*-*-*-*
//*-*              ===========================================
  Int_t bin;
  for (bin=1; bin<=fNbins; bin++) *(center + bin-1) = GetBinCenter(bin);
}

//______________________________________________________________________________
void TAxis::GetLowEdge(Axis_t *edge)
{
//*-*-*-*-*-*-*-*-*Return an array with the lod edge of all bins-*-*-*-*-*-*-*
//*-*              =============================================
  Int_t bin;
  for (bin=1; bin<=fNbins; bin++) *(edge + bin-1) = GetBinLowEdge(bin);
}
//--------------------------------------------------------------------
//                     LABELs methods
//--------------------------------------------------------------------
//______________________________________________________________________________
char *TAxis::GetBinLabel(Int_t)
{
//*-*-*-*-*-*-*-*-*-*Return label associated to bin-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ==============================

   const char *snull = "";
   return  (char*)snull;
}

//______________________________________________________________________________
void TAxis::GetLabel(char *)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

}

//___________________________________________________________________________
void TAxis::RotateTitle(Bool_t rotate)
{
//    rotate title by 180 degrees
//    by default the title is drawn right adjusted.
//    if rotate is TRUE, the title is left adjusted at the end of the axis
//    and rotated by 180 degrees

   if (rotate) SetBit(kRotateTitle);
   else        ResetBit(kRotateTitle);
}

//______________________________________________________________________________
void TAxis::Set(Int_t nbins, Axis_t xlow, Axis_t xup)
{
//*-*-*-*-*-*-*-*-*Initialize axis with fix bins*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =============================
   fNbins   = nbins;
   fXmin    = xlow;
   fXmax    = xup;
   fFirst   = 0;
   fLast    = 0;
   fXlabels = 0;
   fTitle   = "";
   char name[64];
   sprintf(name,"%s%s",GetName(),"x");
   TAttAxis::ResetAttAxis(name);
   fTimeDisplay = 0;
   SetTimeFormat();
}

//______________________________________________________________________________
void TAxis::Set(Int_t nbins, Float_t *xbins)
{
//*-*-*-*-*-*-*-*-*Initialize axis with variable bins*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              ==================================
   Int_t bin;
   fNbins  = nbins;
   fXbins.Set(fNbins+1);
   for (bin=0; bin<= fNbins; bin++)
      fXbins.fArray[bin] = xbins[bin];
   for (bin=1; bin<= fNbins; bin++)
      if (fXbins.fArray[bin] < fXbins.fArray[bin-1])
         Error("TAxis::Set", "bins must be in increasing order");
   fXmin      = fXbins.fArray[0];
   fXmax      = fXbins.fArray[fNbins];
   fFirst     = 0;
   fLast      = 0;
   fXlabels   = 0;
   fTitle     = "";
   char name[64];
   sprintf(name,"%s%s","x",GetName());
   TAttAxis::ResetAttAxis(name);
   fTimeDisplay = 0;
   SetTimeFormat();
}

//______________________________________________________________________________
void TAxis::Set(Int_t nbins, Axis_t *xbins)
{
//*-*-*-*-*-*-*-*-*Initialize axis with variable bins*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              ==================================
   Int_t bin;
   fNbins  = nbins;
   fXbins.Set(fNbins+1);
   for (bin=0; bin<= fNbins; bin++)
      fXbins.fArray[bin] = xbins[bin];
   for (bin=1; bin<= fNbins; bin++)
      if (fXbins.fArray[bin] < fXbins.fArray[bin-1])
         Error("TAxis::Set", "bins must be in increasing order");
   fXmin      = fXbins.fArray[0];
   fXmax      = fXbins.fArray[fNbins];
   fFirst     = 0;
   fLast      = 0;
   fXlabels   = 0;
   fTitle     = "";
   char name[64];
   sprintf(name,"%s%s","x",GetName());
   TAttAxis::ResetAttAxis(name);
   fTimeDisplay = 0;
   SetTimeFormat();
}

//______________________________________________________________________________
void TAxis::SetLabel(const char *)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

}

//______________________________________________________________________________
void TAxis::SetBinLabel(Int_t, char *)
{
//*-*-*-*-*-*-*-*-*Set label associated to bin-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              ===========================

}

//______________________________________________________________________________
void TAxis::SetLimits(Axis_t xmin, Axis_t xmax)
{
//          Set the axis limits

   fXmin = xmin;
   fXmax = xmax;
}

//______________________________________________________________________________
void TAxis::SetRange(Int_t first, Int_t last)
{
//          Set the viewing range for the axis from bin first to last

   if (last == 0) last = fNbins;
   if (last < first) return;
   if (last > fNbins) last = fNbins;
   if (first < 1) first = 1;
   if (first == 1 && last == fNbins) {
      SetBit(kAxisRange,0);
      fFirst = 0;
      fLast  = 0;
   } else {
      SetBit(kAxisRange,1);
      fFirst = first;
      fLast  = last;
   }
}

//______________________________________________________________________________
void TAxis::SetTimeFormat(const char *tformat)
{
//*-*-*-*-*-*-*-*-*-*-*Change the format used for time plotting *-*-*-*-*-*-*-*
//*-*                  ========================================
//  The format string for date and time use the same options as the one used
//  in the standard strftime C function, i.e. :
//    for date :
//      %a abbreviated weekday name
//      %b abbreviated month name
//      %d day of the month (01-31)
//      %m month (01-12)
//      %y year without century
//
//    for time :
//      %H hour (24-hour clock)
//      %I hour (12-hour clock)
//      %p local equivalent of AM or PM
//      %M minute (00-59)
//      %S seconds (00-61)
//      %% %

   fTimeFormat = tformat;
}

//______________________________________________________________________________
void TAxis::Streamer(TBuffer &R__b)
{
   // Stream an object of class TAxis.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      TNamed::Streamer(R__b);
      TAttAxis::Streamer(R__b);
      R__b >> fNbins;
      if (R__v < 5) {
         Float_t xmin,xmax;
         R__b >> xmin; fXmin = xmin;
         R__b >> xmax; fXmax = xmax;
         Float_t *xbins = 0;
         Int_t n = R__b.ReadArray(xbins);
         fXbins.Set(n);
         for (Int_t i=0;i<n;i++) fXbins.fArray[i] = xbins[i];
         delete [] xbins;         
      } else {
         R__b >> fXmin;
         R__b >> fXmax;
         fXbins.Streamer(R__b);
      }
      if (R__v > 2) {
         R__b >> fFirst;
         R__b >> fLast;
          // following lines required to repair for a bug in Root version 1.03
         if (fFirst < 0 || fFirst > fNbins) fFirst = 0;
         if (fLast  < 0 || fLast  > fNbins) fLast  = 0;
         if (fLast  < fFirst) { fFirst = 0; fLast = 0;}
         if (fFirst ==0 && fLast == 0) SetBit(kAxisRange,0);
      }
      if (R__v > 3) {
         R__b >> fTimeDisplay;
         fTimeFormat.Streamer(R__b);
      } else {
         SetTimeFormat();
      }
      R__b.CheckByteCount(R__s, R__c, TAxis::IsA());
   } else {
      R__c = R__b.WriteVersion(TAxis::IsA(), kTRUE);
      TNamed::Streamer(R__b);
      TAttAxis::Streamer(R__b);
      R__b << fNbins;
      R__b << fXmin;
      R__b << fXmax;
      fXbins.Streamer(R__b);
      R__b << fFirst;
      R__b << fLast;
      R__b << fTimeDisplay;
      fTimeFormat.Streamer(R__b);
      R__b.SetByteCount(R__c, kTRUE);
   }
}

//______________________________________________________________________________
void TAxis::UnZoom()
{
   // Reset first & last bin to the full range

   SetRange(0,0);

   // loop on all histograms in the same pad
   if (!gPad) return;
   TIter next(gPad->GetListOfPrimitives());
   TObject *obj;
   while ((obj = next())) {
      if (obj->InheritsFrom("TH1")) {
         TH1 *hobj = (TH1*)obj;
         if (strstr(GetName(),"xaxis")) hobj->GetXaxis()->SetRange(0,0);
         if (strstr(GetName(),"zaxis")) hobj->GetZaxis()->SetRange(0,0);
         if (strstr(GetName(),"yaxis")) {
            hobj->GetYaxis()->SetRange(0,0);
            if (hobj->GetDimension() == 1) {
               hobj->SetMinimum();
               hobj->SetMaximum();
            }
         }
      }
   }
}
