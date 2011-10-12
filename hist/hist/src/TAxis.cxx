// @(#)root/hist:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TAxis.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TError.h"
#include "THashList.h"
#include "TH1.h"
#include "TObjString.h"
#include "TDatime.h"
#include "TROOT.h"
#include "TClass.h"
#include "TMath.h"
#include <time.h>

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
   // Default constructor.

   fNbins   = 1;
   fXmin    = 0;
   fXmax    = 1;
   fFirst   = 0;
   fLast    = 0;
   fParent  = 0;
   fLabels  = 0;
   fBits2   = 0;
   fTimeDisplay = 0;
}

//______________________________________________________________________________
TAxis::TAxis(Int_t nbins,Double_t xlow,Double_t xup): TNamed(), TAttAxis()
{
   // Axis constructor for axis with fix bin size

   fParent  = 0;
   fLabels  = 0;
   Set(nbins,xlow,xup);
}

//______________________________________________________________________________
TAxis::TAxis(Int_t nbins,const Double_t *xbins): TNamed(), TAttAxis()
{
   // Axis constructor for variable bin size

   fParent  = 0;
   fLabels  = 0;
   Set(nbins,xbins);
}

//______________________________________________________________________________
TAxis::~TAxis()
{
   // Destructor.

   if (fLabels) {
      fLabels->Delete();
      delete fLabels;
      fLabels = 0;
   }
}

//______________________________________________________________________________
TAxis::TAxis(const TAxis &axis) : TNamed(axis), TAttAxis(axis)
{
   // Copy constructor.

   ((TAxis&)axis).Copy(*this);
}

//______________________________________________________________________________
void TAxis::CenterLabels(Bool_t center)
{
   //   if center = kTRUE axis labels will be centered (hori axes only)
   //   on the bin center
   //   default is to center on the primary tick marks
   //   This option does not make sense if there are more bins than tick marks

   if (center) SetBit(kCenterLabels);
   else        ResetBit(kCenterLabels);
}

//______________________________________________________________________________
Bool_t TAxis::GetCenterLabels() const
{
   // Return kTRUE if kCenterLabels bit is set, kFALSE otherwise.

   return TestBit(kCenterLabels) ? kTRUE : kFALSE;
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
Bool_t TAxis::GetCenterTitle() const
{
   // Return kTRUE if kCenterTitle bit is set, kFALSE otherwise.

   return TestBit(kCenterTitle) ? kTRUE : kFALSE;
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
   Int_t ndiv,nx1,nx2,n;
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
   n = ndiv - (ndiv/100)*100;
   awidth = length/n;

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
                     if (awidth>=2) {
                        awidth /= 12; reasformat = 7;
                     }
                  }
               }
            }
         }
      }
   }
//   set reasonable format
   switch (reasformat) {
      case 0:
        formatstr = "%S";
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
      case 7:
        formatstr = "%m/%y";
        break;
   }
   return formatstr;
}

//______________________________________________________________________________
void TAxis::Copy(TObject &obj) const
{
   // Copy axis structure to another axis

   TNamed::Copy(obj);
   TAttAxis::Copy(((TAxis&)obj));
   ((TAxis&)obj).fNbins  = fNbins;
   ((TAxis&)obj).fXmin   = fXmin;
   ((TAxis&)obj).fXmax   = fXmax;
   ((TAxis&)obj).fFirst  = fFirst;
   ((TAxis&)obj).fLast   = fLast;
   ((TAxis&)obj).fBits2  = fBits2;
   fXbins.Copy(((TAxis&)obj).fXbins);
   ((TAxis&)obj).fTimeFormat   = fTimeFormat;
   ((TAxis&)obj).fTimeDisplay  = fTimeDisplay;
   ((TAxis&)obj).fParent       = fParent;
   ((TAxis&)obj).fLabels       = 0;
   if (fLabels) {
      for (Int_t i=1;i<=fNbins;i++) ((TAxis&)obj).SetBinLabel(i,this->GetBinLabel(i));
   }
}

//______________________________________________________________________________
Int_t TAxis::DistancetoPrimitive(Int_t, Int_t)
{
   // Compute distance from point px,py to an axis

   return 9999;
}

//______________________________________________________________________________
void TAxis::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Execute action corresponding to one event
   //
   //  This member function is called when an axis is clicked with the locator
   //
   //  The axis range is set between the position where the mouse is pressed
   //  and the position where it is released.
   //  If the mouse position is outside the current axis range when it is released
   //  the axis is unzoomed with the corresponding proportions.
   //  Note that the mouse does not need to be in the pad or even canvas
   //  when it is released.

   if (!gPad) return;
   gPad->ExecuteEventAxis(event,px,py,this);
}

//______________________________________________________________________________
Int_t TAxis::FindBin(Double_t x)
{
   // Find bin number corresponding to abscissa x
   //
   // If x is underflow or overflow, attempt to rebin histogram
   // if the TH1::kCanRebin bit is set otherwise return 0 or fNbins+1

   Int_t bin;
   if (x < fXmin) {              //*-* underflow
      bin = 0;
      if (fParent == 0) return bin;
      if (!fParent->TestBit(TH1::kCanRebin)) return bin;
      ((TH1*)fParent)->RebinAxis(x,this);
      return FindFixBin(x);
   } else  if ( !(x < fXmax)) {     //*-* overflow  (note the way to catch NaN
      bin = fNbins+1;
      if (fParent == 0) return bin;
      if (!fParent->TestBit(TH1::kCanRebin)) return bin;
      ((TH1*)fParent)->RebinAxis(x,this);
      return FindFixBin(x);
   } else {
      if (!fXbins.fN) {        //*-* fix bins
         bin = 1 + int (fNbins*(x-fXmin)/(fXmax-fXmin) );
      } else {                  //*-* variable bin sizes
         //for (bin =1; x >= fXbins.fArray[bin]; bin++);
         bin = 1 + TMath::BinarySearch(fXbins.fN,fXbins.fArray,x);
      }
   }
   return bin;
}

//______________________________________________________________________________
Int_t TAxis::FindBin(const char *label)
{
   // Find bin number with label.
   // If the List of labels does not exist create it
   // If label is not in the list of labels do the following depending on the 
   // bit TH1::kCanRebin of the parent histogram. 
   //   - if the bit is set add the new label and if the number of labels exceeds 
   //      the number of bins, double the number of bins via TH1::LabelsInflate 
   //   - if the bit is not set return 0 (underflow bin) 
   //
   // -1 is returned only when the Axis has no parent histogram

   //create list of labels if it does not exist yet
   if (!fLabels) {
      if (!fParent) return -1;
      fLabels = new THashList(1,1);
      fParent->SetBit(TH1::kCanRebin);
      if (fXmax <= fXmin) { 
         //L.M. Dec 2010 in case of no min and max specified use 0 ->NBINS 
         fXmin = 0; 
         fXmax = fNbins;
      }
   }

   // search for label in the existing list
   TObjString *obj = (TObjString*)fLabels->FindObject(label);
   if (obj) return (Int_t)obj->GetUniqueID();

   //Not yet in the list. Can we rebin the histogram ?
   // if we cannot re-bin put label in the underflow bins
   if (!fParent->TestBit(TH1::kCanRebin)) { 
      if (gDebug>0) 
         Info("FindBin","Label %s is not in the list and the axis cannot be rebinned - the entry will be added in the underflow bin",label);
      return 0;
   }

   Int_t n = fLabels->GetEntries();
   TH1 *h = (TH1*)fParent;

   //may be we have to resize the histogram (doubling number of channels)
   if (n >= fNbins) h->LabelsInflate(GetName());

   //add new label to the list: assign bin number
   obj = new TObjString(label);
   fLabels->Add(obj);
   obj->SetUniqueID(n+1);
   return n+1;
}

//______________________________________________________________________________
Int_t TAxis::FindFixBin(Double_t x) const
{
   // Find bin number corresponding to abscissa x

   Int_t bin;
   if (x < fXmin) {              //*-* underflow
      bin = 0;
   } else  if ( !(x < fXmax)) {     //*-* overflow  (note the way to catch NaN
      bin = fNbins+1;
   } else {
      if (!fXbins.fN) {        //*-* fix bins
         bin = 1 + int (fNbins*(x-fXmin)/(fXmax-fXmin) );
      } else {                  //*-* variable bin sizes
//         for (bin =1; x >= fXbins.fArray[bin]; bin++);
         bin = 1 + TMath::BinarySearch(fXbins.fN,fXbins.fArray,x);
      }
   }
   return bin;
}

//______________________________________________________________________________
const char *TAxis::GetBinLabel(Int_t bin) const
{
   // Return label for bin

   if (!fLabels) return "";
   if (bin <= 0 || bin > fNbins) return "";
   TIter next(fLabels);
   TObjString *obj;
   while ((obj=(TObjString*)next())) {
      Int_t binid = (Int_t)obj->GetUniqueID();
      if (binid == bin) return obj->GetName();
   }
   return "";
}

//______________________________________________________________________________
Int_t TAxis::GetFirst() const
{
   //             return first bin on the axis
   //       ie 1 if no range defined
   //       NOTE: in some cases a zero is returned (see TAxis::SetRange)

   if (!TestBit(kAxisRange)) return 1;
   return fFirst;
}

//______________________________________________________________________________
Int_t TAxis::GetLast() const
{
   //             return last bin on the axis
   //       ie fNbins if no range defined
   //       NOTE: in some cases a zero is returned (see TAxis::SetRange)

   if (!TestBit(kAxisRange)) return fNbins;
   return fLast;
}

//______________________________________________________________________________
Double_t TAxis::GetBinCenter(Int_t bin) const
{
   // Return center of bin

   Double_t binwidth;
   if (!fXbins.fN || bin<1 || bin>fNbins) {
      binwidth = (fXmax - fXmin) / Double_t(fNbins);
      return fXmin + (bin-1) * binwidth + 0.5*binwidth;
   } else {
      binwidth = fXbins.fArray[bin] - fXbins.fArray[bin-1];
      return fXbins.fArray[bin-1] + 0.5*binwidth;
   }
}

//______________________________________________________________________________
Double_t TAxis::GetBinCenterLog(Int_t bin) const
{
   // Return center of bin in log
   // With a log-equidistant binning for a bin with low and up edges, the mean is : 
   // 0.5*(ln low + ln up) i.e. sqrt(low*up) in logx (e.g. sqrt(10^0*10^2) = 10). 
   //Imagine a bin with low=1 and up=100 : 
   // - the center in lin is (100-1)/2=50.5 
   // - the center in log would be sqrt(1*100)=10 (!=log(50.5)) 
   // NB: if the low edge of the bin is negative, the function returns the bin center
   //     as computed by TAxis::GetBinCenter
   
   Double_t low,up;
   if (!fXbins.fN || bin<1 || bin>fNbins) {
      Double_t binwidth = (fXmax - fXmin) / Double_t(fNbins);
      low = fXmin + (bin-1) * binwidth;
      up  = low+binwidth;
   } else {
      low = fXbins.fArray[bin-1];
      up  = fXbins.fArray[bin];
   }
   if (low <=0 ) return GetBinCenter(bin);
   return TMath::Sqrt(low*up);
}
//______________________________________________________________________________
Double_t TAxis::GetBinLowEdge(Int_t bin) const
{
   // Return low edge of bin

   if (fXbins.fN && bin > 0 && bin <=fNbins) return fXbins.fArray[bin-1];
   Double_t binwidth = (fXmax - fXmin) / Double_t(fNbins);
   return fXmin + (bin-1) * binwidth;
}

//______________________________________________________________________________
Double_t TAxis::GetBinUpEdge(Int_t bin) const
{
   // Return up edge of bin

   Double_t binwidth;
   if (!fXbins.fN || bin<1 || bin>fNbins) {
      binwidth = (fXmax - fXmin) / Double_t(fNbins);
      return fXmin + bin*binwidth;
   } else {
      binwidth = fXbins.fArray[bin] - fXbins.fArray[bin-1];
      return fXbins.fArray[bin-1] + binwidth;
   }
}

//______________________________________________________________________________
Double_t TAxis::GetBinWidth(Int_t bin) const
{
   // Return bin width

   if (fNbins <= 0) return 0;
   if (fXbins.fN <= 0)  return (fXmax - fXmin) / Double_t(fNbins);
   if (bin >fNbins) bin = fNbins;
   if (bin <1 ) bin = 1;
   return fXbins.fArray[bin] - fXbins.fArray[bin-1];
}


//______________________________________________________________________________
void TAxis::GetCenter(Double_t *center) const
{
   // Return an array with the center of all bins

   Int_t bin;
   for (bin=1; bin<=fNbins; bin++) *(center + bin-1) = GetBinCenter(bin);
}

//______________________________________________________________________________
void TAxis::GetLowEdge(Double_t *edge) const
{
   // Return an array with the lod edge of all bins

   Int_t bin;
   for (bin=1; bin<=fNbins; bin++) *(edge + bin-1) = GetBinLowEdge(bin);
}

//______________________________________________________________________________
const char *TAxis::GetTimeFormatOnly() const
{
   // Return *only* the time format from the string fTimeFormat

   static TString timeformat;
   Int_t idF = fTimeFormat.Index("%F");
   if (idF>=0) {
      timeformat = fTimeFormat(0,idF);
   } else {
      timeformat = fTimeFormat;
   }
   return timeformat.Data();
}

//______________________________________________________________________________
const char *TAxis::GetTicks() const
{
   // Return the ticks option (see SetTicks)

   if (TestBit(kTickPlus) && TestBit(kTickMinus)) return "+-";
   if (TestBit(kTickMinus)) return "-";
   return "+";
}

//______________________________________________________________________________
void TAxis::LabelsOption(Option_t *option)
{
   //  Set option(s) to draw axis with labels
   //  option = "a" sort by alphabetic order
   //         = ">" sort by decreasing values
   //         = "<" sort by increasing values
   //         = "h" draw labels horizonthal
   //         = "v" draw labels vertical
   //         = "u" draw labels up (end of label right adjusted)
   //         = "d" draw labels down (start of label left adjusted)

   if (!fLabels) {
      Warning("Sort","Cannot sort. No labels");
      return;
   }
   TH1 *h = (TH1*)GetParent();
   if (!h) {
      Error("Sort","Axis has no parent");
      return;
   }

   h->LabelsOption(option,GetName());
}

//___________________________________________________________________________
Bool_t TAxis::GetRotateTitle() const
{
   // Return kTRUE if kRotateTitle bit is set, kFALSE otherwise.

   return TestBit(kRotateTitle) ? kTRUE : kFALSE;
}

//______________________________________________________________________________
void TAxis::ImportAttributes(const TAxis *axis)
{
   // Copy axis attributes to this 

   SetTitle(axis->GetTitle());
   SetNdivisions(axis->GetNdivisions());
   SetAxisColor(axis->GetAxisColor());
   SetLabelColor(axis->GetLabelColor());
   SetLabelFont(axis->GetLabelFont());
   SetLabelOffset(axis->GetLabelOffset());
   SetLabelSize(axis->GetLabelSize());
   SetTickLength(axis->GetTickLength());
   SetTitleOffset(axis->GetTitleOffset());
   SetTitleSize(axis->GetTitleSize());
   SetTitleColor(axis->GetTitleColor());
   SetTitleFont(axis->GetTitleFont());
   SetBit(TAxis::kCenterTitle,   axis->TestBit(TAxis::kCenterTitle));
   SetBit(TAxis::kCenterLabels,  axis->TestBit(TAxis::kCenterLabels));
   SetBit(TAxis::kRotateTitle,   axis->TestBit(TAxis::kRotateTitle));
   SetBit(TAxis::kNoExponent,    axis->TestBit(TAxis::kNoExponent));
   SetBit(TAxis::kTickPlus,      axis->TestBit(TAxis::kTickPlus));
   SetBit(TAxis::kTickMinus,     axis->TestBit(TAxis::kTickMinus));
   SetBit(TAxis::kMoreLogLabels, axis->TestBit(TAxis::kMoreLogLabels));
   if (axis->GetDecimals())      SetBit(TAxis::kDecimals); //the bit is in TAxis::fAxis2   
   SetTimeFormat(axis->GetTimeFormat());
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
void TAxis::SaveAttributes(ostream &out, const char *name, const char *subname)
{
    // Save axis attributes as C++ statement(s) on output stream out

   char quote = '"';
   if (strlen(GetTitle())) {
      out<<"   "<<name<<subname<<"->SetTitle("<<quote<<GetTitle()<<quote<<");"<<endl;
   }
   if (fTimeDisplay) {
      out<<"   "<<name<<subname<<"->SetTimeDisplay(1);"<<endl;
      out<<"   "<<name<<subname<<"->SetTimeFormat("<<quote<<GetTimeFormat()<<quote<<");"<<endl;
   }
   if (fLabels) {
      TIter next(fLabels);
      TObjString *obj;
      while ((obj=(TObjString*)next())) {
         out<<"   "<<name<<subname<<"->SetBinLabel("<<obj->GetUniqueID()<<","<<quote<<obj->GetName()<<quote<<");"<<endl;
      }
   }

   if (fFirst || fLast) {
      out<<"   "<<name<<subname<<"->SetRange("<<fFirst<<","<<fLast<<");"<<endl;
   }

   if (TestBit(kLabelsHori)) {
      out<<"   "<<name<<subname<<"->SetBit(TAxis::kLabelsHori);"<<endl;
   }

   if (TestBit(kLabelsVert)) {
      out<<"   "<<name<<subname<<"->SetBit(TAxis::kLabelsVert);"<<endl;
   }

   if (TestBit(kLabelsDown)) {
      out<<"   "<<name<<subname<<"->SetBit(TAxis::kLabelsDown);"<<endl;
   }

   if (TestBit(kLabelsUp)) {
      out<<"   "<<name<<subname<<"->SetBit(TAxis::kLabelsUp);"<<endl;
   }

   if (TestBit(kCenterTitle)) {
      out<<"   "<<name<<subname<<"->CenterTitle(true);"<<endl;
   }

   if (TestBit(kRotateTitle)) {
      out<<"   "<<name<<subname<<"->RotateTitle(true);"<<endl;
   }

   if (TestBit(kMoreLogLabels)) {
      out<<"   "<<name<<subname<<"->SetMoreLogLabels();"<<endl;
   }

   if (TestBit(kNoExponent)) {
      out<<"   "<<name<<subname<<"->SetNoExponent();"<<endl;
   }

   TAttAxis::SaveAttributes(out,name,subname);
}

//______________________________________________________________________________
void TAxis::Set(Int_t nbins, Double_t xlow, Double_t xup)
{
   // Initialize axis with fix bins

   fNbins   = nbins;
   fXmin    = xlow;
   fXmax    = xup;
   if (!fParent) SetDefaults();
   if (fXbins.fN > 0) fXbins.Set(0);
}

//______________________________________________________________________________
void TAxis::Set(Int_t nbins, const Float_t *xbins)
{
   // Initialize axis with variable bins

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
   if (!fParent) SetDefaults();
}

//______________________________________________________________________________
void TAxis::Set(Int_t nbins, const Double_t *xbins)
{
   // Initialize axis with variable bins

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
   if (!fParent) SetDefaults();
}

//______________________________________________________________________________
void TAxis::SetDefaults()
{
   // Set axis default values (from TStyle)
   
   fFirst   = 0;
   fLast    = 0;
   fBits2   = 0;
   char name[2];
   strlcpy(name,GetName(),2);
   name[1] = 0;
   TAttAxis::ResetAttAxis(name);
   fTimeDisplay = 0;
   SetTimeFormat();
}

//______________________________________________________________________________
Bool_t TAxis::GetDecimals() const
{
   // Returns kTRUE if kDecimals bit is set, kFALSE otherwise.
   // see TAxis::SetDecimals
   
   if ((fBits2 & kDecimals) != 0) return kTRUE;
   else                           return kFALSE;
}


//______________________________________________________________________________
void TAxis::SetDecimals(Bool_t dot)
{
   // Set the Decimals flag
   // By default, blank characters are stripped, and then the
   // label is correctly aligned. The dot, if last character of the string, 
   // is also stripped, unless this option is specified.
   // One can disable the option by calling axis.SetDecimals(kTRUE).
   // The flag (in fBits2) is passed to the drawing function TGaxis::PaintAxis

   if (dot) fBits2 |=  kDecimals;
   else     fBits2 &= ~kDecimals;
}

//______________________________________________________________________________
void TAxis::SetBinLabel(Int_t bin, const char *label)
{
   // Set label for bin
   // In this case we create a label list in the axis but we do not 
   // set the kCanRebin bit. 
   // New labels will not be added with the Fill method but will end-up in the 
   // underflow bin. See documentation of TAxis::FindBin(const char*)

   if (!fLabels) fLabels = new THashList(fNbins,3);

   if (bin <= 0 || bin > fNbins) {
      Error("SetBinLabel","Illegal bin number: %d",bin);
      return;
   }

   // Check whether this bin already has a label.
   TIter next(fLabels);
   TObjString *obj;
   while ((obj=(TObjString*)next())) {
      if ( obj->GetUniqueID()==(UInt_t)bin ) {
         // It does. Overwrite it.
         obj->SetString(label);
         return;
      }
   }
   // It doesn't. Add this new label.
   obj = new TObjString(label);
   fLabels->Add(obj);
   obj->SetUniqueID((UInt_t)bin);
}

//______________________________________________________________________________
void TAxis::SetLimits(Double_t xmin, Double_t xmax)
{
   //          Set the axis limits

   fXmin = xmin;
   fXmax = xmax;
}

//______________________________________________________________________________
Bool_t TAxis::GetMoreLogLabels() const
{
   // Return kTRUE if kMoreLogLabels bit is set, kFALSE otherwise.

   return TestBit(kMoreLogLabels) ? kTRUE : kFALSE;
}

//______________________________________________________________________________
void TAxis::SetMoreLogLabels(Bool_t more)
{
   // Set the kMoreLogLabels bit flag
   // When this option is selected more labels are drawn when in log scale
   // and there is a small number of decades  (<3).
   // The flag (in fBits) is passed to the drawing function TGaxis::PaintAxis

   if (more) SetBit(kMoreLogLabels);
   else      ResetBit(kMoreLogLabels);
}

//______________________________________________________________________________
Bool_t TAxis::GetNoExponent() const
{
   // Returns kTRUE if kNoExponent bit is set, kFALSE otherwise.
   // see TAxis::SetNoExponent

   return TestBit(kNoExponent) ? kTRUE : kFALSE;
}

//______________________________________________________________________________
void TAxis::SetNoExponent(Bool_t noExponent)
{
   // Set the NoExponent flag
   // By default, an exponent of the form 10^N is used when the label values
   // are either all very small or very large.
   // One can disable the exponent by calling axis.SetNoExponent(kTRUE).
   // The flag (in fBits) is passed to the drawing function TGaxis::PaintAxis

   if (noExponent) SetBit(kNoExponent);
   else            ResetBit(kNoExponent);
}

//______________________________________________________________________________
void TAxis::SetRange(Int_t first, Int_t last)
{
   //  Set the viewing range for the axis from bin first to last
   //  To set a range using the axis coordinates, use TAxis::SetRangeUser.
   //  if first<=1 and last>=Nbins or if last < first the range is reset by removing the  
   //  bit TAxis::kAxisRange. In this case the functions TAxis::GetFirst() and TAxis::GetLast() 
   //  will return 1 and Nbins. 
   //  NOTE: If the bit has been set manually by the user in case of no range defined
   //         GetFirst() and GetLast() will return 0. 

   if (last <= 0) last = fNbins;
   if (last > fNbins) last = fNbins;
   if (last  < first) { first = 1; last = fNbins; }
   if (first < 1)     first = 1;
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
void TAxis::SetRangeUser(Double_t ufirst, Double_t ulast)
{
   //  Set the viewing range for the axis from ufirst to ulast (in user coordinates)
   //  To set a range using the axis bin numbers, use TAxis::SetRange.

   if (!strstr(GetName(),"xaxis")) {
      TH1 *hobj = (TH1*)GetParent();
      if (hobj &&
          ((hobj->GetDimension() == 2 && strstr(GetName(),"zaxis")) 
           || (hobj->GetDimension() == 1 && strstr(GetName(),"yaxis")))) {
         hobj->SetMinimum(ufirst);
         hobj->SetMaximum(ulast);
         return;
      }
   }
   SetRange(FindFixBin(ufirst),FindFixBin(ulast));
}

//______________________________________________________________________________
void TAxis::SetTicks(Option_t *option)
{
   //  set ticks orientation
   //  option = "+"  ticks drawn on the "positive side" (default)
   //  option = "-"  ticks drawn on the "negative side"
   //  option = "+-" ticks drawn on both sides

   ResetBit(kTickPlus);
   ResetBit(kTickMinus);
   if (strchr(option,'+')) SetBit(kTickPlus);
   if (strchr(option,'-')) SetBit(kTickMinus);
}

//______________________________________________________________________________
void TAxis::SetTimeFormat(const char *tformat)
{
   // Change the format used for time plotting
   //
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
   //
   //    This function allows also to define the time offset. It is done via %F
   //    which should be appended at the end of the format string. The time
   //    offset has the following format: 'yyyy-mm-dd hh:mm:ss'
   //    Example:
   //
   //          h = new TH1F("Test","h",3000,0.,200000.);
   //          h->GetXaxis()->SetTimeDisplay(1);
   //          h->GetXaxis()->SetTimeFormat("%d\/%m\/%y%F2000-02-28 13:00:01");
   //
   //    This defines the time format being "dd/mm/yy" and the time offset as the
   //    February 28th 2003 at 13:00:01
   //
   //    If %F is not specified, the time offset used will be the one defined by:
   //    gStyle->SetTimeOffset. For example like that:
   //
   //          TDatime da(2003,02,28,12,00,00);
   //          gStyle->SetTimeOffset(da.Convert());

   TString timeformat = tformat;

   if (timeformat.Index("%F")>=0 || timeformat.IsNull()) {
      fTimeFormat = timeformat;
      return;
   }

   Int_t idF = fTimeFormat.Index("%F");
   if (idF>=0) {
      Int_t lnF = fTimeFormat.Length();
      TString stringtimeoffset = fTimeFormat(idF,lnF);
      fTimeFormat = tformat;
      fTimeFormat.Append(stringtimeoffset);
   } else {
      fTimeFormat = tformat;
      SetTimeOffset(gStyle->GetTimeOffset());
   }
}


//______________________________________________________________________________
void TAxis::SetTimeOffset(Double_t toffset, Option_t *option)
{
   // Change the time offset
   // If option = "gmt" the time offset is treated as a GMT time.

   TString opt = option;
   opt.ToLower();

   Bool_t gmt = kFALSE;
   if (opt.Contains("gmt")) gmt = kTRUE;

   char tmp[20];
   time_t timeoff;
   struct tm* utctis;
   Int_t idF = fTimeFormat.Index("%F");
   if (idF>=0) fTimeFormat.Remove(idF);
   fTimeFormat.Append("%F");

   timeoff = (time_t)((Long_t)(toffset));
   utctis = gmtime(&timeoff);

   strftime(tmp,20,"%Y-%m-%d %H:%M:%S",utctis);
   fTimeFormat.Append(tmp);

   // append the decimal part of the time offset
   Double_t ds = toffset-(Int_t)toffset;
   if(ds!= 0) {
      snprintf(tmp,20,"s%g",ds);
      fTimeFormat.Append(tmp);
   }

   // If the time is GMT, stamp fTimeFormat
   if (gmt) fTimeFormat.Append(" GMT");
}


//______________________________________________________________________________
void TAxis::Streamer(TBuffer &R__b)
{
   // Stream an object of class TAxis.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 5) {
         R__b.ReadClassBuffer(TAxis::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
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
      //====end of old versions

   } else {
      R__b.WriteClassBuffer(TAxis::Class(),this);
   }
}

//______________________________________________________________________________
void TAxis::UnZoom()
{
   // Reset first & last bin to the full range


   gPad->SetView();

   //unzoom object owning this axis
   SetRange(0,0);
   TH1 *hobj1 = (TH1*)GetParent();
   if (!strstr(GetName(),"xaxis")) {
      if (!hobj1) return;
      if (hobj1->GetDimension() == 2) {
         if (strstr(GetName(),"zaxis")) {
            hobj1->SetMinimum();
            hobj1->SetMaximum();
            hobj1->ResetBit(TH1::kIsZoomed);
            return;
         }
      }
      if (strcmp(hobj1->GetName(),"hframe") == 0 ) {
         hobj1->SetMinimum(fXmin);
         hobj1->SetMaximum(fXmax);
      } else {
         hobj1->SetMinimum();
         hobj1->SetMaximum();
         hobj1->ResetBit(TH1::kIsZoomed);
      }
   }
   //must unzoom all histograms in the pad
   TIter next(gPad->GetListOfPrimitives());
   TObject *obj;
   while ((obj= next())) {
      if (!obj->InheritsFrom(TH1::Class())) continue;
      TH1 *hobj = (TH1*)obj;
      if (hobj == hobj1) continue;
      if (!strstr(GetName(),"xaxis")) {
         if (hobj->GetDimension() == 2) {
            if (strstr(GetName(),"zaxis")) {
               hobj->SetMinimum();
               hobj->SetMaximum();
               hobj->ResetBit(TH1::kIsZoomed);
            } else {
               hobj->GetYaxis()->SetRange(0,0);
            }
            return;
         }
         if (strcmp(hobj->GetName(),"hframe") == 0 ) {
            hobj->SetMinimum(fXmin);
            hobj->SetMaximum(fXmax);
         } else {
            hobj->SetMinimum();
            hobj->SetMaximum();
            hobj->ResetBit(TH1::kIsZoomed);
         }
      } else {
         hobj->GetXaxis()->SetRange(0,0);
      }
   }
}

//______________________________________________________________________________
void TAxis::ZoomOut(Double_t factor, Double_t offset)
{
   // Zoom out by a factor of 'factor' (default =2)
   //   uses previous zoom factor by default
   // Keep center defined by 'offset' fixed
   //   ie. -1 at left of current range, 0 in center, +1 at right
   
   if (factor <= 0) factor = 2;
   Double_t center = (GetFirst()*(1-offset) + GetLast()*(1+offset))/2.;
   Int_t first = int(TMath::Floor(center+(GetFirst()-center)*factor + 0.4999999));
   Int_t last  = int(TMath::Floor(center+(GetLast() -center)*factor + 0.5000001));
   if (first==GetFirst() && last==GetLast()) { first--; last++; }
   SetRange(first,last);
}
