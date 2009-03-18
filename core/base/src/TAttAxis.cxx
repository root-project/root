// @(#)root/base:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TAttAxis.h"
#include "TStyle.h"
#include "TVirtualPad.h"
#include "TColor.h"
#include "TClass.h"
#include "TMathBase.h"
#include <stdlib.h>

ClassImp(TAttAxis)


//______________________________________________________________________________
//
//  Manages histogram axis attributes
//

TAttAxis::TAttAxis()
{
   // Constructor.
   ResetAttAxis();
}


//______________________________________________________________________________
TAttAxis::~TAttAxis()
{
   // Destructor.
}


//______________________________________________________________________________
void TAttAxis::Copy(TAttAxis &attaxis) const
{
   // Copy of the object.

   attaxis.fNdivisions  = fNdivisions;
   attaxis.fAxisColor   = fAxisColor;
   attaxis.fLabelColor  = fLabelColor;
   attaxis.fLabelFont   = fLabelFont;
   attaxis.fLabelOffset = fLabelOffset;
   attaxis.fLabelSize   = fLabelSize;
   attaxis.fTickLength  = fTickLength;
   attaxis.fTitleOffset = fTitleOffset;
   attaxis.fTitleSize   = fTitleSize;
   attaxis.fTitleColor  = fTitleColor;
   attaxis.fTitleFont   = fTitleFont;
}


//______________________________________________________________________________
void TAttAxis::ResetAttAxis(Option_t *option)
{
   // Reset axis attributes

   if (gStyle) {
      fNdivisions  = gStyle->GetNdivisions(option);
      fAxisColor   = gStyle->GetAxisColor(option);
      fLabelColor  = gStyle->GetLabelColor(option);
      fLabelFont   = gStyle->GetLabelFont(option);
      fLabelOffset = gStyle->GetLabelOffset(option);
      fLabelSize   = gStyle->GetLabelSize(option);
      fTickLength  = gStyle->GetTickLength(option);
      fTitleOffset = gStyle->GetTitleOffset(option);
      fTitleSize   = gStyle->GetTitleSize(option);
      fTitleColor  = gStyle->GetTitleColor(option);
      fTitleFont   = gStyle->GetTitleFont(option);
   } else {
      fNdivisions  = 510;
      fAxisColor   = 1;
      fLabelColor  = 1;
      fLabelFont   = 62;
      fLabelOffset = 0.005;
      fLabelSize   = 0.04;
      fTickLength  = 0.03;
      fTitleOffset = 1;
      fTitleSize   = fLabelSize;
      fTitleColor  = 1;
      fTitleFont   = 62;
   }
}


//______________________________________________________________________________
void TAttAxis::SaveAttributes(ostream &out, const char *name, const char *subname)
{
    // Save axis attributes as C++ statement(s) on output stream out

   if (fNdivisions != 510) {
      out<<"   "<<name<<subname<<"->SetNdivisions("<<fNdivisions<<");"<<endl;
   }
   if (fAxisColor != 1) {
      if (fAxisColor > 228) {
         TColor::SaveColor(out, fAxisColor);
         out<<"   "<<name<<subname<<"->SetAxisColor(ci);" << endl;
      } else
         out<<"   "<<name<<subname<<"->SetAxisColor("<<fAxisColor<<");"<<endl;
   }
   if (fLabelColor != 1) {
      if (fLabelColor > 228) {
         TColor::SaveColor(out, fLabelColor);
         out<<"   "<<name<<subname<<"->SetLabelColor(ci);" << endl;
      } else
         out<<"   "<<name<<subname<<"->SetLabelColor("<<fLabelColor<<");"<<endl;
   }
   if (fLabelFont != 62) {
      out<<"   "<<name<<subname<<"->SetLabelFont("<<fLabelFont<<");"<<endl;
   }
   if (TMath::Abs(fLabelOffset-0.005) > 0.0001) {
      out<<"   "<<name<<subname<<"->SetLabelOffset("<<fLabelOffset<<");"<<endl;
   }
   if (TMath::Abs(fLabelSize-0.04) > 0.001) {
      out<<"   "<<name<<subname<<"->SetLabelSize("<<fLabelSize<<");"<<endl;
   }
   if (TMath::Abs(fTitleSize-0.04) > 0.001) {
      out<<"   "<<name<<subname<<"->SetTitleSize("<<fTitleSize<<");"<<endl;
   }
   if (TMath::Abs(fTickLength-0.03) > 0.001) {
      out<<"   "<<name<<subname<<"->SetTickLength("<<fTickLength<<");"<<endl;
   }
   if (TMath::Abs(fTitleOffset-1) > 0.001) {
      out<<"   "<<name<<subname<<"->SetTitleOffset("<<fTitleOffset<<");"<<endl;
   }
   if (fTitleColor != 1) {
      if (fTitleColor > 228) {
         TColor::SaveColor(out, fTitleColor);
         out<<"   "<<name<<subname<<"->SetTitleColor(ci);" << endl;
      } else
         out<<"   "<<name<<subname<<"->SetTitleColor("<<fTitleColor<<");"<<endl;
   }
   if (fTitleFont != 62) {
      out<<"   "<<name<<subname<<"->SetTitleFont("<<fTitleFont<<");"<<endl;
   }
}


//______________________________________________________________________________
void TAttAxis::SetAxisColor(Color_t color)
{
   // Set color of the line axis and tick marks

   fAxisColor = color;
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TAttAxis::SetLabelColor(Color_t color)
{
   // Set color of labels

   fLabelColor = color;
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TAttAxis::SetLabelFont(Style_t font)
{
   // Set labels' font.

   fLabelFont = font;
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TAttAxis::SetLabelOffset(Float_t offset)
{
   // Set distance between the axis and the labels
   // The distance is expressed in per cent of the pad width

   fLabelOffset = offset;
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TAttAxis::SetLabelSize(Float_t size)
{
   // Set size of axis labels
   // The size is expressed in per cent of the pad width

   fLabelSize = size;
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TAttAxis::SetNdivisions(Int_t n, Bool_t optim)
{
   // Set the number of divisions for this axis.
   //
   // if optim = kTRUE (default), the number of divisions will be
   //                   optimized around the specified value.
   // if optim = kFALSE, or n < 0, the axis will be forced to use
   //                   exactly n divisions.
   //
   // n = n1 + 100*n2 + 10000*n3
   //
   // Where n1 is the number of primary divisions,
   // n2 is the number of second order divisions and
   // n3 is the number of third order divisions.
   //
   // e.g. 512 means 12 primary and 5 secondary divisions.
   //
   // If the number of divisions is "optimized" (see above) n1, n2, n3 are
   // maximum values.

   fNdivisions = n;
   if (!optim) fNdivisions = -abs(n);
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TAttAxis::SetNdivisions(Int_t n1, Int_t n2, Int_t n3, Bool_t optim)
{
   //see function above
   SetNdivisions(n1+100*n2+10000*n3, optim);
}


//______________________________________________________________________________
void TAttAxis::SetTickLength(Float_t length)
{
   // Set tick mark length
   // The length is expressed in per cent of the pad width

   fTickLength = length;
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TAttAxis::SetTitleOffset(Float_t offset)
{
   // Set distance between the axis and the axis title
   // Offset is a correction factor with respect to the "standard" value.
   // offset = 1  uses the default position that is computed in function
   // of the label offset and size.
   // offset = 1.2 will add 20 per cent more to the default offset.

   fTitleOffset = offset;
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TAttAxis::SetTitleSize(Float_t size)
{
   // Set size of axis title
   // The size is expressed in per cent of the pad width

   fTitleSize = size;
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TAttAxis::SetTitleColor(Color_t color)
{
   // Set color of axis title

   fTitleColor = color;
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TAttAxis::SetTitleFont(Style_t font)
{
   // Set the title font.

   fTitleFont = font;
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TAttAxis::Streamer(TBuffer &R__b)
{
   // Stream an object of class TAttAxis.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 3) {
         R__b.ReadClassBuffer(TAttAxis::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      R__b >> fNdivisions;
      R__b >> fAxisColor;
      R__b >> fLabelColor;
      R__b >> fLabelFont;
      R__b >> fLabelOffset;
      R__b >> fLabelSize;
      R__b >> fTickLength;
      R__b >> fTitleOffset;

      if (R__v > 1 && R__b.GetVersionOwner() > 900)
         R__b >> fTitleSize;
      else
         fTitleSize = fLabelSize;
      if (R__v > 2) {
         R__b >> fTitleColor;
         R__b >> fTitleFont;
      }
      //====end of old versions

   } else {
      R__b.WriteClassBuffer(TAttAxis::Class(),this);
   }
}
