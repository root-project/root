// @(#)root/gui:$Name:  $:$Id: TGGC.cxx,v 1.1 2000/09/29 08:52:52 rdm Exp $
// Author: Fons Rademakers   20/9/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGGC and TGGCPool                                                    //
//                                                                      //
// Encapsulate a graphics context used in the low level graphics.       //
// TGGCPool provides a pool of graphics contexts.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGGC.h"
#include "TVirtualX.h"
#include "TList.h"
#include <string.h>


#if 0
class TGGCElement : public TObject, public TRefCnt {
public:
   TGGC   *fContext;
   ~TGGCElement() { delete fContext; }
   Bool_t  IsEqual(TObject *obj) { return fContext == obj; }
};
#endif


ClassImp(TGGC)

//______________________________________________________________________________
TGGC::TGGC(GCValues_t *values)
{
   // Create a graphics context.

   if (values) {
      fValues = *values;
      fContext = gVirtualX->CreateGC(gVirtualX->GetDefaultRootWindow(), values);
      if (values->fMask & kGCDashList) {
         if (values->fDashLen > (Int_t)sizeof(fValues.fDashes))
            Warning("TGGC", "dash list can have only up to %d elements",
                    sizeof(fValues.fDashes));
         fValues.fDashLen = TMath::Min(values->fDashLen, (Int_t)sizeof(fValues.fDashes));
         gVirtualX->SetDashes(fContext, fValues.fDashOffset, fValues.fDashes,
                              fValues.fDashLen);
      }
   } else {
      memset(&fValues, 0, sizeof(GCValues_t));
      fContext = 0;
   }
   fDelete = kTRUE;
}

//______________________________________________________________________________
TGGC::TGGC(const TGGC &g) : TObject(g)
{
   // Copy a graphics context.

   fValues = g.fValues;
   if (g.fContext) {
      fContext = gVirtualX->CreateGC(gVirtualX->GetDefaultRootWindow(), &fValues);
      if (fValues.fMask & kGCDashList)
         gVirtualX->SetDashes(fContext, fValues.fDashOffset, fValues.fDashes,
                              fValues.fDashLen);
   } else
      fContext = 0;
   fDelete = kTRUE;
}

//______________________________________________________________________________
TGGC::~TGGC()
{
   // Delete graphics context.

   if (fContext && fDelete)
      gVirtualX->DeleteGC(fContext);
}

//______________________________________________________________________________
TGGC &TGGC::operator=(const TGGC &rhs)
{
   // Graphics context assignment operator. Use this operator to share
   // a graphics context. Using this operator you will not get a copy of the
   // context.

   if (this != &rhs) {
      if (fContext && fDelete)
         gVirtualX->DeleteGC(fContext);
      TObject::operator=(rhs);
      fValues  = rhs.fValues;
      fContext = rhs.fContext;
      if (fContext)
         fDelete = kFALSE;
      else
         fDelete = kTRUE;
   }
   return *this;
}

//______________________________________________________________________________
GContext_t TGGC::operator()() const
{
   // Not inline due to a bug in g++ 2.96 20000731 (Red Hat Linux 7.0)

   return fContext;
}

//______________________________________________________________________________
void TGGC::UpdateValues(GCValues_t *values)
{
   // Update values + mask.

   fValues.fMask |= values->fMask;

   for (Mask_t bit = 1; bit <= kGCArcMode; bit <<= 1) {
      switch (bit & values->fMask) {
         default:
         case 0:
            continue;
            break;
         case kGCFunction:
            fValues.fFunction = values->fFunction;
            break;
         case kGCPlaneMask:
            fValues.fPlaneMask = values->fPlaneMask;
            break;
         case kGCForeground:
            fValues.fForeground = values->fForeground;
            break;
         case kGCBackground:
            fValues.fBackground = values->fBackground;
            break;
         case kGCLineWidth:
            fValues.fLineWidth = values->fLineWidth;
            break;
         case kGCLineStyle:
            fValues.fLineStyle = values->fLineStyle;
            break;
         case kGCCapStyle:
            fValues.fCapStyle = values->fCapStyle;
            break;
         case kGCJoinStyle:
            fValues.fJoinStyle = values->fJoinStyle;
            break;
         case kGCFillStyle:
            fValues.fFillStyle = values->fFillStyle;
            break;
         case kGCFillRule:
            fValues.fFillRule = values->fFillRule;
            break;
         case kGCTile:
            fValues.fTile = values->fTile;
            break;
         case kGCStipple:
            fValues.fStipple = values->fStipple;
            break;
         case kGCTileStipXOrigin:
            fValues.fTsXOrigin = values->fTsXOrigin;
            break;
         case kGCTileStipYOrigin:
            fValues.fTsYOrigin = values->fTsYOrigin;
            break;
         case kGCFont:
            fValues.fFont = values->fFont;
            break;
         case kGCSubwindowMode:
            fValues.fSubwindowMode = values->fSubwindowMode;
            break;
         case kGCGraphicsExposures:
            fValues.fGraphicsExposures = values->fGraphicsExposures;
            break;
         case kGCClipXOrigin:
            fValues.fClipXOrigin = values->fClipXOrigin;
            break;
         case kGCClipYOrigin:
            fValues.fClipYOrigin = values->fClipYOrigin;
            break;
         case kGCClipMask:
            fValues.fClipMask = values->fClipMask;
            break;
         case kGCDashOffset:
            fValues.fDashOffset = values->fDashOffset;
            break;
         case kGCDashList:
            if (values->fDashLen > (Int_t)sizeof(fValues.fDashes))
               Warning("UpdateValues", "dash list can have only up to %d elements",
                       sizeof(fValues.fDashes));
            fValues.fDashLen = TMath::Min(values->fDashLen, (Int_t)sizeof(fValues.fDashes));
            memcpy(fValues.fDashes, values->fDashes, fValues.fDashLen);
            break;
         case kGCArcMode:
            fValues.fArcMode = values->fArcMode;
            break;
      }
   }
}

//______________________________________________________________________________
void TGGC::SetAttributes(GCValues_t *values)
{
   // Set attributes as specified in the values structure.

   if (fContext)
      gVirtualX->ChangeGC(fContext, values);
   else
      fContext = gVirtualX->CreateGC(gVirtualX->GetDefaultRootWindow(), values);
   UpdateValues(values);
   if (values->fMask & kGCDashList)
      gVirtualX->SetDashes(fContext, fValues.fDashOffset, fValues.fDashes,
                           fValues.fDashLen);
}

//______________________________________________________________________________
void TGGC::SetFunction(EGraphicsFunction v)
{
   // Set graphics context drawing function.

   GCValues_t values;
   values.fFunction = v;
   values.fMask     = kGCFunction;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetPlaneMask(ULong_t v)
{
   // Set plane mask.

   GCValues_t values;
   values.fPlaneMask = v;
   values.fMask      = kGCPlaneMask;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetForeground(ULong_t v)
{
   // Set foreground color.

   GCValues_t values;
   values.fForeground = v;
   values.fMask       = kGCForeground;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetBackground(ULong_t v)
{
   // Set background color.

   GCValues_t values;
   values.fBackground = v;
   values.fMask       = kGCBackground;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetLineWidth(Int_t v)
{
   // Set line width.

   GCValues_t values;
   values.fLineWidth = v;
   values.fMask      = kGCLineWidth;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetLineStyle(Int_t v)
{
   // Set line style (kLineSolid, kLineOnOffDash, kLineDoubleDash).

   GCValues_t values;
   values.fLineStyle = v;
   values.fMask      = kGCLineStyle;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetCapStyle(Int_t v)
{
   // Set cap style (kCapNotLast, kCapButt, kCapRound, kCapProjecting).

   GCValues_t values;
   values.fCapStyle = v;
   values.fMask     = kGCCapStyle;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetJoinStyle(Int_t v)
{
   // Set line join style (kJoinMiter, kJoinRound, kJoinBevel).

   GCValues_t values;
   values.fJoinStyle = v;
   values.fMask      = kGCJoinStyle;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetFillStyle(Int_t v)
{
   // Set fill style (kFillSolid, kFillTiled, kFillStippled,
   // kFillOpaeueStippled).

   GCValues_t values;
   values.fFillStyle = v;
   values.fMask      = kGCFillStyle;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetFillRule(Int_t v)
{
   // Set fill rule (kEvenOddRule, kWindingRule).

   GCValues_t values;
   values.fFillRule = v;
   values.fMask     = kGCFillRule;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetTile(Pixmap_t v)
{
   // Set tile pixmap for tiling operations.

   GCValues_t values;
   values.fTile = v;
   values.fMask = kGCTile;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetStipple(Pixmap_t v)
{
   // Set 1 plane pixmap for stippling.

   GCValues_t values;
   values.fStipple = v;
   values.fMask    = kGCStipple;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetTileStipXOrigin(Int_t v)
{
   // X offset for tile or stipple operations.

   GCValues_t values;
   values.fTsXOrigin = v;
   values.fMask      = kGCTileStipXOrigin;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetTileStipYOrigin(Int_t v)
{
   // Y offset for tile or stipple operations.

   GCValues_t values;
   values.fTsYOrigin = v;
   values.fMask      = kGCTileStipYOrigin;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetFont(FontH_t v)
{
   // Set font.

   GCValues_t values;
   values.fFont = v;
   values.fMask = kGCFont;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetSubwindowMode(Int_t v)
{
   // Set sub window mode (kClipByChildren, kIncludeInferiors).

   GCValues_t values;
   values.fSubwindowMode = v;
   values.fMask          = kGCSubwindowMode;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetGraphicsExposures(Bool_t v)
{
   // True if graphics exposure should be generated.

   GCValues_t values;
   values.fGraphicsExposures = v;
   values.fMask              = kGCGraphicsExposures;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetClipXOrigin(Int_t v)
{
   // X origin for clipping.

   GCValues_t values;
   values.fClipXOrigin = v;
   values.fMask        = kGCClipXOrigin;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetClipYOrigin(Int_t v)
{
   // Y origin for clipping.

   GCValues_t values;
   values.fClipYOrigin = v;
   values.fMask        = kGCClipYOrigin;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetClipMask(Pixmap_t v)
{
   // Bitmap for clipping.

   GCValues_t values;
   values.fClipMask = v;
   values.fMask     = kGCClipMask;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetDashOffset(Int_t v)
{
   // Patterned/dashed line offset.

   GCValues_t values;
   values.fDashOffset = v;
   values.fMask       = kGCDashOffset;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetDashList(char v[], Int_t len)
{
   // Set dash pattern. First use SetDashOffset() if not 0.

   GCValues_t values;
   if (len > (Int_t)sizeof(values.fDashes))
      Warning("SetDashList", "dash list can have only up to %d elements",
              sizeof(values.fDashes));
   values.fDashLen = TMath::Min(len, (Int_t)sizeof(values.fDashes));
   memcpy(values.fDashes, v, values.fDashLen);
   values.fMask    = kGCDashList;
   SetAttributes(&values);
}

//______________________________________________________________________________
void TGGC::SetArcMode(Int_t v)
{
   // Set arc mode (kArcChord, kArcPieSlice).

   GCValues_t values;
   values.fArcMode = v;
   values.fMask    = kGCArcMode;
   SetAttributes(&values);
}


ClassImp(TGGCPool)

//______________________________________________________________________________
TGGCPool::TGGCPool(TGClient *client)
{
   // Create graphics context pool.

   fClient = client;
   fList   = new TList;
   fList->SetOwner();
}

//______________________________________________________________________________
TGGCPool::~TGGCPool()
{
   // Delete graphics context pool.

   delete fList;
}

//______________________________________________________________________________
void TGGCPool::FreeGC(TGGC *gc)
{
   // Delete graphics context if it not used anymore.

   TGGCElement *el = (TGGCElement *) fList->FindObject(gc);

   if (el) {
      el->RemoveReference();
      if (!el->References()) {
         fList->Remove(gc);
         delete el;
      }
   }
}

//______________________________________________________________________________
TGGC *TGGCPool::GetGC(GCValues_t *values)
{
   // Get the best matching graphics context depending on values.

   TGGCElement *el, *best_match = 0;
   Int_t matching_bits, best_matching_bits = -1;
   Bool_t exact = kFALSE;

   // First, try to find an exact matching GC.
   // If no one found, then use the closest one.

   TIter next(fList);

   while ((el = (TGGCElement *) next())) {
      matching_bits = MatchGC(el->fContext, values);
      if (matching_bits > best_matching_bits) {
         best_matching_bits = matching_bits;
         best_match = el;
         if ((el->fContext->fValues.fMask & values->fMask) == values->fMask) {
            exact = kTRUE;
            break;
         }
      }
   }

   if (best_match) {
      if (gDebug > 0)
         Printf("<TGGCPool::GetGC>: %smatching GC found\n", exact ? "exact " : "");
      best_match->AddReference();
      if (!exact) {
         // add missing values to the best_match'ing GC...
         UpdateGC(best_match->fContext, values);
      }
      return best_match->fContext;
   }

   TGGC *gc = new TGGC(values);

   el = new TGGCElement;
   el->fContext = gc;
   fList->Add(el);

   return gc;
}

//______________________________________________________________________________
Int_t TGGCPool::MatchGC(TGGC *gc, GCValues_t *values)
{
   // Try to find matching graphics context. On success returns the amount
   // of matching bits (which may be zero if masks have no common bits),
   // -1 on failure (when there are common bits but not a single match).

   Mask_t bit, common_bits;
   Int_t  matching_bits = -1;
   Bool_t match = kFALSE;
   GCValues_t *gcv = &gc->fValues;

   common_bits = values->fMask & gcv->fMask;

   if (common_bits == 0) return 0;  // no common bits, a possible
                                    // candidate anyway.

   for (bit = 1; bit <= common_bits; bit <<= 1) {
      switch (bit & common_bits) {
         default:
         case 0:
            continue;
            break;
         case kGCFunction:
            match = (values->fFunction == gcv->fFunction);
            break;
         case kGCPlaneMask:
            match = (values->fPlaneMask == gcv->fPlaneMask);
            break;
         case kGCForeground:
            match = (values->fForeground == gcv->fForeground);
            break;
         case kGCBackground:
            match = (values->fBackground == gcv->fBackground);
            break;
         case kGCLineWidth:
            match = (values->fLineWidth == gcv->fLineWidth);
            break;
         case kGCLineStyle:
            match = (values->fLineStyle == gcv->fLineStyle);
            break;
         case kGCCapStyle:
            match = (values->fCapStyle == gcv->fCapStyle);
            break;
         case kGCJoinStyle:
            match = (values->fJoinStyle == gcv->fJoinStyle);
            break;
         case kGCFillStyle:
            match = (values->fFillStyle == gcv->fFillStyle);
            break;
         case kGCFillRule:
            match = (values->fFillRule == gcv->fFillRule);
            break;
         case kGCTile:
            match = (values->fTile == gcv->fTile);
            break;
         case kGCStipple:
            match = (values->fStipple == gcv->fStipple);
            break;
         case kGCTileStipXOrigin:
            match = (values->fTsXOrigin == gcv->fTsXOrigin);
            break;
         case kGCTileStipYOrigin:
            match = (values->fTsYOrigin == gcv->fTsYOrigin);
            break;
         case kGCFont:
            match = (values->fFont == gcv->fFont);
            break;
         case kGCSubwindowMode:
            match = (values->fSubwindowMode == gcv->fSubwindowMode);
            break;
         case kGCGraphicsExposures:
            match = (values->fGraphicsExposures == gcv->fGraphicsExposures);
            break;
         case kGCClipXOrigin:
            match = (values->fClipXOrigin == gcv->fClipXOrigin);
            break;
         case kGCClipYOrigin:
            match = (values->fClipYOrigin == gcv->fClipYOrigin);
            break;
         case kGCClipMask:
            match = (values->fClipMask == gcv->fClipMask);
            break;
         case kGCDashOffset:
            match = (values->fDashOffset == gcv->fDashOffset);
            break;
         case kGCDashList:
            if (values->fDashLen == gcv->fDashLen)
               match = (strncmp(values->fDashes, gcv->fDashes, gcv->fDashLen) == 0);
            break;
         case kGCArcMode:
           match = (values->fArcMode == gcv->fArcMode);
           break;
      }
      if (!match)
         return -1;
      matching_bits++;
      match = kFALSE;
   }

   return matching_bits;
}

//______________________________________________________________________________
void TGGCPool::UpdateGC(TGGC *gc, GCValues_t *values)
{
   // Update graphics context with the values spcified in values->fMask.

   gc->SetAttributes(values);
}
