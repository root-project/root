// @(#)root/gui:$Name:  $:$Id: TGGC.cxx,v 1.4 2001/06/22 16:10:17 rdm Exp $
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

#include "TGClient.h"
#include "TGGC.h"
#include "TVirtualX.h"
#include "THashTable.h"
#include <string.h>


ClassImp(TGGC)

//______________________________________________________________________________
TGGC::TGGC(GCValues_t *values, Bool_t)
{
   // Create a graphics context (only called via TGGCPool::GetGC()).

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
   SetRefCount(1);
}

//______________________________________________________________________________
TGGC::TGGC(GCValues_t *values)
{
   // Create a graphics context, registers GC in GCPool.

   // case of default ctor at program startup before gClient exists
   if (!values) {
      memset(&fValues, 0, sizeof(GCValues_t));
      fContext = 0;
      SetRefCount(1);
      return;
   }

   if (gClient)
      gClient->GetGC(values, kTRUE);
   else {
      fContext = 0;
      Error("TGGC", "TGClient not yet initialized, should never happen");
   }
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
   SetRefCount(1);

   if (gClient)
      gClient->GetGCPool()->fList->Add(this);
}

//______________________________________________________________________________
TGGC::~TGGC()
{
   // Delete graphics context.

   if (gClient)
      gClient->GetGCPool()->ForceFreeGC(this);

   if (fContext)
      gVirtualX->DeleteGC(fContext);
}

//______________________________________________________________________________
TGGC &TGGC::operator=(const TGGC &rhs)
{
   // Graphics context assignment operator.

   if (this != &rhs) {
      if (!fContext && gClient) {
         TGGC *gc = gClient->GetGCPool()->FindGC(this);
         if (!gc)
            gClient->GetGCPool()->fList->Add(this);
      }
      if (fContext)
         gVirtualX->DeleteGC(fContext);
      TObject::operator=(rhs);
      fValues  = rhs.fValues;
      fContext = gVirtualX->CreateGC(gVirtualX->GetDefaultRootWindow(), &fValues);
      if (fValues.fMask & kGCDashList)
         gVirtualX->SetDashes(fContext, fValues.fDashOffset, fValues.fDashes,
                              fValues.fDashLen);
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

   for (Mask_t bit = 1; bit <= fValues.fMask; bit <<= 1) {
      switch (bit & values->fMask) {
         default:
         case 0:
            continue;
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

   if (!fContext && gClient) {
      TGGC *gc = gClient->GetGCPool()->FindGC(this);
      if (!gc)
         gClient->GetGCPool()->fList->Add(this);
   }

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
void TGGC::SetDashList(const char v[], Int_t len)
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

//______________________________________________________________________________
void TGGC::Print(Option_t *) const
{
   // Print graphics contexts info.

   Printf("TGGC: mask = %lx, handle = %lx, ref cnt = %u", fValues.fMask,
          fContext, References());
}


ClassImp(TGGCPool)

//______________________________________________________________________________
TGGCPool::TGGCPool(TGClient *client)
{
   // Create graphics context pool.

   fClient = client;
   fList   = new THashTable;
   fList->SetOwner();
}

//______________________________________________________________________________
TGGCPool::~TGGCPool()
{
   // Delete graphics context pool.

   delete fList;
}

//______________________________________________________________________________
void TGGCPool::ForceFreeGC(const TGGC *gct)
{
   // Force remove graphics context from list. Is only called via ~TGGC().

   TGGC *gc = (TGGC *) fList->FindObject(gct);

   if (gc) {
      if (gc->References() > 1)
         Error("ForceFreeGC", "removed a shared graphics context\n",
               "best to use graphics contexts via the TGGCPool()");
      fList->Remove(gc);
   }
}

//______________________________________________________________________________
void TGGCPool::FreeGC(const TGGC *gct)
{
   // Delete graphics context if it is not used anymore.

   TGGC *gc = (TGGC *) fList->FindObject(gct);

   if (gc) {
      if (!gc->RemoveReference() == 0) {
         fList->Remove(gc);
         delete gc;
      }
   }
}

//______________________________________________________________________________
void TGGCPool::FreeGC(GContext_t gct)
{
   // Delete graphics context if it is not used anymore.

   TIter next(fList);

   while (TGGC *gc = (TGGC *) next()) {
      if (gc->fContext == gct) {
         if (gc->RemoveReference() == 0) {
            fList->Remove(gc);
            delete gc;
            return;
         }
      }
   }
}

//______________________________________________________________________________
TGGC *TGGCPool::FindGC(const TGGC *gct)
{
   // Find graphics context. Returns 0 in case gc is not found.

   return (TGGC*) fList->FindObject(gct);
}

//______________________________________________________________________________
TGGC *TGGCPool::FindGC(GContext_t gct)
{
   // Find graphics context based on its GContext_t handle. Returns 0
   // in case gc is not found.

   TIter next(fList);

   while (TGGC *gc = (TGGC *) next()) {
      if (gc->fContext == gct)
         return gc;
   }
   return 0;
}

//______________________________________________________________________________
TGGC *TGGCPool::GetGC(GCValues_t *values, Bool_t rw)
{
   // Get the best matching graphics context depending on values.
   // If rw is false only a readonly, not modifiable graphics context
   // is returned. If rw is true a new modifiable graphics context is
   // returned.

   TGGC *gc, *best_match = 0;
   Int_t matching_bits, best_matching_bits = -1;
   Bool_t exact = kFALSE;

   if (!values)
      rw = kTRUE;

   if (!rw) {

      // First, try to find an exact matching GC.
      // If no one found, then use the closest one.

      TIter next(fList);

      while ((gc = (TGGC *) next())) {
         matching_bits = MatchGC(gc, values);
         if (matching_bits > best_matching_bits) {
            best_matching_bits = matching_bits;
            best_match = gc;
            if ((gc->fValues.fMask & values->fMask) == values->fMask) {
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
            UpdateGC(best_match, values);
         }
         return best_match;
      }
   }

   gc = new TGGC(values, kTRUE);

   fList->Add(gc);

   return gc;
}

//______________________________________________________________________________
Int_t TGGCPool::MatchGC(const TGGC *gc, GCValues_t *values)
{
   // Try to find matching graphics context. On success returns the amount
   // of matching bits (which may be zero if masks have no common bits),
   // -1 on failure (when there are common bits but not a single match).

   Mask_t bit, common_bits;
   Int_t  matching_bits = -1;
   Bool_t match = kFALSE;
   const GCValues_t *gcv = &gc->fValues;

   common_bits = values->fMask & gcv->fMask;

   if (common_bits == 0) return 0;  // no common bits, a possible
                                    // candidate anyway.

   // Careful, check first the tile and stipple mask bits, as these
   // influence nearly all other GC functions... (do the same for
   // some other such bits as GCFunction, etc...). Perhaps we should
   // allow only exact GC matches.

   if (gcv->fMask & kGCTile)
      if ((gcv->fTile != kNone) && !(values->fMask & kGCTile)) return -1;
   if (values->fMask & kGCTile)
      if ((values->fTile != kNone) && !(gcv->fMask & kGCTile)) return -1;
   if (gcv->fMask & kGCStipple)
      if ((gcv->fStipple != kNone) && !(values->fMask & kGCStipple)) return -1;
   if (values->fMask & kGCStipple)
      if ((values->fStipple != kNone) && !(gcv->fMask & kGCStipple)) return -1;

   for (bit = 1; bit <= common_bits; bit <<= 1) {
      switch (bit & common_bits) {
         default:
         case 0:
            continue;
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

//______________________________________________________________________________
void TGGCPool::Print(Option_t *) const
{
   // List all graphics contexts in the pool.

   fList->Print();
}
