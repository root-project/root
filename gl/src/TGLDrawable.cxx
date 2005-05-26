// @(#)root/gl:$Name:$:$Id:$
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// TODO: Function descriptions
// TODO: Class def - same as header

#include "TGLDrawable.h"
#include "TGLDisplayListCache.h"

ClassImp(TGLDrawable)

//______________________________________________________________________________
TGLDrawable::TGLDrawable(UInt_t ID, Bool_t DLCache) :
   fID(ID), fDLCache(DLCache)
{
}

//______________________________________________________________________________
TGLDrawable::~TGLDrawable()
{
   Purge();
}

//______________________________________________________________________________
Bool_t TGLDrawable::UseDLCache(UInt_t /*LOD*/) const
{
   return fDLCache;
}

//______________________________________________________________________________
Bool_t TGLDrawable::SetDLCache(Bool_t DLCache)
{
   if (DLCache == fDLCache) {
      return kFALSE;
   }

   fDLCache = DLCache;
   if (!fDLCache) {
      TGLDisplayListCache::Instance().Purge(*this);
   }

   return true;
}

//______________________________________________________________________________
void TGLDrawable::Draw(UInt_t LOD) const
{
   TGLDisplayListCache & cache = TGLDisplayListCache::Instance();

   // If shape is not cached, or a capture to cache is already in progress
   // perform a direct draw
   // DL can be nested, but not created in nested fashion. As we only
   // build DL on draw demands have to protected against this here.
   if (!UseDLCache(LOD) || cache.InsideCapture())
   {
      DirectDraw(LOD);
      return;
   }

   if (!cache.Draw(*this, LOD))
   {
      // Capture the shape draw into compiled DL
      // If the cache is disabled the capture is ignored and
      // the shape is directly drawn
      cache.OpenCapture(*this, LOD);
      DirectDraw(LOD);

      // TODO: Time 100% draw and pass back for sorting?

      // If capture was done then DL was just GL_COMPILE - we need to actually
      // draw it now
      if (cache.CloseCapture()) {
         Bool_t ok = cache.Draw(*this, LOD);
         assert(ok);
      }
   }
}

//______________________________________________________________________________
void TGLDrawable::Purge()
{
   TGLDisplayListCache::Instance().Purge(*this);
}
