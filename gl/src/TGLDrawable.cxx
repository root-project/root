// @(#)root/gl:$Name:  $:$Id: TGLDrawable.cxx,v 1.12 2006/02/20 11:02:19 brun Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLDrawable.h"
#include "TGLDrawFlags.h"
#include "TGLDisplayListCache.h"

// For debug tracing
#include "TClass.h" 
#include "TError.h"

TGLQuadric TGLDrawable::fgQuad;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLDrawable                                                          //      
//                                                                      //
// Abstract base class for all GL drawable objects - TGLPhysicalShape & //
// TGLLogicalShape hierarchy. Provides hooks for using the display list //
// cache in TGLDrawable::Draw() - the external draw method for all      //
// shapes.                                                              //
//                                                                      //
// Defines pure virtual TGLDrawable::DirectDraw() which derived classes //
// must implement with actual GL drawing.                               //
//                                                                      //
// Display list caching can occur at either the physical or logical     //
// level (with or without translation). Currently we cache only certain //
// derived logical shapes as not all logicals can respect the LOD draw  //
// flag which is used in caching.                                       //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLDrawable)

//______________________________________________________________________________
TGLDrawable::TGLDrawable(ULong_t ID, Bool_t cached) :
   fID(ID), fCached(cached)
{
   // Construct GL drawable object, with 'ID'. Bool 'DLCache'
   // indicates if draws should be captured to the display list cache
   // singleton (TGLDisplayListCache).
}

//______________________________________________________________________________
TGLDrawable::TGLDrawable(const TGLDrawable& gld) : 
  fID(gld.fID),
  fCached(gld.fCached),
  fBoundingBox(gld.fBoundingBox)
{ }

//______________________________________________________________________________
const TGLDrawable& TGLDrawable::operator=(const TGLDrawable& gld) 
{
  if(this!=&gld) {
    fID=gld.fID;
    fCached=gld.fCached;
    fBoundingBox=gld.fBoundingBox;
  } return *this;
}

//______________________________________________________________________________
TGLDrawable::~TGLDrawable()
{
   // Destroy the GL drawable.
   Purge();
}

//______________________________________________________________________________
Bool_t TGLDrawable::SetCached(Bool_t cached)
{
   // Modify capture of draws into display list cache
   // kTRUE - capture, kFALSE direct draw
   // Return kTRUE is state changed, kFALSE if not
   if (cached == fCached) {
      return kFALSE;
   }

   fCached = cached;

   // Purge out any existing DL cache entries
   // Note: This does nothing at present as per drawable purging is not implemented
   // in TGLDisplayListCache.
   if (!fCached) {
      TGLDisplayListCache::Instance().Purge(*this);
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLDrawable::ShouldCache(const TGLDrawFlags & /*flags*/) const
{
   // Returns kTRUE if draws should be display list cache
   // kFALSE otherwise

   // Default is to ignore flags and use internal bool. In some cases
   // shapes may want to override and constrain caching to certain
   // styles/LOD found in flags
   return fCached;
}

//______________________________________________________________________________
void TGLDrawable::Draw(const TGLDrawFlags & flags) const
{
   // Draw the GL drawable, using draw flags. If DL caching is enabled
   // (see SetCached) then attempt to draw from the cache, if not found
   // attempt to capture the draw - done by DirectDraw() - into a new cache entry.
   // If not cached just call DirectDraw() for normal non DL cached drawing.
   
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLDrawable::Draw", "this %d (class %s) LOD %d", this, IsA()->GetName(), flags.LOD());
   }

   TGLDisplayListCache & cache = TGLDisplayListCache::Instance();

   // If shape is not cached, or a capture to cache is already in progress
   // perform a direct draw
   // DL can be nested, but not created in nested fashion. As we only
   // build DL on draw demands have to protected against this here.
   if (!ShouldCache(flags) || cache.CaptureIsOpen())
   {
      DirectDraw(flags);
      return;
   }

   // Attempt to draw from the cache
   if (!cache.Draw(*this, flags))
   {
      // Draw failed - shape draw for flags is not cached
      // Attempt to capture the shape draw into cache now
      // If the cache is disabled the capture is ignored and
      // the shape is directly drawn
      cache.OpenCapture(*this, flags);
      DirectDraw(flags);

      // If capture was done then DL was just GL_COMPILE - we need to actually
      // draw it from the cache now
      if (cache.CloseCapture()) {
         Bool_t ok = cache.Draw(*this, flags);
         assert(ok);
      }
   }
}

//______________________________________________________________________________
void TGLDrawable::Purge()
{
   // Purge all entries for all LODs for this drawable from the display list cache.
   //
   // Note: This does nothing at present as per drawable purging is not implemented
   // in TGLDisplayListCache.
   TGLDisplayListCache::Instance().Purge(*this);
}
