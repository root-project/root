// @(#)root/gl:$Name:  $:$Id: TGLDisplayListCache.cxx,v 1.4 2005/06/15 10:22:57 brun Exp $
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

#include "TGLDisplayListCache.h"
#include "TGLUtil.h"
#include "TGLIncludes.h"
#include "TError.h"
#include "Riostream.h"

ClassImp(TGLDisplayListCache)

TGLDisplayListCache * TGLDisplayListCache::fInstance = 0;
const UInt_t TGLDisplayListCache::INVALID_DL_NAME = 0;

//______________________________________________________________________________
TGLDisplayListCache & TGLDisplayListCache::Instance()
{
   if (!fInstance) {
      fInstance = new TGLDisplayListCache();
   }

   return *fInstance;
}

//______________________________________________________________________________
TGLDisplayListCache::TGLDisplayListCache(Bool_t enable, UInt_t size) :
      fSize(size), fInit(kFALSE), fEnabled(enable), fCaptureOpen(kFALSE),
      fDLBase(INVALID_DL_NAME), fDLNextFree(INVALID_DL_NAME)
{
}

//______________________________________________________________________________
TGLDisplayListCache::~TGLDisplayListCache()
{
   glDeleteLists(fDLBase,fSize);
}

//______________________________________________________________________________
void TGLDisplayListCache::Init()
{
   fDLBase = glGenLists(fSize);
   fDLNextFree = fDLBase;
   TGLUtil::CheckError();
   fInit = true;
}

//______________________________________________________________________________
Bool_t TGLDisplayListCache::Draw(const TGLDrawable & drawable, UInt_t LOD) const
{
   if (!fEnabled) {
      return kFALSE;
   }

   // TODO: Cache the lookup here ? As may have many calls of same draw/qual in a row
   UInt_t drawList = Find(MakeCacheID(drawable, LOD));

   if (drawList == INVALID_DL_NAME) {
      if (gDebug>4) {
         Info("TGLDisplayListCache::Draw", "no cache for drawable %d LOD %d", &drawable, LOD);
      }
      return kFALSE;
   }

   if (gDebug>4) {
      Info("TGLDisplayListCache::Draw", "drawable %d LOD %d", &drawable, LOD);
   }
   glCallList(drawList);
   return true;
}

//______________________________________________________________________________
Bool_t TGLDisplayListCache::OpenCapture(const TGLDrawable & drawable, UInt_t LOD)
{
   if (!fEnabled) {
      return kFALSE;
   }

   if (fCaptureOpen) {
      Error("TGLDisplayListCache::OpenCapture", "capture already ");
      return(kFALSE);
   }

   // Cache full?
   if (fDLNextFree > fDLBase + fSize) {
      return kFALSE;
   }

   fCaptureOpen = kTRUE;

   if (!fInit)
   {
      Init();
   }

   if (gDebug>4) {
      Info("TGLDisplayListCache::OpenCapture", "for drawable %d LOD %d", &drawable, LOD);
   }

   // TODO: Overflow of list cache - start to loop or recycle in another fashion?
   CacheID_t cacheID = MakeCacheID(drawable, LOD);
   fCacheDLMap.insert(CacheDLMap_t::value_type(cacheID,fDLNextFree));
   assert( Find(cacheID) == fDLNextFree );

   glNewList(fDLNextFree,GL_COMPILE);
   TGLUtil::CheckError();

   fDLNextFree++;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLDisplayListCache::CloseCapture()
{
   if (!fEnabled) {
      return kFALSE;
   }

   if (!fCaptureOpen) {
      Error("TGLDisplayListCache::CloseCapture", "no current capture open");
      return kFALSE;
   }

   glEndList();
   TGLUtil::CheckError();
   fCaptureOpen = kFALSE;

   if (gDebug>4) {
      Info("TGLDisplayListCache::CloseCapture","complete");
   }

   return kTRUE;
}

//______________________________________________________________________________
void TGLDisplayListCache::Purge()
{
   // Purge everything from the cache
   glDeleteLists(fDLBase,fSize);
   fCacheDLMap.erase(fCacheDLMap.begin(), fCacheDLMap.end());
   fInit = kFALSE;
}

//______________________________________________________________________________
void TGLDisplayListCache::Purge(const TGLDrawable & /* drawable */) 
{ 
   if(fCaptureOpen) {
      Error("TGLDisplayListCache::Purge", "attempt to purge while capture open");
      return;
   }

   // TODO

   return; 
}

//______________________________________________________________________________
void TGLDisplayListCache::Purge(const TGLDrawable & /* drawable */, UInt_t /* LOD */) 
{ 
   if(fCaptureOpen) {
      Error("TGLDisplayListCache::Purge", "attempt to purge while capture open");
      return;
   }

   // TODO

   return;
}

//______________________________________________________________________________
UInt_t TGLDisplayListCache::Find(CacheID_t cacheID) const
{
   // Look at Effect STL on efficiency .
   CacheDLMap_t::const_iterator cacheDLMapIt;
   cacheDLMapIt = fCacheDLMap.find(cacheID);

   if (cacheDLMapIt != fCacheDLMap.end()) {
      return (*cacheDLMapIt).second;
   }
   return INVALID_DL_NAME;

}

// TODO: Inline this
//______________________________________________________________________________
TGLDisplayListCache::CacheID_t TGLDisplayListCache::MakeCacheID(const TGLDrawable & drawable,
                                                                const UInt_t LOD) const
{
   return CacheID_t(&drawable, LOD);
}

//______________________________________________________________________________
void TGLDisplayListCache::Dump() const
{
   Info("TGLDisplayListCache::Dump", "%d of %d used", (fDLNextFree - fDLBase), fSize);
}
