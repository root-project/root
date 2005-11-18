// @(#)root/gl:$Name:  $:$Id: TGLDisplayListCache.cxx,v 1.6 2005/08/30 10:29:52 brun Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLDisplayListCache.h"
#include "TGLUtil.h"
#include "TGLIncludes.h"
#include "TError.h"
#include "Riostream.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLDisplayListCache                                                  //
//                                                                      //
// Singleton cache of GL display lists. Provides lazy automatic GL      //
// display list capture of draws by a TGLDrawable at a certain 'LOD'    //
// which can be disable on a global, per drawable or LOD basis.         //
//                                                                      //
// Internally the cache creates a block of GL display lists of fSize,   // 
// and maintains a stl::map, mapping CacheID_t ids (created from        //
// TGLDrawable and LOD draw flag) to 'name' entries in the GL display   //
// list block.                                                          //
//                                                                      //
// See TGLDrawable::Draw() for use.                                     //
// NOTE: Purging of individual TGLDrawables not currently implemented:  //
//       Purge(const TGLDrawable & drawable)                            //    
//       Purge(const TGLDrawable & drawable, UInt_t LOD)                // //////////////////////////////////////////////////////////////////////////

ClassImp(TGLDisplayListCache)

TGLDisplayListCache * TGLDisplayListCache::fgInstance = 0;
const UInt_t TGLDisplayListCache::fgInvalidDLName = 0;

//______________________________________________________________________________
TGLDisplayListCache & TGLDisplayListCache::Instance()
{
   // Create (if required) and return the singleton display list cache
   if (!fgInstance) {
      fgInstance = new TGLDisplayListCache();
   }

   return *fgInstance;
}

//______________________________________________________________________________
TGLDisplayListCache::TGLDisplayListCache(Bool_t enable, UInt_t size) :
      fSize(size), fInit(kFALSE), fEnabled(enable), fCaptureOpen(kFALSE),
      fDLBase(fgInvalidDLName), fDLNextFree(fgInvalidDLName)
{
   // Construct display list cache.
   // Private constructor - cache is singleton obtained through 
   // TGLDisplayListCache::Instance()
}

//______________________________________________________________________________
TGLDisplayListCache::~TGLDisplayListCache()
{
   // Destroy display list cache - deleting internal GL display list block
   glDeleteLists(fDLBase,fSize);
}

//______________________________________________________________________________
void TGLDisplayListCache::Init()
{
   // Initialise the cache - create the internal GL display list block of size
   // fSize
   fDLBase = glGenLists(fSize);
   fDLNextFree = fDLBase;
   TGLUtil::CheckError();
   fInit = kTRUE;
}

//______________________________________________________________________________
Bool_t TGLDisplayListCache::Draw(const TGLDrawable & drawable, UInt_t LOD) const
{
   // Draw (call) the GL dislay list entry associated with the drawable / LOD
   // flag pair, and return kTRUE. If no list item associated, return KFALSE.
   if (!fEnabled) {
      return kFALSE;
   }

   // TODO: Cache the lookup here ? As may have many calls of same draw/qual in a row
   UInt_t drawList = Find(MakeCacheID(drawable, LOD));

   if (drawList == fgInvalidDLName) {
      if (gDebug>4) {
         Info("TGLDisplayListCache::Draw", "no cache for drawable %d LOD %d", &drawable, LOD);
      }
      return kFALSE;
   }

   if (gDebug>4) {
      Info("TGLDisplayListCache::Draw", "drawable %d LOD %d", &drawable, LOD);
   }
   glCallList(drawList);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLDisplayListCache::OpenCapture(const TGLDrawable & drawable, UInt_t LOD)
{
   // Open capture of GL draw commands into cache entry, associated with 
   // drawable / LOD pair. Capture is done in GL_COMPILE mode - so the cache 
   // entry has to be drawn using TGLDisplayListCache::Draw() after capture close.
   //
   // Return kTRUE if opened, kFALSE if not. Capture is not opened if cache not 
   // enabled or cache if full.
   // GL display lists can be nested at 'run' (draw) time, but cannot be nested
   // in capture - so attempting to nest captures is rejected.
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
   // Close currently open capture - return kTRUE if one open and successfully
   // closed.
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
   // Purge all entries for all drawable/LOD pairs from cache
   glDeleteLists(fDLBase,fSize);
   fCacheDLMap.erase(fCacheDLMap.begin(), fCacheDLMap.end());
   fInit = kFALSE;
}

//______________________________________________________________________________
void TGLDisplayListCache::Purge(const TGLDrawable & /* drawable */) 
{ 
   // Purge all entries for a drawable at any LOD from cache
   // NOTE: NOT IMPLEMENTED AT PRESENT!
   if(fCaptureOpen) {
      Error("TGLDisplayListCache::Purge", "attempt to purge while capture open");
      return;
   }

   // TODO
}

//______________________________________________________________________________
void TGLDisplayListCache::Purge(const TGLDrawable & /* drawable */, UInt_t /* LOD */) 
{ 
   // Purge entry for a drawable/LOD from cache
   // NOTE: NOT IMPLEMENTED AT PRESENT!
   if(fCaptureOpen) {
      Error("TGLDisplayListCache::Purge", "attempt to purge while capture open");
      return;
   }

   // TODO
}

//______________________________________________________________________________
UInt_t TGLDisplayListCache::Find(CacheID_t cacheID) const
{
   // Find GL display list block 'name' assocaited with the cacheID 
   // cacheID is unqiuely generated from drawable/LOD pair - see 
   // TGLDisplayListCache::MakeCacheID()
   // Look at Effect STL on efficiency .
   CacheDLMap_t::const_iterator cacheDLMapIt;
   cacheDLMapIt = fCacheDLMap.find(cacheID);

   if (cacheDLMapIt != fCacheDLMap.end()) {
      return (*cacheDLMapIt).second;
   }
   return fgInvalidDLName;

}

// TODO: Inline this
//______________________________________________________________________________
TGLDisplayListCache::CacheID_t TGLDisplayListCache::MakeCacheID(const TGLDrawable & drawable,
                                                                const UInt_t LOD) const
{
   // Create a CacheID_t from drawable/LOD pair provided. The CacheID_t returned 
   // is actually a std::pair<const TGLDrawable *, const UInt_t>, and used as key
   // in cache stl::map to map to GL display list 'name'
   return CacheID_t(&drawable, LOD);
}

//______________________________________________________________________________
void TGLDisplayListCache::Dump() const
{
   // Dump usage count of cache
   Info("TGLDisplayListCache::Dump", "%d of %d used", (fDLNextFree - fDLBase), fSize);
}
