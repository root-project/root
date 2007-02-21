// @(#)root/gl:$Name:  $:$Id: TGLDisplayListCache.cxx,v 1.13 2006/12/09 23:01:54 rdm Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLDisplayListCache.h"
#include "TGLDrawFlags.h"
#include "TGLUtil.h"
#include "TGLIncludes.h"
#include "TVirtualGL.h"

#include "Riostream.h"
#include "TError.h"

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
//       Purge(const TGLDrawable & drawable, UInt_t LOD)                // 
//////////////////////////////////////////////////////////////////////////

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
      fDLBase(fgInvalidDLName), fDLNextFree(fgInvalidDLName),
      fCaptureFullReported(kFALSE)
{
   // Construct display list cache.
   // Private constructor - cache is singleton obtained through 
   // TGLDisplayListCache::Instance()
}

//______________________________________________________________________________
TGLDisplayListCache::~TGLDisplayListCache()
{
   // Destroy display list cache - deleting internal GL display list block
   if (gVirtualGL)
      gVirtualGL->DeleteGLLists(fDLBase,fSize);
}

//______________________________________________________________________________
void TGLDisplayListCache::Init()
{
   // Initialise the cache - create the internal GL display list block of size
   // fSize
   if (gVirtualGL)
      fDLBase = gVirtualGL->CreateGLLists(fSize);
   fDLNextFree = fDLBase;
   TGLUtil::CheckError("TGLDisplayListCache::Init");
   fInit = kTRUE;
}

//______________________________________________________________________________
Bool_t TGLDisplayListCache::Draw(const TGLDrawable & drawable, const TGLDrawFlags & flags) const
{
   // Draw (call) the GL dislay list entry associated with the drawable / LOD
   // flag pair, and return kTRUE. If no list item associated, return KFALSE.
   if (!fEnabled) {
      return kFALSE;
   }

   // TODO: Cache the lookup here ? As may have many calls of same draw/qual in a row
   UInt_t drawList = Find(MakeCacheID(drawable, flags));

   if (drawList == fgInvalidDLName) {
      if (gDebug>4) {
         Info("TGLDisplayListCache::Draw", "no cache for drawable %d LOD %d", &drawable, flags.LOD());
      }
      return kFALSE;
   }

   if (gDebug>4) {
      Info("TGLDisplayListCache::Draw", "drawable %d LOD %d", &drawable, flags.LOD());
   }
   glCallList(drawList);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLDisplayListCache::OpenCapture(const TGLDrawable & drawable, const TGLDrawFlags & flags)
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
   // TODO: Start to loop or recycle in another fashion?
   if (fDLNextFree > fDLBase + fSize) {
      if (!fCaptureFullReported) {
         Warning("TGLDisplayListCache::OpenCapture", "cache is full");
         fCaptureFullReported = kTRUE;
      }
      return kFALSE;
   }

   fCaptureOpen = kTRUE;

   if (!fInit)
   {
      Init();
   }

   if (gDebug>4) {
      Info("TGLDisplayListCache::OpenCapture", "for drawable %d LOD %d", &drawable, flags.LOD());
   }

   CacheID_t cacheID = MakeCacheID(drawable, flags);
   fCacheDLMap.insert(CacheDLMap_t::value_type(cacheID,fDLNextFree));
   //assert( Find(cacheID) == fDLNextFree );

   glNewList(fDLNextFree,GL_COMPILE);
   TGLUtil::CheckError("TGLDisplayListCache::OpenCapture");

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
   TGLUtil::CheckError("TGLDisplayListCache::CloseCapture");
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
   if (gVirtualGL)
      gVirtualGL->DeleteGLLists(fDLBase,fSize);
   fCacheDLMap.erase(fCacheDLMap.begin(), fCacheDLMap.end());
   fInit = kFALSE;
   fCaptureFullReported = kFALSE;
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
void TGLDisplayListCache::Purge(const TGLDrawable & /* drawable */, const TGLDrawFlags & /* flags */) 
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
   // Return the GL display list block name associated with 'cacheID' 
   // cacheID is generated from drawable/flags pair - see 
   // TGLDisplayListCache::MakeCacheID()
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
                                                                const TGLDrawFlags & flags) const
{
   // Create a CacheID_t from drawable/flags pair provided. 

   // NOTE: Display Lists CAN capture GL state changes as well as geometry, but will 
   // not capture the current state set BEFORE the DL is opened.

   // The CacheID_t returned is a std::pair<const TGLDrawable *, const UInt_t>, 
   // and used as key into fCacheDLMap - see TGLDisplayListCache::Find()
   return CacheID_t(&drawable, flags.LOD());
}

//______________________________________________________________________________
void TGLDisplayListCache::Dump() const
{
   // Dump usage count of cache
   Info("TGLDisplayListCache::Dump", "%d of %d used", (fDLNextFree - fDLBase), fSize);
}
