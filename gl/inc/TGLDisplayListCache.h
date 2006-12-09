// @(#)root/gl:$Name:  $:$Id: TGLDisplayListCache.h,v 1.8 2006/02/08 10:49:26 couet Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLDisplayListCache
#define ROOT_TGLDisplayListCache

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include <map>

class TGLDrawable;
class TGLDrawFlags;

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

class TGLDisplayListCache 
{
   private:
   typedef std::pair<const TGLDrawable *, const Short_t> CacheID_t;
   typedef std::map<CacheID_t,UInt_t>                    CacheDLMap_t;

   // Fields
   UInt_t         fSize;         //!
   Bool_t         fInit;         //!
   Bool_t         fEnabled;      //!
   Bool_t         fCaptureOpen;  //!
   UInt_t         fDLBase;       //!
   UInt_t         fDLNextFree;   //!
   CacheDLMap_t   fCacheDLMap;   //!
   Bool_t         fCaptureFullReported; //!

   // Static Fields
   static TGLDisplayListCache * fgInstance; //! the singleton cache instance
   static const UInt_t fgInvalidDLName;

   // Methods
   TGLDisplayListCache(Bool_t enable = true, UInt_t size = 10000);
   virtual ~TGLDisplayListCache(); // ClassDef introduces virtual fns

   // We can be used as common static cache - but we
   // can't call glGenLists before gl context is created (somewhere).
   // So we defer until first use of BeginNew and test fInit flag
   //TODO These may not be required now singleton?
   void Init();

   CacheID_t MakeCacheID(const TGLDrawable & drawable, const TGLDrawFlags & flags) const;
   UInt_t    Find(CacheID_t cacheID) const;

public:
   static TGLDisplayListCache & Instance();

   // Cache manipulators
   void   Enable(Bool_t enable)   { fEnabled = enable; }
   Bool_t IsEnabled()             { return fEnabled; }
   //void   Resize(UInt_t size)     {}; //TODO
   void   Purge();               // purge entire cache
   void   Dump() const;

   // Cache entities (TLGDrawable) manipulators
   Bool_t Draw(const TGLDrawable & drawable, const TGLDrawFlags & flags) const;
   Bool_t OpenCapture(const TGLDrawable & drawable, const TGLDrawFlags & flags);
   Bool_t CloseCapture();
   Bool_t CaptureIsOpen() { return fCaptureOpen; }
   void   Purge(const TGLDrawable & drawable);                             // NOT IMPLEMENTED
   void   Purge(const TGLDrawable & drawable, const TGLDrawFlags & flags); // NOT IMPLEMENTED

   ClassDef(TGLDisplayListCache,0) // a cache of GL display lists (singleton)
};

#endif // ROOT_TGLDisplayListCache

