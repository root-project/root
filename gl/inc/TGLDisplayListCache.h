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

/*************************************************************************
 * TGLDisplayListCache - TODO
 *
 *
 *
 *************************************************************************/
class TGLDisplayListCache {
private:   
   typedef std::pair<const TGLDrawable *, const UInt_t> CacheID_t;
   typedef std::map<CacheID_t,UInt_t>                  CacheDLMap_t;

   // Fields
   UInt_t         fSize;       //!
   Bool_t         fInit;       //!
   Bool_t         fEnabled;    //!
   Bool_t         fAddingNew;  //!
   UInt_t         fDLBase;     //!
   UInt_t         fDLNextFree; //!  
   CacheDLMap_t   fCacheDLMap; //!
   
   // Static Fields
   static TGLDisplayListCache * fInstance; //! the singleton cache instance
   static const UInt_t INVALID_DL_NAME;

   // Methods
   TGLDisplayListCache(Bool_t enable = true, UInt_t size = 10000);
   virtual ~TGLDisplayListCache(); // ClassDef introduces virtual fns
   
   // We can be used as common static cache - but we
   // can't call glGenLists before gl context is created (somewhere).
   // So we defer until first use of BeginNew and test fInit flag
   //TODO These may not be required now singleton?
   void Init();
   
   CacheID_t MakeCacheID(const TGLDrawable & drawable, UInt_t LOD) const;   
   UInt_t    Find(CacheID_t cacheID) const;

public:
   static TGLDisplayListCache & Instance();
   
   // Cache manipulators
   void   Enable(Bool_t enable)   { fEnabled = enable; }
   Bool_t IsEnabled()             { return fEnabled; }
   //void   Resize(UInt_t size)     {}; //TODO
   void   Dump() const; 

   // Cache entities (TLGDrawable) manipulators
   Bool_t Draw(const TGLDrawable & drawable, UInt_t LOD) const;
   Bool_t OpenCapture(const TGLDrawable & drawable, UInt_t LOD);
   Bool_t CloseCapture();
   Bool_t InsideCapture() { return fAddingNew; }
   Bool_t Purge(const TGLDrawable & /* drawable */ ) { /*assert(!fAddingNew);*/ return true; } //TODO
   Bool_t Purge(const TGLDrawable & /* drawable */, UInt_t /* LOD */) { /*assert(!fAddingNew);*/ return true; } //TODO

   ClassDef(TGLDisplayListCache,0) // a cache of GL display lists (singleton)
};

#endif // ROOT_TGLDisplayListCache

