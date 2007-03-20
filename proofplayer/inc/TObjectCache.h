// @(#)root/proofplayer:$Name:  $:$Id: TObjectCache.h,v 1.3 2006/07/01 11:39:37 rdm Exp $
// Author: M. Biskup 2/4/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TObjectCache
#define ROOT_TObjectCache

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjectCache                                                         //
//                                                                      //
// Manage small caches for TDirectories, TFiles or TTrees using LRU     //
// strategy.                                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TError
#include "TError.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif

#include <map>
#include <list>

class TFile;
class TDirectory;
class TTree;

// Template base class for a cache. ObjKey is a class used to identify objects and
// Obj class are the objects themselves. The cache uses LRU strategy and removes objects
// if the number of objects in the cache is too big. The lookup in the cache takes O(n)
// operations, where n is the cache size.
// Both ObjKey and Obj classes should implement ordering operators (equal, less then).

template <class ObjKey, class Obj> class TObjectCache {
public:
   typedef ObjKey TCacheKey;
   typedef Obj    TCacheObject;
   typedef std::pair<TCacheObject, Bool_t> ObjectAndBool_t;

private:
   // A list element. Helper structure for the implementation of the object cache.
   class TCacheElem : public TObject {
   public:
      TCacheObject fObj;
      TCacheKey    fKey;
      Int_t        fCount;
      TCacheElem(const TCacheObject &obj, const TCacheKey &objKey)
         : fObj(obj), fKey(objKey), fCount(0) { }
   };

   Int_t fMaxSize;   // The maximum size of the cache. If the size exceeds this
                     // value the least recently used object are removed,
                     // until the cache size drops below this threshold or there
                     // all other objects are in use.
   Int_t fSize;      // The number of objects currently stored in the cache.
   TList fMRU;       // Most recently used list element (but not used right now)
   TList fInUse;     // A list of all the objects in use.

   std::map<TCacheObject, TObjLink*> fObjMap; // a map from objects (any stored in the cache)
                                              // to list links
   std::map<TCacheKey, TObjLink*> fKeyMap;    // a map from object keys (any stored in the cache)
                                              // to list links

   TObjLink* FindByKey(const TCacheKey &k)
   {
      // Returns TObjLink entry storing a given object key. If no object with this
      // key is stored in the cache, 0 is returned.
      typename std::map<TCacheKey, TObjLink*>::iterator i = fKeyMap.find(k);
      if (i != fKeyMap.end())
         return i->second;
      return 0;
   }

   TObjLink* FindByObj(const TCacheObject &obj)
   {
      // Returns TObjLink entry storing a given object. If this object is not stored
      // in the cache, 0 is returned.
      typename std::map<TCacheObject, TObjLink*>::iterator i = fObjMap.find(obj);
      if (i != fObjMap.end())
         return i->second;
      else
         return 0;
   }

   void RemoveOld() {
      // Removes unused elements from the cache.
      // Uses LRU strategy.
      while (fMRU.GetSize() > 0 && fSize > fMaxSize) {
         TCacheElem *elem = dynamic_cast<TCacheElem*>(fMRU.Remove(fMRU.LastLink()));
         fObjMap.erase(elem->fObj);
         fKeyMap.erase(elem->fKey);
         Unload(elem->fObj);
         delete elem;
         fSize--;
      }
   }

protected:
   // The load function loads an objects given its key. Should be defined in subclasses.
   // The first element of the pair returned is the loaded object. The second tells wether
   // or not the loading was successfull (kTRUE - success, kFALSE - error). Only if the value
   // is kTRUE the object will be stored in the cache.
   virtual ObjectAndBool_t Load(const TCacheKey &k) = 0;

   // Called when removing the object from the cache. Can, e.g., delete the object (if it's a pointer).
   virtual void Unload(TCacheObject &/* o */) {}

public:
   TObjectCache() : fMaxSize(10), fSize(0) { }

   virtual ~TObjectCache()
   {
      // Destructor. Calls Unload for all the objects cached.
      // The existence of objects which haven't been released is considered
      // as an error.
      if (fInUse.GetSize() > 0) {
         ::Error("TEventIterTree::TObjectCache::~TObjectCache()",
                 "deleting the cache while some object haven't been released");
      }
      TIter next(&fInUse);
      while (TCacheElem* elem = dynamic_cast<TCacheElem*>(next())) {
         Unload(elem->fObj);
      }
      fInUse.SetOwner();  // ~fInUse will delete all the elements

      TIter next2(&fMRU);
      while (TCacheElem* elem = dynamic_cast<TCacheElem*>(next2())) {
         Unload(elem->fObj);
      }
      fMRU.SetOwner();  // ~fMRU will delete all the elements
   }

   TCacheObject Acquire(const TCacheKey &k)
   {
      // Returns an object given it's key. The object should be later released.
      // The same object can be acquired several times (returning the same, cached object)
      // The number of Release() calls should be equal to the number of
      // Acquire() calls.
      TObjLink *link = FindByKey(k);
      TCacheElem* elem;
      if (link == 0) {        // create a new element
         ObjectAndBool_t loadResult = Load(k);
         if (!loadResult.second)
            return loadResult.first;        // this value musn't be cached
         elem = new TCacheElem(loadResult.first, k);
         fInUse.AddFirst(elem);
         link = fInUse.FirstLink();
         fObjMap[loadResult.first] = link;
         fKeyMap[k] = link;
         fSize++;
      }
      else {
         elem = dynamic_cast<TCacheElem*>(link->GetObject());
         if (elem->fCount == 0) {  // bring the element to the InUse list
            fMRU.Remove(link);
            fInUse.AddFirst(elem);
            link = fInUse.FirstLink();
            fObjMap[elem->fObj] = link;
            fKeyMap[elem->fKey] = link;
         }
      }
      elem->fCount++;
      RemoveOld();
      return elem->fObj;
   }

   TCacheObject Acquire(const TString& fileName, const TString& dirName)
   {
      return Acquire(std::make_pair(fileName, dirName));
   }

   TCacheObject Acquire(const TString& fileName, const TString& dirName, const TString& treeName)
   {
      return Acquire(std::make_pair(fileName, std::make_pair(dirName, treeName)));
   }

   void Release(TCacheObject &obj)
   {
      // Releases the element acquired using Acquire() function. The object is not
      // deleted imediatelly but is kept in the cache. The next Acquire call with
      // the key of this object will return the same obj value. Objects are deleted
      // from the cache when its size is too big.
      TObjLink *link = FindByObj(obj);
      if (link == 0)
         ::Error("TEventIterTree::TObjectCache::Release()",
                       "Releasing an object not present in the cache");
      else {
         TCacheElem * elem = dynamic_cast<TCacheElem*>(link->GetObject());
         elem->fCount--;
         if (elem->fCount < 0) {
            ::Error("TEventIterTree::TObjectCache::Release()",
                    "Releasing an object more times than it has been acquired");
            elem->fCount = 0;
         }
         else if (elem->fCount == 0) {      // move it to the MRU list
            fInUse.Remove(link);
            fMRU.AddFirst(elem);
            link = fMRU.FirstLink();
            fObjMap[elem->fObj] = link;
            fKeyMap[elem->fKey] = link;
            RemoveOld();
         }
      }
   }

   Int_t GetMaxSize() const { return fMaxSize; }   // returns the maximum size of the cache
   void  SetMaxSize(Int_t size) { fMaxSize = size; }  // sets the maximum size of the cache
};


// A File Cache. Uses a singleton pattern with Instance() method.
class TFileCache : public TObjectCache<TString, TFile*> {
private:
   static TFileCache *fgInstance;
   TFileCache() {}
public:
   ObjectAndBool_t Load(const TString &fileName);
   virtual void Unload(TFile* &f);
   static TFileCache *Instance();
};

// A Directory Cache. Uses TFileCache for opening files.
class TDirectoryCache : public TObjectCache<std::pair<TString, TString>, TDirectory*> {
private:
   std::map<TDirectory*, TFile*> fDirectoryFiles; // an association Directory <-> its File
   static TDirectoryCache *fgInstance;
   TDirectoryCache() {}
public:
   ObjectAndBool_t Load(const TCacheKey &k);
   virtual void Unload(TDirectory* &dir);
   static TDirectoryCache *Instance();
};


// A Tree Cache. Uses TFileCache for opening files.
class TTreeFileCache : public TObjectCache<std::pair<TString, std::pair<TString, TString> >, TTree*> {
private:
   std::map<TTree*, TDirectory*> fTreeDirectories; // an association Tree <-> its File
public:
   ObjectAndBool_t Load(const TCacheKey &k);
   virtual void Unload(TTree* &tree);
};

#endif
