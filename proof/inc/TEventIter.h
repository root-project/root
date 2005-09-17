// @(#)root/proof:$Name:  $:$Id: TEventIter.h,v 1.9 2005/07/09 04:03:23 brun Exp $
// Author: Maarten Ballintijn   07/01/02

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEventIter
#define ROOT_TEventIter

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEventIter                                                           //
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

class TDSet;
class TDSetElement;
class TFile;
class TDirectory;
class TSelector;
class TList;
class TIter;
class TTree;
class TEventList;

//------------------------------------------------------------------------

class TEventIter : public TObject {

protected:
   TDSet         *fDSet;         // data set over which to iterate

   TDSetElement  *fElem;         // Current Element

   TString        fFilename;     // Name of the current file
   TFile         *fFile;         // Current file
   Long64_t       fOldBytesRead; // last reported number of bytes read
   TString        fPath;         // Path to current TDirectory
   TDirectory    *fDir;          // directory containing the objects or the TTree
   Long64_t       fElemFirst;    // first entry to process for this element
   Long64_t       fElemNum;      // number of entries to process for this element
   Long64_t       fElemCur;      // current entry for this element

   TSelector     *fSel;          // selector to be used
   Long64_t       fFirst;        // first entry to process
   Long64_t       fNum;          // number of entries to process
   Long64_t       fCur;          // current entry
   Bool_t         fStop;         // termination of run requested
   TEventList    *fEventList;    //! eventList for processing
   Int_t          fEventListPos; //! current position in the eventList

   Int_t          LoadDir();     // Load the directory pointed to by fElem

public:
   TEventIter();
   TEventIter(TDSet *dset, TSelector *sel, Long64_t first, Long64_t num);
   virtual ~TEventIter();

   virtual Long64_t  GetNextEvent() = 0;
   virtual void      StopProcess(Bool_t abort);

   static TEventIter *Create(TDSet *dset, TSelector *sel, Long64_t first, Long64_t num);

   ClassDef(TEventIter,0)  // Event iterator used by TProofPlayer's
};


//------------------------------------------------------------------------

class TEventIterObj : public TEventIter {

private:
   TString  fClassName;    // class name of objects to iterate over
   TList   *fKeys;         // list of keys
   TIter   *fNextKey;      // next key in directory
   TObject *fObj;          // object found

public:
   TEventIterObj();
   TEventIterObj(TDSet *dset, TSelector *sel, Long64_t first, Long64_t num);
   ~TEventIterObj();

   Long64_t GetNextEvent();

   ClassDef(TEventIterObj,0)  // Event iterator for objects
};


//------------------------------------------------------------------------

class TEventIterTree : public TEventIter {

public:

   template <class ObjKey, class Obj> class TObjectCache {
      // Template base class for a cache. ObjKey is a class used to identify objects and
      // Obj class are the objects themselves. The cache uses LRU strategy and removes objects
      // if the number of objects in the cache is too big. The lookup in the cache takes O(n) 
      // operations, where n is the cache size.
      // Both ObjKey and Obj classes should implement ordering operators (equal, less then).
   public:
      typedef ObjKey TCacheKey;
      typedef Obj    TCacheObject;
      typedef std::pair<TCacheObject, Bool_t> ObjectAndBool_t;

   private:
      // A list element. Helper structure for the implementation of the object cache.
      class TCacheElem : public TObject {
      public:
         TCacheElem(const Obj &obj, const TCacheKey &objKey) 
            : fObj(obj), fKey(objKey), fCount(0) {}
         TCacheObject fObj;
         TCacheKey fKey;
         Int_t fCount;
      };
      // The maximum size of the cache. If the size exceeds this value the least recently 
      // used object are removed, until the cache size drops below this threshold or there
      // all other objects are in use.
      Int_t fMaxSize;

      TObjLink* FindByKey(const TCacheKey &k)
      {
         // Returns TObjLink entry storing a given object key. If no object with this
         // key is stored in the cache, 0 is returned.
         typename std::map<TCacheKey, TObjLink*>::iterator i = fKeyMap.find(k);
         if (i != fKeyMap.end())
            return i->second;
         else 
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
      TList fMRU;                // Most recently used list element
      TList fInUse;              // A list of all the objects in use.

      Int_t fSize;                // The number of objects currently stored in the cache.
      std::map<TCacheObject, TObjLink*> fObjMap;    // a map from objects (any stored in the cache)
                                                    // to list links
      std::map<TCacheKey, TObjLink*> fKeyMap; // a map from object keys (any stored in the cache)
                                              // to list links

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
      TObjectCache() 
      {
         // Constructor.
         fSize = 0;
         fMaxSize = 10;
      }
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
            delete elem;
         }
         TIter next2(&fMRU);
         while (TCacheElem* elem = dynamic_cast<TCacheElem*>(next2())) {
            Unload(elem->fObj);
            delete elem;
         }
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

      void Release(TCacheObject &obj) {
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
      void SetMaxSize(Int_t size) { fMaxSize = size; }  // sets the maximum size of the cache
   };


   // A File Cache. Uses a singleton pattern with Instance() method.
   class TFileCache : public TObjectCache<TString, TFile*> {
   private:
      static TFileCache* fgInstance;
      TFileCache() {}
   public:
      ObjectAndBool_t Load(const TString &fileName);
      virtual void Unload(TFile* &f);
      static TFileCache* Instance();
   };

   // A Directory Cache. Uses TFileCache for opening files.
   class TDirectoryCache : public TObjectCache<std::pair<TString, TString>, TDirectory*> {
   private:
      std::map<TDirectory*, TFile*> fDirectoryFiles; // an assotiation Directory <-> its File
      static TDirectoryCache* fgInstance;
      TDirectoryCache() {}
   public:
      TDirectory* Acquire(const TString& fileName, const TString& dirName) ;
      ObjectAndBool_t Load(const TCacheKey &k);
      virtual void Unload(TDirectory* &dir);
      static TDirectoryCache* Instance();
   };


   // A Tree Cache. Uses TFileCache for opening files.
   class TTreeCache : public TObjectCache<std::pair<TString, std::pair<TString, TString> >, TTree*> {
   private:
      std::map<TTree*, TDirectory*> fTreeDirectories; // an assotiation Tree <-> its File
   public:
      TTree* Acquire(const TString& fileName, const TString& dirName, const TString& treeName);
      ObjectAndBool_t Load(const TCacheKey &k);
      virtual void Unload(TTree* &tree);
   };

private:
   TString  fTreeName;  // name of the tree object to iterate over
   TTree   *fTree;      // tree we are iterating over
   TTreeCache fTreeCache;   // tree cache
   std::list<TTree*> fAcquiredTrees;   // a list of acquired trees.

   TTree* GetTrees(TDSetElement *elem);
   void   ReleaseAllTrees();
public:
   TEventIterTree();
   TEventIterTree(TDSet *dset, TSelector *sel, Long64_t first, Long64_t num);
   ~TEventIterTree();

   Long64_t GetNextEvent();

   ClassDef(TEventIterTree,0)  // Event iterator for Trees
};

#endif
