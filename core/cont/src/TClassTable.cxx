// @(#)root/cont:$Id$
// Author: Fons Rademakers   11/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TClassTable
\ingroup Containers
This class registers for all classes their name, id and dictionary
function in a hash table. Classes are automatically added by the
ctor of a special init class when a global of this init class is
initialized when the program starts (see the ClassImp macro).

All functions in TClassTable are thread-safe.
*/

#include "TClassTable.h"

#include "TClass.h"
#include "TClassEdit.h"
#include "TProtoClass.h"
#include "TList.h"
#include "TROOT.h"
#include "TString.h"
#include "TError.h"
#include "TRegexp.h"

#include "TObjString.h"
#include "TMap.h"

#include "TInterpreter.h"

#include <map>
#include <memory>
#include <typeinfo>
#include <cstdlib>
#include <string>
#include <mutex>
#include <unordered_map>

using namespace ROOT;

TClassTable *gClassTable;

TClassAlt           **TClassTable::fgAlternate;
TClassRec           **TClassTable::fgTable;
TClassRec           **TClassTable::fgSortedTable;
UInt_t                TClassTable::fgSize;
std::atomic<UInt_t>   TClassTable::fgTally;
Bool_t                TClassTable::fgSorted;
UInt_t                TClassTable::fgCursor;
TClassTable::IdMap_t *TClassTable::fgIdMap;

ClassImp(TClassTable);

static std::mutex &GetClassTableMutex()
{
   static std::mutex sMutex;
   return sMutex;
}

// RAII to first normalize the input classname (operation that
// both requires the ROOT global lock and might call `TClassTable`
// resursively) and then acquire a lock on `TClassTable` local
// mutex.
class TClassTable::NormalizeThenLock {
   std::string fNormalizedName;

public:
   NormalizeThenLock() = delete;
   NormalizeThenLock(const NormalizeThenLock&) = delete;
   NormalizeThenLock(NormalizeThenLock&&) = delete;
   NormalizeThenLock& operator=(const NormalizeThenLock&) = delete;
   NormalizeThenLock& operator=(NormalizeThenLock&&) = delete;

   NormalizeThenLock(const char *cname)
   {
      if (!TClassTable::CheckClassTableInit())
         return;

      // The recorded name is normalized, let's make sure we convert the
      // input accordingly.  This operation will take the ROOT global lock
      // and might call recursively `TClassTable`, so this must be done
      // outside of the `TClassTable` critical section.
      TClassEdit::GetNormalizedName(fNormalizedName, cname);

      GetClassTableMutex().lock();
   }

   ~NormalizeThenLock() {
      GetClassTableMutex().unlock();
   }

   const std::string &GetNormalizedName() const {
      return fNormalizedName;
   }
};

////////////////////////////////////////////////////////////////////////////////

namespace ROOT {
   class TClassRec {
   public:
      TClassRec(TClassRec *next) :
        fName(nullptr), fId(0), fDict(nullptr), fInfo(nullptr), fProto(nullptr), fNext(next)
      {}

      ~TClassRec() {
         // TClassTable::fgIdMap->Remove(r->fInfo->name());
         delete [] fName;
         delete fProto;
         delete fNext;
      }

      char            *fName;
      Version_t        fId;
      Int_t            fBits;
      DictFuncPtr_t    fDict;
      const std::type_info *fInfo;
      TProtoClass     *fProto;
      TClassRec       *fNext;
   };

   class TClassAlt {
   public:
      TClassAlt(const char*alternate, const char *normName, TClassAlt *next) :
         fName(alternate), fNormName(normName), fNext(next)
      {}

      ~TClassAlt() {
         // Nothing more to delete.
      }

      const char *fName;     // Do not own
      const char *fNormName; // Do not own
      std::unique_ptr<TClassAlt> fNext;
   };

#define R__USE_STD_MAP
   class TMapTypeToClassRec {
#if defined R__USE_STD_MAP
     // This wrapper class allow to avoid putting #include <map> in the
     // TROOT.h header file.
   public:
      typedef std::map<std::string, TClassRec*> IdMap_t;
      typedef IdMap_t::key_type                 key_type;
      typedef IdMap_t::const_iterator           const_iterator;
      typedef IdMap_t::size_type                size_type;
#ifdef R__WIN32
      // Window's std::map does NOT defined mapped_type
      typedef TClassRec*                        mapped_type;
#else
      typedef IdMap_t::mapped_type              mapped_type;
#endif

   private:
      IdMap_t fMap;

   public:
      void Add(const key_type &key, mapped_type &obj) {
         fMap[key] = obj;
      }

      mapped_type Find(const key_type &key) const {
         IdMap_t::const_iterator iter = fMap.find(key);
         mapped_type cl = nullptr;
         if (iter != fMap.end()) cl = iter->second;
         return cl;
      }

      void Remove(const key_type &key) { fMap.erase(key); }

      void Print() {
         Info("TMapTypeToClassRec::Print", "printing the typeinfo map in TClassTable");
         for (const_iterator iter = fMap.begin(); iter != fMap.end(); ++iter) {
            printf("Key: %40s 0x%zx\n", iter->first.c_str(), (size_t)iter->second);
         }
      }
#else
   private:
      TMap fMap;
   public:
#ifdef R__COMPLETE_MEM_TERMINATION
      ~TMapTypeToClassRec() {
         TIter next(&fMap);
         TObjString *key;
         while((key = (TObjString*)next())) {
            delete key;
         }
      }
#endif

      void Add(const char *key, TClassRec *&obj)
      {
         // Add <key,value> pair to the map.

         TObjString *realkey = new TObjString(key);
         fMap.Add(realkey, (TObject*)obj);
      }

      TClassRec *Find(const char *key) const {
         // Find the value corresponding the key.
         const TPair *a = (const TPair *)fMap.FindObject(key);
         if (a) return (TClassRec*) a->Value();
         return 0;
      }

      void Remove(const char *key) {
         // Remove the value corresponding the key.
         TObjString realkey(key);
         TObject *actual = fMap.Remove(&realkey);
         delete actual;
      }

      void Print() {
         // Print the content of the map.
         Info("TMapTypeToClassRec::Print", "printing the typeinfo map in TClassTable");
         TIter next(&fMap);
         TObjString *key;
         while((key = (TObjString*)next())) {
            printf("Key: %s\n",key->String().Data());
            TClassRec *data = (TClassRec*)fMap.GetValue(key);
            if (data) {
               printf("  class: %s %d\n",data->fName,data->fId);
            } else {
               printf("  no class: \n");
            }
         }
      }
#endif
   };

   static UInt_t ClassTableHash(const char *name, UInt_t size)
   {
      auto p = reinterpret_cast<const unsigned char*>( name );
      UInt_t slot = 0;

      while (*p) slot = slot<<1 ^ *p++;
      slot %= size;

      return slot;
   }

   std::vector<std::unique_ptr<TClassRec>> &GetDelayedAddClass()
   {
      static std::vector<std::unique_ptr<TClassRec>> delayedAddClass;
      return delayedAddClass;
   }

   std::vector<std::pair<const char *, const char *>> &GetDelayedAddClassAlternate()
   {
      static std::vector<std::pair<const char *, const char *>> delayedAddClassAlternate;
      return delayedAddClassAlternate;
   }
}

std::unordered_map<ROOT::TClassRec *, std::vector<ROOT::TClassAlt *>> &GetClassRecToAltMap()
{
   static std::unordered_map<ROOT::TClassRec *, std::vector<ROOT::TClassAlt *>> classRecToAltMap;
   return classRecToAltMap;
}

////////////////////////////////////////////////////////////////////////////////
/// TClassTable is a singleton (i.e. only one can exist per application).

TClassTable::TClassTable()
{
   if (gClassTable) return;

   fgSize  = 1009;  //this is the result of (int)TMath::NextPrime(1000);
   fgTable = new TClassRec* [fgSize];
   fgAlternate = new TClassAlt* [fgSize];
   fgIdMap = new IdMap_t;
   memset(fgTable, 0, fgSize * sizeof(TClassRec*));
   memset(fgAlternate, 0, fgSize * sizeof(TClassAlt *));
   gClassTable = this;

   for (auto &&r : GetDelayedAddClass()) {
      AddClass(r->fName, r->fId, *r->fInfo, r->fDict, r->fBits);
   };
   GetDelayedAddClass().clear();

   for (auto &&r : GetDelayedAddClassAlternate()) {
      AddAlternate(r.first, r.second);
   }
   GetDelayedAddClassAlternate().clear();
}

////////////////////////////////////////////////////////////////////////////////
/// TClassTable singleton is deleted in Terminate().

TClassTable::~TClassTable()
{
   // Try to avoid spurious warning from memory leak checkers.
   if (gClassTable != this) return;

   for (UInt_t i = 0; i < fgSize; i++) {
      delete fgTable[i]; // Will delete all the elements in the chain.
   }
   delete [] fgTable; fgTable = nullptr;
   delete [] fgSortedTable; fgSortedTable = nullptr;
   delete fgIdMap; fgIdMap = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Return true fs the table exist.
/// If the table does not exist but the delayed list does, then
/// create the table and return true.

inline Bool_t TClassTable::CheckClassTableInit()
{
   // This will be set at the lastest during TROOT construction, so before
   // any threading could happen.
   if (!gClassTable || !fgTable) {
      if (GetDelayedAddClass().size()) {
         new TClassTable;
         return kTRUE;
      }
      return kFALSE;
   }
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Print the class table. Before printing the table is sorted
/// alphabetically. Only classes specified in option are listed.
/// The default is to list all classes.
/// Standard wildcarding notation supported.

void TClassTable::Print(Option_t *option) const
{
   std::lock_guard<std::mutex> lock(GetClassTableMutex());

   // This is the very rare case (i.e. called before any dictionary load)
   // so we don't need to execute this outside of the critical section.
   if (fgTally == 0 || !fgTable)
      return;

   SortTable();

   int n = 0, ninit = 0, nl = 0;

   if (!option) option = "";
   int nch = strlen(option);
   TRegexp re(option, kTRUE);

   Printf("\nDefined classes");
   Printf("class                                 version  bits  initialized");
   Printf("================================================================");
   for (UInt_t i = 0; i < fgTally; i++) {
      TClassRec *r = fgSortedTable[i];
      if (!r) break;
      n++;
      TString s = r->fName;
      if (nch && strcmp(option,r->fName) && s.Index(re) == kNPOS) continue;
      nl++;
      if (TClass::GetClass(r->fName, kFALSE)) {
         ninit++;
         Printf("%-35s %6d %7d       Yes", r->fName, r->fId, r->fBits);
      } else
         Printf("%-35s %6d %7d       No",  r->fName, r->fId, r->fBits);
   }
   Printf("----------------------------------------------------------------");
   Printf("Listed Classes: %4d  Total classes: %4d   initialized: %4d",nl, n, ninit);
   Printf("================================================================\n");
}

//---- static members --------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
/// Returns class at index from sorted class table. Don't use this iterator
/// while modifying the class table. The class table can be modified
/// when making calls like TClass::GetClass(), etc.
/// Returns 0 if index points beyond last class name.

char *TClassTable::At(UInt_t index)
{
   if (index < fgTally) {
      std::lock_guard<std::mutex> lock(GetClassTableMutex());

      SortTable();
      TClassRec *r = fgSortedTable[index];
      if (r)
         return r->fName;
   }
   return nullptr;
}

//______________________________________________________________________________
int   TClassTable::Classes() { return fgTally; }
//______________________________________________________________________________
void  TClassTable::Init() { fgCursor = 0; SortTable(); }

namespace ROOT { class TForNamespace {}; } // Dummy class to give a typeid to namespace (see also TGenericClassInfo)

////////////////////////////////////////////////////////////////////////////////
/// Add a class to the class table (this is a static function).
/// Note that the given cname *must* be already normalized.

void TClassTable::Add(const char *cname, Version_t id,  const std::type_info &info,
                      DictFuncPtr_t dict, Int_t pragmabits)
{
   if (!cname || *cname == 0)
      ::Fatal("TClassTable::Add()", "Failed to deduce type for '%s'", info.name());

   // This will be set at the lastest during TROOT construction, so before
   // any threading could happen.
   if (!gClassTable)
      new TClassTable;

   std::unique_lock<std::mutex> lock(GetClassTableMutex());

   // check if already in table, if so return
   TClassRec *r = FindElement(cname, kTRUE);
   if (r->fName && r->fInfo) {
      if ( strcmp(r->fInfo->name(), typeid(ROOT::TForNamespace).name()) ==0
           && strcmp(info.name(), typeid(ROOT::TForNamespace).name()) ==0 ) {
         // We have a namespace being reloaded.
         // This okay we just keep the old one.
         return;
      }
      if (!TClassEdit::IsStdClass(cname)) {
         lock.unlock(); // Warning might recursively call TClassTable during gROOT init
         // Warn only for class that are not STD classes
         ::Warning("TClassTable::Add", "class %s already in TClassTable", cname);
      }
      return;
   } else if (ROOT::Internal::gROOTLocal && gCling) {
      TClass *oldcl = (TClass*)gROOT->GetListOfClasses()->FindObject(cname);
      if (oldcl) { //  && oldcl->GetClassInfo()) {
         // As a work-around to ROOT-6012, we need to register the class even if
         // it is not a template instance, because a forward declaration in the header
         // files loaded by the current dictionary wil also de-activate the update
         // class info mechanism!

         // The TClass exist and already has a class info, so it must
         // correspond to a class template instantiation which the interpreter
         // was able to make with the library containing the TClass Init.
         // Because it is already known to the interpreter, the update class info
         // will not be triggered, we need to force it.
         gCling->RegisterTClassUpdate(oldcl, dict);
      }
   }

   if (!r->fName)
      r->fName = StrDup(cname);
   r->fId   = id;
   r->fBits = pragmabits;
   r->fDict = dict;
   r->fInfo = &info;

   fgIdMap->Add(info.name(),r);

   fgSorted = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a class to the class table (this is a static function).
/// The caller of this function should be holding the ROOT Write lock.

void TClassTable::Add(TProtoClass *proto)
{
   // This will be set at the lastest during TROOT construction, so before
   // any threading could happen.
   if (!gClassTable)
      new TClassTable;

   std::unique_lock<std::mutex> lock(GetClassTableMutex());

   // By definition the name in the TProtoClass is (must be) the normalized
   // name, so there is no need to tweak it.
   const char *cname = proto->GetName();

   // check if already in table, if so return
   TClassRec *r = FindElement(cname, kTRUE);
   if (r->fName) {
      if (r->fProto) delete r->fProto;
      r->fProto = proto;
      TClass *oldcl = (TClass*)gROOT->GetListOfClasses()->FindObject(cname);

      lock.unlock(); // FillTClass might recursively call TClassTable during gROOT init
      if (oldcl && oldcl->GetState() == TClass::kHasTClassInit)
         proto->FillTClass(oldcl);
      return;
   } else if (ROOT::Internal::gROOTLocal && gCling) {
      TClass *oldcl = (TClass*)gROOT->GetListOfClasses()->FindObject(cname);
      if (oldcl) { //  && oldcl->GetClassInfo()) {
                   // As a work-around to ROOT-6012, we need to register the class even if
                   // it is not a template instance, because a forward declaration in the header
                   // files loaded by the current dictionary wil also de-activate the update
                   // class info mechanism!

         lock.unlock(); // Warning might recursively call TClassTable during gROOT init
         ::Warning("TClassTable::Add(TProtoClass*)","Called for existing class without a prior call add the dictionary function.");
      }
   }

   r->fName = StrDup(cname);
   r->fId   = 0;
   r->fBits = 0;
   r->fDict = nullptr;
   r->fInfo = nullptr;
   r->fProto= proto;

   fgSorted = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////

ROOT::TClassAlt* TClassTable::AddAlternate(const char *normName, const char *alternate)
{
   // This will be set at the lastest during TROOT construction, so before
   // any threading could happen.
   if (!gClassTable)
      new TClassTable;

   std::lock_guard<std::mutex> lock(GetClassTableMutex());

   UInt_t slot = ROOT::ClassTableHash(alternate, fgSize);

   for (const TClassAlt *a = fgAlternate[slot]; a; a = a->fNext.get()) {
      if (strcmp(alternate,a->fName)==0) {
         if (strcmp(normName,a->fNormName) != 0) {
            fprintf(stderr,"Error in TClassTable::AddAlternate: "
                    "Second registration of %s with a different normalized name (old: '%s', new: '%s')\n",
                    alternate, a->fNormName, normName);
         }
         return nullptr;
      }
   }

   fgAlternate[slot] = new TClassAlt(alternate,normName,fgAlternate[slot]);

   UInt_t slotNorm = ROOT::ClassTableHash(normName, fgSize);
   if (fgTable[slotNorm]) {
      auto &classToAlt = GetClassRecToAltMap();
      // Let others connect a class record to its class alternative names
      if (auto it = classToAlt.find(fgTable[slotNorm]); it == classToAlt.end())
         classToAlt[fgTable[slotNorm]] = std::vector<ROOT::TClassAlt *>{fgAlternate[slot]};
      else
         classToAlt[fgTable[slotNorm]].push_back(fgAlternate[slot]);
   }

   return fgAlternate[slot];
}

////////////////////////////////////////////////////////////////////////////////
///
void TClassTable::RemoveAlternate(ROOT::TClassAlt *alt)
{
   if (!alt || !gClassTable)
      return;

   std::lock_guard<std::mutex> lock(GetClassTableMutex());

   UInt_t slot = ROOT::ClassTableHash(alt->fName, fgSize);

   if (!fgAlternate[slot])
      return;

   if (fgAlternate[slot] == alt)
      fgAlternate[slot] = alt->fNext.release();
   else {
      for (TClassAlt *a = fgAlternate[slot]; a; a = a->fNext.get()) {
         if (a->fNext.get() == alt) {
            a->fNext.swap( alt->fNext );
	    assert( alt == alt->fNext.get());
	    alt->fNext.release();
	 }
      }
   }
   delete alt;
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TClassTable::Check(const char *cname, std::string &normname)
{
   if (!CheckClassTableInit())
      return kFALSE;

   std::lock_guard<std::mutex> lock(GetClassTableMutex());

   UInt_t slot = ROOT::ClassTableHash(cname, fgSize);

   // Check if 'cname' is a known normalized name.
   for (TClassRec *r = fgTable[slot]; r; r = r->fNext)
      if (strcmp(cname,r->fName)==0) return kTRUE;

   // See if 'cname' is register in the list of alternate names
   for (const TClassAlt *a = fgAlternate[slot]; a; a = a->fNext.get()) {
      if (strcmp(cname,a->fName)==0) {
         normname = a->fNormName;
         return kTRUE;
      }
   }

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove a class from the class table. This happens when a shared library
/// is unloaded (i.e. the dtor's of the global init objects are called).

void TClassTable::Remove(const char *cname)
{
   if (!CheckClassTableInit())
      return;

   std::lock_guard<std::mutex> lock(GetClassTableMutex());

   UInt_t slot = ROOT::ClassTableHash(cname, fgSize);

   TClassRec *r;
   TClassRec *prev = nullptr;
   for (r = fgTable[slot]; r; r = r->fNext) {
      if (!strcmp(r->fName, cname)) {
         if (prev)
            prev->fNext = r->fNext;
         else
            fgTable[slot] = r->fNext;
         fgIdMap->Remove(r->fInfo->name());
         r->fNext = nullptr; // Do not delete the others.
         delete r;
         fgTally--;
         fgSorted = kFALSE;
         break;
      }
      prev = r;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Find a class by name in the class table (using hash of name). Returns
/// 0 if the class is not in the table. Unless arguments insert is true in
/// which case a new entry is created and returned.
/// `cname` must be the normalized name of the class.

TClassRec *TClassTable::FindElement(const char *cname, Bool_t insert)
{
   // Internal routine, no explicit lock needed here.

   UInt_t slot = ROOT::ClassTableHash(cname, fgSize);

   for (TClassRec *r = fgTable[slot]; r; r = r->fNext)
      if (strcmp(cname, r->fName) == 0)
         return r;

   if (!insert)
      return nullptr;

   fgTable[slot] = new TClassRec(fgTable[slot]);

   fgTally++;
   return fgTable[slot];
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the ID of a class.

Version_t TClassTable::GetID(const char *cname)
{
   NormalizeThenLock guard(cname);

   TClassRec *r = FindElement(guard.GetNormalizedName().c_str(), kFALSE);
   if (r)
      return r->fId;
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the pragma bits as specified in the LinkDef.h file.

Int_t TClassTable::GetPragmaBits(const char *cname)
{
   NormalizeThenLock guard(cname);

   TClassRec *r = FindElement(guard.GetNormalizedName().c_str(), kFALSE);
   if (r)
      return r->fBits;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Given the class name returns the Dictionary() function of a class
/// (uses hash of name).

DictFuncPtr_t TClassTable::GetDict(const char *cname)
{
   if (gDebug > 9) {
      ::Info("GetDict", "searches for %s", cname);
      fgIdMap->Print();
   }
   NormalizeThenLock guard(cname);

   TClassRec *r = FindElement(guard.GetNormalizedName().c_str(), kFALSE);
   if (r)
      return r->fDict;
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Given the std::type_info returns the Dictionary() function of a class
/// (uses hash of std::type_info::name()).

DictFuncPtr_t TClassTable::GetDict(const std::type_info& info)
{
   if (!CheckClassTableInit())
      return nullptr;

   if (gDebug > 9)
      ROOT::GetROOT(); // Info might recursively call TClassTable during the gROOT init

   std::lock_guard<std::mutex> lock(GetClassTableMutex());

   if (gDebug > 9) {
      ::Info("GetDict", "searches for %s at 0x%zx", info.name(), (size_t)&info);
      fgIdMap->Print();
   }

   TClassRec *r = fgIdMap->Find(info.name());
   if (r)
      return r->fDict;
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Given the normalized class name returns the Dictionary() function of a class
/// (uses hash of name).

DictFuncPtr_t TClassTable::GetDictNorm(const char *cname)
{
   if (!CheckClassTableInit())
      return nullptr;

   if (gDebug > 9)
      ROOT::GetROOT(); // Info might recursively call TClassTable during the gROOT init

   std::lock_guard<std::mutex> lock(GetClassTableMutex());

   if (gDebug > 9) {
      ::Info("GetDict", "searches for %s", cname);
      fgIdMap->Print();
   }

   TClassRec *r = FindElement(cname, kFALSE);
   if (r)
      return r->fDict;
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Given the class name returns the TClassProto object for the class.
/// (uses hash of name).

TProtoClass *TClassTable::GetProto(const char *cname)
{
   if (gDebug > 9) {
      ::Info("GetDict", "searches for %s", cname);
   }

   if (!CheckClassTableInit())
      return nullptr;

   NormalizeThenLock guard(cname);

   if (gDebug > 9) {
      // Because of the early call to Info, gROOT is already initialized
      // and thus this will not cause a recursive call to TClassTable.
      ::Info("GetDict", "searches for %s", cname);
      fgIdMap->Print();
   }

   TClassRec *r = FindElement(guard.GetNormalizedName().c_str(), kFALSE);
   if (r)
      return r->fProto;
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Given the class normalized name returns the TClassProto object for the class.
/// (uses hash of name).

TProtoClass *TClassTable::GetProtoNorm(const char *cname)
{
   if (gDebug > 9) {
      ::Info("GetDict", "searches for %s", cname);
   }

   if (!CheckClassTableInit())
      return nullptr;

   std::lock_guard<std::mutex> lock(GetClassTableMutex());

   if (gDebug > 9) {
      fgIdMap->Print();
   }

   TClassRec *r = FindElement(cname, kFALSE);
   if (r)
      return r->fProto;
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////

extern "C" {
   static int ClassComp(const void *a, const void *b)
   {
      // Function used for sorting classes alphabetically.

      return strcmp((*(TClassRec **)a)->fName, (*(TClassRec **)b)->fName);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Returns next class from sorted class table. Don't use this iterator
/// while modifying the class table. The class table can be modified
/// when making calls like TClass::GetClass(), etc.

char *TClassTable::Next()
{
   std::lock_guard<std::mutex> lock(GetClassTableMutex());

   if (fgCursor < fgTally) {
      TClassRec *r = fgSortedTable[fgCursor++];
      return r->fName;
   }

   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Print the class table. Before printing the table is sorted
/// alphabetically.

void TClassTable::PrintTable()
{
   if (fgTally == 0 || !fgTable)
      return;

   std::lock_guard<std::mutex> lock(GetClassTableMutex());

   SortTable();

   int n = 0, ninit = 0;

   Printf("\nDefined classes");
   Printf("class                                 version  bits  initialized");
   Printf("================================================================");
   UInt_t last = fgTally;
   for (UInt_t i = 0; i < last; i++) {
      TClassRec *r = fgSortedTable[i];
      if (!r) break;
      n++;
      // Do not use TClass::GetClass to avoid any risk of autoloading.
      if (gROOT->GetListOfClasses()->FindObject(r->fName)) {
         ninit++;
         Printf("%-35s %6d %7d       Yes", r->fName, r->fId, r->fBits);
      } else
         Printf("%-35s %6d %7d       No",  r->fName, r->fId, r->fBits);
   }
   Printf("----------------------------------------------------------------");
   Printf("Total classes: %4d   initialized: %4d", n, ninit);
   Printf("================================================================\n");
}

////////////////////////////////////////////////////////////////////////////////
/// Sort the class table by ascending class ID's.

void TClassTable::SortTable()
{
   // Internal routine.

   if (!fgSorted) {
      delete [] fgSortedTable;
      fgSortedTable = new TClassRec* [fgTally];

      int j = 0;
      for (UInt_t i = 0; i < fgSize; i++)
         for (TClassRec *r = fgTable[i]; r; r = r->fNext)
            fgSortedTable[j++] = r;

      ::qsort(fgSortedTable, fgTally, sizeof(TClassRec *), ::ClassComp);
      fgSorted = kTRUE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Deletes the class table (this static class function calls the dtor).

void TClassTable::Terminate()
{
   if (gClassTable) {
      for (UInt_t i = 0; i < fgSize; i++)
         delete fgTable[i]; // Will delete all the elements in the chain.

      delete [] fgTable; fgTable = nullptr;
      delete [] fgSortedTable; fgSortedTable = nullptr;
      delete fgIdMap; fgIdMap = nullptr;
      fgSize = 0;
      SafeDelete(gClassTable);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Global function called by the ctor of a class's init class
/// (see the ClassImp macro).

void ROOT::AddClass(const char *cname, Version_t id,
                    const std::type_info& info,
                    DictFuncPtr_t dict,
                    Int_t pragmabits)
{
   if (!TROOT::Initialized() && !gClassTable) {
      auto r = std::unique_ptr<TClassRec>(new TClassRec(nullptr));
      r->fName = StrDup(cname);
      r->fId   = id;
      r->fBits = pragmabits;
      r->fDict = dict;
      r->fInfo = &info;
      GetDelayedAddClass().emplace_back(std::move(r));
   } else {
      TClassTable::Add(cname, id, info, dict, pragmabits);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Global function called by GenerateInitInstance.
/// (see the ClassImp macro).

ROOT::TClassAlt* ROOT::AddClassAlternate(const char *normName, const char *alternate)
{
   if (!TROOT::Initialized() && !gClassTable) {
      GetDelayedAddClassAlternate().emplace_back(normName, alternate);
      // If a library is loaded before gROOT is initialized we can assume
      // it is hard linked along side libCore (or is libCore) thus can't
      // really be unloaded.
      return nullptr;
   } else {
      return TClassTable::AddAlternate(normName, alternate);
   }
}

void ROOT::RemoveClassAlternate(TClassAlt *alt)
{
   // This routine is meant to be called (indirectly) by dlclose so we
   // we are guaranteed that the library initialization has completed.
   TClassTable::RemoveAlternate(alt);
}

////////////////////////////////////////////////////////////////////////////////
/// Global function to update the version number.
/// This is called via the RootClassVersion macro.
///
/// if cl!=0 and cname==-1, set the new class version if and only is
/// greater than the existing one and greater or equal to 2;
/// and also ignore the request if fVersionUsed is true.
///
/// Note on class version number:
///  - If no class has been specified, TClass::GetVersion will return -1
///  - The Class Version 0 request the whole object to be transient
///  - The Class Version 1, unless specify via ClassDef indicates that the
///    I/O should use the TClass checksum to distinguish the layout of the class
void ROOT::ResetClassVersion(TClass *cl, const char *cname, Short_t newid)
{
   if (cname && cname != (void*)-1 && TClassTable::CheckClassTableInit()) {
      TClassTable::NormalizeThenLock guard(cname);
      TClassRec *r = TClassTable::FindElement(guard.GetNormalizedName().c_str(), kFALSE);
      if (r)
         r->fId = newid;
   }
   if (cl) {
      if (cl->fVersionUsed) {
         // Problem, the reset is called after the first usage!
         if (cname!=(void*)-1)
            Error("ResetClassVersion","Version number of %s can not be changed after first usage!",
                  cl->GetName());
      } else {
         if (newid < 0) {
            Error("SetClassVersion","The class version (for %s) must be positive (value %d is ignored)",cl->GetName(),newid);
         }
         if (cname==(void*)-1) {
            if (cl->fClassVersion<newid && 2<=newid) {
               cl->SetClassVersion(newid);
            }
         } else {
            cl->SetClassVersion(newid);
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Global function called by the dtor of a class's init class
/// (see the ClassImp macro).
/// The caller of this function should be holding the ROOT Write lock.

void ROOT::RemoveClass(const char *cname, TClass *oldcl)
{
   // don't delete class information since it is needed by the I/O system
   // to write the StreamerInfo to file
   if (cname) {
      // Let's still remove this information to allow reloading later.
      // Anyway since the shared library has been unloaded, the dictionary
      // pointer is now invalid ....
      // We still keep the TClass object around because TFile needs to
      // get to the TStreamerInfo.
      if (oldcl)
         oldcl->SetUnloaded();
      TClassTable::Remove(cname);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Global function to register the implementation file and line of
/// a class template (i.e. NOT a concrete class).

TNamed *ROOT::RegisterClassTemplate(const char *name, const char *file,
                                    Int_t line)
{
   static TList table;
   static Bool_t isInit = []() {
      table.SetOwner(kTRUE);
      table.UseRWLock();
      return true;
   }();
   (void)isInit;

   TString classname(name);
   Ssiz_t loc = classname.Index("<");
   if (loc >= 1)
      classname.Remove(loc);
   TNamed *reg = (TNamed*)table.FindObject(classname);
   if (file) {
      if (reg)
         reg->SetTitle(file);
      else {
         reg = new TNamed((const char*)classname, file);
         table.Add(reg);
      }
      reg->SetUniqueID(line);
   }
   return reg;
}

std::vector<std::string> TClassTable::GetClassAlternativeNames(const char *cname)
{
   std::lock_guard<std::mutex> lock(GetClassTableMutex());

   UInt_t slot = ROOT::ClassTableHash(cname, fgSize);
   if (!fgTable[slot])
      return {};
   const auto &classRecToAltMap = GetClassRecToAltMap();
   if (auto it = classRecToAltMap.find(fgTable[slot]); it == classRecToAltMap.end())
      return {};

   const auto &classAlts = classRecToAltMap.at(fgTable[slot]);
   std::vector<std::string> ret;
   ret.reserve(classAlts.size());
   for (const auto *classAlt : classAlts) {
      ret.push_back(classAlt->fName);
   }

   return ret;
}
