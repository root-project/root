// @(#)root/cont:$Id$
// Author: Fons Rademakers   11/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// This class registers for all classes their name, id and dictionary   //
// function in a hash table. Classes are automatically added by the     //
// ctor of a special init class when a global of this init class is     //
// initialized when the program starts (see the ClassImp macro).        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "RConfig.h"
#include <stdlib.h>
#include <string>
#include <map>
#include <typeinfo>
#include "Riostream.h"

#include "TClassTable.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TROOT.h"
#include "TString.h"
#include "TError.h"
#include "TRegexp.h"

#include "TObjString.h"
#include "TMap.h"

TClassTable *gClassTable;

TClassRec  **TClassTable::fgTable;
TClassRec  **TClassTable::fgSortedTable;
int          TClassTable::fgSize;
int          TClassTable::fgTally;
Bool_t       TClassTable::fgSorted;
int          TClassTable::fgCursor;
TClassTable::IdMap_t *TClassTable::fgIdMap;

ClassImp(TClassTable)

//______________________________________________________________________________
namespace ROOT {

   class TMapTypeToClassRec {
#if defined R__USE_STD_MAP
     // This wrapper class allow to avoid putting #include <map> in the
     // TROOT.h header file.
   public:
#ifdef R__GLOBALSTL
      typedef map<string, TClassRec*>           IdMap_t;
#else
      typedef std::map<std::string, TClassRec*> IdMap_t;
#endif
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
         mapped_type cl = 0;
         if (iter != fMap.end()) cl = iter->second;
         return cl;
      }

      void Remove(const key_type &key) { fMap.erase(key); }

      void Print() {
         Info("TMapTypeToClassRec::Print", "printing the typeinfo map in TClassTable");
         for (const_iterator iter = fMap.begin(); iter != fMap.end(); iter++) {
            printf("Key: %40s 0x%lx\n", iter->first.c_str(), iter->second);
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
}

//______________________________________________________________________________
TClassTable::TClassTable()
{
   // TClassTable is a singleton (i.e. only one can exist per application).

   if (gClassTable) return;

   fgSize  = 1009;  //this is thge result of (int)TMath::NextPrime(1000);
   fgTable = new TClassRec* [fgSize];
   fgIdMap = new IdMap_t;
   memset(fgTable, 0, fgSize*sizeof(TClassRec*));
   gClassTable = this;
}

//______________________________________________________________________________
TClassTable::~TClassTable()
{
   // TClassTable singleton is deleted in Terminate().

   // Try to avoid spurrious warning from memory leak checkers.
   if (gClassTable != this) return;

   for (Int_t i = 0; i < fgSize; i++) {
      TClassRec *r = fgTable[i];
      while (r) {
         delete [] r->fName;
         TClassRec *next = r->fNext;
         delete r;
         r = next;
      }
   }
   delete [] fgTable; fgTable = 0;
   delete [] fgSortedTable; fgSortedTable = 0;
   delete fgIdMap; fgIdMap = 0;
}

//______________________________________________________________________________
void TClassTable::Print(Option_t *option) const
{
   // Print the class table. Before printing the table is sorted
   // alphabetically. Only classes specified in option are listed.
   // The default is to list all classes.
   // Standard wilcarding notation supported.

   if (fgTally == 0 || !fgTable)
      return;

   SortTable();

   int n = 0, ninit = 0, nl = 0;

   int nch = strlen(option);
   TRegexp re(option, kTRUE);

   Printf("\nDefined classes");
   Printf("class                                 version  bits  initialized");
   Printf("================================================================");
   for (int i = 0; i < fgTally; i++) {
      if (!fgTable[i]) continue;
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

//______________________________________________________________________________
char *TClassTable::At(int index)
{
    // Returns class at index from sorted class table. Don't use this iterator
    // while modifying the class table. The class table can be modified
    // when making calls like TClass::GetClass(), etc.
    // Returns 0 if index points beyond last class name.

   SortTable();
   if (index >= 0 && index < fgTally) {
      TClassRec *r = fgSortedTable[index];
      if (r) return r->fName;
   }
   return 0;
}

//______________________________________________________________________________
int   TClassTable::Classes() { return fgTally; }
//______________________________________________________________________________
void  TClassTable::Init() { fgCursor = 0; SortTable(); }

namespace ROOT { class TForNamespace {}; } // Dummy class to give a typeid to namespace (see also TGenericClassInfo)

//______________________________________________________________________________
void TClassTable::Add(const char *cname, Version_t id,  const type_info &info,
                      VoidFuncPtr_t dict, Int_t pragmabits)
{
   // Add a class to the class table (this is a static function).

   if (!gClassTable)
      new TClassTable;

   // Only register the name without the default STL template arguments ...
   TClassEdit::TSplitType splitname( cname, TClassEdit::kLong64 );
   std::string shortName;
   splitname.ShortType(shortName, TClassEdit::kDropStlDefault);

   // check if already in table, if so return
   TClassRec *r = FindElementImpl(shortName.c_str(), kTRUE);
   if (r->fName) {
      if ( strcmp(r->fInfo->name(),typeid(ROOT::TForNamespace).name())==0
           && strcmp(info.name(),typeid(ROOT::TForNamespace).name())==0 ) {
         // We have a namespace being reloaded.
         // This okay we just keep the old one.
         return;
      }
      if (splitname.IsSTLCont()==0) {
         // Warn only for class that are not STL containers.
         ::Warning("TClassTable::Add", "class %s already in TClassTable", cname);
      }
      return;
   }

   r->fName = StrDup(shortName.c_str());
   r->fId   = id;
   r->fBits = pragmabits;
   r->fDict = dict;
   r->fInfo = &info;

   fgIdMap->Add(info.name(),r);

   fgTally++;
   fgSorted = kFALSE;
}

//______________________________________________________________________________
void TClassTable::Remove(const char *cname)
{
   // Remove a class from the class table. This happens when a shared library
   // is unloaded (i.e. the dtor's of the global init objects are called).

   if (!gClassTable || !fgTable) return;

   int slot = 0;
   const char *p = cname;

   while (*p) slot = slot<<1 ^ *p++;
   if (slot < 0) slot = -slot;
   slot %= fgSize;

   TClassRec *r;
   TClassRec *prev = 0;
   for (r = fgTable[slot]; r; r = r->fNext) {
      if (!strcmp(r->fName, cname)) {
         if (prev)
            prev->fNext = r->fNext;
         else
            fgTable[slot] = r->fNext;
         fgIdMap->Remove(r->fInfo->name());
         delete [] r->fName;
         delete r;
         fgTally--;
         fgSorted = kFALSE;
         break;
      }
      prev = r;
   }
}

//______________________________________________________________________________
TClassRec *TClassTable::FindElementImpl(const char *cname, Bool_t insert)
{
   // Find a class by name in the class table (using hash of name). Returns
   // 0 if the class is not in the table. Unless arguments insert is true in
   // which case a new entry is created and returned.

   int slot = 0;
   const char *p = cname;

   while (*p) slot = slot<<1 ^ *p++;
   if (slot < 0) slot = -slot;
   slot %= fgSize;

   TClassRec *r;

   for (r = fgTable[slot]; r; r = r->fNext)
      if (strcmp(cname,r->fName)==0) return r;

   if (!insert) return 0;

   r = new TClassRec;
   r->fName = 0;
   r->fId   = 0;
   r->fDict = 0;
   r->fInfo = 0;
   r->fNext = fgTable[slot];
   fgTable[slot] = r;

   return r;
}

//______________________________________________________________________________
TClassRec *TClassTable::FindElement(const char *cname, Bool_t insert)
{
   // Find a class by name in the class table (using hash of name). Returns
   // 0 if the class is not in the table. Unless arguments insert is true in
   // which case a new entry is created and returned.

   if (!fgTable) return 0;

   // Only register the name without the default STL template arguments ...
   TClassEdit::TSplitType splitname( cname, TClassEdit::kLong64 );
   std::string shortName;
   splitname.ShortType(shortName, TClassEdit::kDropStlDefault);

   return FindElementImpl(shortName.c_str(), insert);
}

//______________________________________________________________________________
Version_t TClassTable::GetID(const char *cname)
{
   // Returns the ID of a class.

   TClassRec *r = FindElement(cname);
   if (r) return r->fId;
   return -1;
}

//______________________________________________________________________________
Int_t TClassTable::GetPragmaBits(const char *cname)
{
   // Returns the pragma bits as specified in the LinkDef.h file.

   TClassRec *r = FindElement(cname);
   if (r) return r->fBits;
   return 0;
}

//______________________________________________________________________________
VoidFuncPtr_t TClassTable::GetDict(const char *cname)
{
   // Given the class name returns the Dictionary() function of a class
   // (uses hash of name).

   if (gDebug > 9) {
      ::Info("GetDict", "searches for %s", cname);
      fgIdMap->Print();
   }

   TClassRec *r = FindElement(cname);
   if (r) return r->fDict;
   return 0;
}

//______________________________________________________________________________
VoidFuncPtr_t TClassTable::GetDict(const type_info& info)
{
   // Given the type_info returns the Dictionary() function of a class
   // (uses hash of type_info::name()).

   if (gDebug > 9) {
      ::Info("GetDict", "searches for %s at 0x%lx", info.name(), (Long_t)&info);
      fgIdMap->Print();
   }

   TClassRec *r = fgIdMap->Find(info.name());
   if (r) return r->fDict;
   return 0;
}


//______________________________________________________________________________
extern "C" {
   static int ClassComp(const void *a, const void *b)
   {
      // Function used for sorting classes alphabetically.

      return strcmp((*(TClassRec **)a)->fName, (*(TClassRec **)b)->fName);
   }
}

//______________________________________________________________________________
char *TClassTable::Next()
{
    // Returns next class from sorted class table. Don't use this iterator
    // while modifying the class table. The class table can be modified
    // when making calls like TClass::GetClass(), etc.

   if (fgCursor < fgTally) {
      TClassRec *r = fgSortedTable[fgCursor++];
      return r->fName;
   } else
      return 0;
}

//______________________________________________________________________________
void TClassTable::PrintTable()
{
   // Print the class table. Before printing the table is sorted
   // alphabetically.

   if (fgTally == 0 || !fgTable)
      return;

   SortTable();

   int n = 0, ninit = 0;

   Printf("\nDefined classes");
   Printf("class                                 version  bits  initialized");
   Printf("================================================================");
   for (int i = 0; i < fgTally; i++) {
      if (!fgTable[i]) continue;
      TClassRec *r = fgSortedTable[i];
      if (!r) break;
      n++;
      if (TClass::GetClass(r->fName, kFALSE)) {
         ninit++;
         Printf("%-35s %6d %7d       Yes", r->fName, r->fId, r->fBits);
      } else
         Printf("%-35s %6d %7d       No",  r->fName, r->fId, r->fBits);
   }
   Printf("----------------------------------------------------------------");
   Printf("Total classes: %4d   initialized: %4d", n, ninit);
   Printf("================================================================\n");
}

//______________________________________________________________________________
void TClassTable::SortTable()
{
   // Sort the class table by ascending class ID's.

   if (!fgSorted) {
      delete [] fgSortedTable;
      fgSortedTable = new TClassRec* [fgTally];

      int j = 0;
      for (int i = 0; i < fgSize; i++)
         for (TClassRec *r = fgTable[i]; r; r = r->fNext)
            fgSortedTable[j++] = r;

      ::qsort(fgSortedTable, fgTally, sizeof(TClassRec *), ::ClassComp);
      fgSorted = kTRUE;
   }
}

//______________________________________________________________________________
void TClassTable::Terminate()
{
   // Deletes the class table (this static class function calls the dtor).

   if (gClassTable) {
      for (int i = 0; i < fgSize; i++)
         for (TClassRec *r = fgTable[i]; r; ) {
            TClassRec *t = r;
            r = r->fNext;
            fgIdMap->Remove(r->fInfo->name());
            delete [] t->fName;
            delete t;
         }
      delete [] fgTable; fgTable = 0;
      delete [] fgSortedTable; fgSortedTable = 0;
      delete fgIdMap; fgIdMap = 0;
      fgSize = 0;
      SafeDelete(gClassTable);
   }
}

//______________________________________________________________________________
void ROOT::AddClass(const char *cname, Version_t id,
                    const type_info& info,
                    VoidFuncPtr_t dict,
                    Int_t pragmabits)
{
   // Global function called by the ctor of a class's init class
   // (see the ClassImp macro).

   TClassTable::Add(cname, id, info, dict, pragmabits);
}

//______________________________________________________________________________
void ROOT::ResetClassVersion(TClass *cl, const char *cname, Short_t newid)
{
   // Global function to update the version number.
   // This is called via the RootClassVersion macro.
   //
   // if cl!=0 and cname==-1, set the new class version if and only is
   // greater than the existing one and greater or equal to 2;
   // and also ignore the request if fVersionUsed is true.
   //
   // Note on class version number:
   //   If no class has been specified, TClass::GetVersion will return -1
   //   The Class Version 0 request the whole object to be transient
   //   The Class Version 1, unless specify via ClassDef indicates that the
   //      I/O should use the TClass checksum to distinguish the layout of the class

   if (cname && cname!=(void*)-1) {
      TClassRec *r = TClassTable::FindElement(cname,kFALSE);
      if (r) r->fId = newid;
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


//______________________________________________________________________________
void ROOT::RemoveClass(const char *cname)
{
   // Global function called by the dtor of a class's init class
   // (see the ClassImp macro).

   // don't delete class information since it is needed by the I/O system
   // to write the StreamerInfo to file
   if (cname) {
      // Let's still remove this information to allow reloading later.
      // Anyway since the shared library has been unloaded, the dictionary
      // pointer is now invalid ....
      // We still keep the TClass object around because TFile needs to
      // get to the TStreamerInfo.
      if (gROOT && gROOT->GetListOfClasses()) {
         TObject *pcname;
         if ((pcname=gROOT->GetListOfClasses()->FindObject(cname))) {
            TClass *cl = dynamic_cast<TClass*>(pcname);
            if (cl) cl->SetUnloaded();
         }
      }
      TClassTable::Remove(cname);
   }
}

//______________________________________________________________________________
TNamed *ROOT::RegisterClassTemplate(const char *name, const char *file,
                                    Int_t line)
{
   // Global function to register the implementation file and line of
   // a class template (i.e. NOT a concrete class).

   static TList table;
   static Bool_t isInit = kFALSE;
   if (!isInit) {
      table.SetOwner(kTRUE);
      isInit = kTRUE;
   }

   TString classname(name);
   Ssiz_t loc = classname.Index("<");
   if (loc >= 1) classname.Remove(loc);
   if (file) {
      TNamed *obj = new TNamed((const char*)classname, file);
      obj->SetUniqueID(line);
      table.Add(obj);
      return obj;
   } else {
      return (TNamed*)table.FindObject(classname);
   }
}
