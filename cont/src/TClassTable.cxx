// @(#)root/cont:$Name:  $:$Id: TClassTable.cxx,v 1.16 2002/05/10 11:07:22 rdm Exp $
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

#include <stdlib.h>
#include <string>
#include <map>
#include "Riostream.h"

#include "TClassTable.h"
#include "TClass.h"
#include "TROOT.h"
#include "TMath.h"
#include "TString.h"
#include "TError.h"
#include "TRegexp.h"

#include "TObjString.h"
#include "TMap.h"

TClassTable *gClassTable;

ClassRec_t **TClassTable::fgTable;
ClassRec_t **TClassTable::fgSortedTable;
int          TClassTable::fgSize;
int          TClassTable::fgTally;
Bool_t       TClassTable::fgSorted;
int          TClassTable::fgCursor;
TClassTable::IdMap_t *TClassTable::fgIdMap;

ClassImp(TClassTable)

//______________________________________________________________________________
namespace ROOT {
   //#define R__USE_STD_MAP
   class TMapTypeToClassRec {
#if defined R__USE_STD_MAP
     // This wrapper class allow to avoid putting #include <map> in the
     // TROOT.h header file.
   public:
#ifdef R__GLOBALSTL
      typedef map<string,ClassRec_t*> IdMap_t;
#else
      typedef std::map<std::string,ClassRec_t*> IdMap_t;
#endif
      typedef IdMap_t::key_type                 key_type;
      typedef IdMap_t::const_iterator           const_iterator;
      typedef IdMap_t::size_type                size_type;
#ifdef R__WIN32
      // Window's std::map does NOT defined mapped_type
      typedef ClassRec_t*                       mapped_type;
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

      void printall() {
         cerr << "Printing the typeinfo map in TClassTable\n";
         for (const_iterator iter = fMap.begin();
              iter != fMap.end();
              iter++) {
            cerr << "Key: " << iter->first.c_str()
                 << " points to " << iter->second << endl;
         }
      }
#else
   private:
      TMap fMap;
   public:
      void Add(const char* key, ClassRec_t *&obj) {
         TObjString *realkey = new TObjString(key);
         fMap.Add(realkey, (TObject*)obj);
      }
      ClassRec_t* Find(const char* key) const {
         const TAssoc* a = (const TAssoc *)fMap.FindObject(key);
         if (a) return (ClassRec_t*) a->Value();
         return 0;
      }
      void Remove(const char* key) {
         TObjString realkey(key);
         TObject *actual = fMap.Remove(&realkey);
         delete actual;
      }

#endif
   };
}

//______________________________________________________________________________
TClassTable::TClassTable()
{
   // TClassTable is a singleton (i.e. only one can exist per application).

   if (gClassTable)
      Error("TClassTable", "only one instance of TClassTable allowed");
   fgSize  = (int)TMath::NextPrime(1000);
   fgTable = new ClassRec_t* [fgSize];
   fgIdMap = new IdMap_t;
   memset(fgTable, 0, fgSize*sizeof(ClassRec_t*));
}

//______________________________________________________________________________
TClassTable::~TClassTable()
{
   // TClassTable singleton is deleted in Terminate().
}

//______________________________________________________________________________
void TClassTable::Print(Option_t *option) const
{
   // Print the class table. Before printing the table is sorted
   // alphabetically. Only classes specified in option are listed.
   // default is to list all classes.
   // Standard wilcarding notation supported.

   if (fgTally == 0 || !fgTable)
      return;

   SortTable();

   int n = 0, ninit = 0, nl = 0;

   int nch = strlen(option);
   TRegexp re(option,kTRUE);
   Printf("");
   Printf("Defined classes");
   Printf("class                              version  bits  initialized");
   Printf("=============================================================");
   for (int i = 0; i < fgTally; i++) {
      ClassRec_t *r = fgSortedTable[i];
      n++;
      TString s = r->name;
      if (nch && strcmp(option,r->name) && s.Index(re) == kNPOS) continue;
      nl++;
      if (gROOT->GetClass(r->name, kFALSE)) {
         ninit++;
         Printf("%-32s %6d %7d       Yes", r->name, r->id, r->bits);
      } else
         Printf("%-32s %6d %7d       No",  r->name, r->id, r->bits);
   }
   Printf("-------------------------------------------------------------");
   Printf("Listed Classes: %4d  Total classes: %4d   initialized: %4d",nl, n, ninit);
   Printf("=============================================================");

   Printf("");
}

//---- static members --------------------------------------------------------

//______________________________________________________________________________
int   TClassTable::Classes() { return fgTally; }
//______________________________________________________________________________
void  TClassTable::Init() { fgCursor = 0; SortTable(); }


//______________________________________________________________________________
void TClassTable::Add(const char *cname, Version_t id,  const type_info &info,
                      VoidFuncPtr_t dict, Int_t pragmabits)
{
   // Add a class to the class table (this is a static function).

   if (!gClassTable)
      gClassTable = new TClassTable;

   // check if already in table, if so return
   ClassRec_t *r = FindElement(cname, kTRUE);
   if (r->name) {
      ::Warning("TClassTable::Add", "class %s allready in TClassTable", cname);
      return;
   }

   r->name = StrDup(cname);
   r->id   = id;
   r->bits = pragmabits;
   r->dict = dict;
   r->info = &info;

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

   ClassRec_t *r;
   ClassRec_t *prev = 0;
   for (r = fgTable[slot]; r; r = r->next) {
      if (!strcmp(r->name, cname)) {
         if (prev)
            prev->next = r->next;
         else
            fgTable[slot] = r->next;
         fgIdMap->Remove(r->info->name());
         delete [] r->name;
         delete r;
         fgTally--;
         fgSorted = kFALSE;
         break;
      }
      prev = r;
   }
}

//______________________________________________________________________________
ClassRec_t *TClassTable::FindElement(const char *cname, Bool_t insert)
{
   // Find a class by name in the class table (using hash of name). Returns
   // 0 if the class is not in the table. Unless arguments insert is true in
   // which case a new entry is created and returned.

   if (!fgTable) return 0;

   int slot = 0;
   const char *p = cname;

   while (*p) slot = slot<<1 ^ *p++;
   if (slot < 0) slot = -slot;
   slot %= fgSize;

   ClassRec_t *r;

   for (r = fgTable[slot]; r; r = r->next)
      if (!strcmp(r->name, cname)) return r;

   if (!insert) return 0;

   r = new ClassRec_t;
   r->name = 0;
   r->id   = 0;
   r->dict = 0;
   r->info = 0;
   r->next = fgTable[slot];
   fgTable[slot] = r;

   return r;
}

//______________________________________________________________________________
Version_t TClassTable::GetID(const char *cname)
{
   // Returns the ID of a class.

   ClassRec_t *r = FindElement(cname);
   if (r) return r->id;
   return -1;
}

//______________________________________________________________________________
Int_t TClassTable::GetPragmaBits(const char *cname)
{
   // Returns the pragma bits as specified in the LinkDef.h file.

   ClassRec_t *r = FindElement(cname);
   if (r) return r->bits;
   return 0;
}

//______________________________________________________________________________
VoidFuncPtr_t TClassTable::GetDict(const char *cname)
{
   // Given the class name returns the Dictionary() function of a class
   // (uses hash of name).

   ClassRec_t *r = FindElement(cname);
   if (r) return r->dict;
   return 0;
}

//______________________________________________________________________________
VoidFuncPtr_t TClassTable::GetDict(const type_info& info)
{
   // Given the class name returns the Dictionary() function of a class
   // (uses hash of name).

#ifdef DEBUG_ID
   cerr << "While Table searches for " << info.name() << " at " << &info << endl;
   fgIdMap->printall();
#endif

   ClassRec_t *r = fgIdMap->Find(info.name());
   if (r) return r->dict;
   return 0;
}


//______________________________________________________________________________
extern "C" {
   static int ClassComp(const void *a, const void *b)
   {
      // Function used for sorting classes alphabetically.

      return strcmp((*(ClassRec_t **)a)->name, (*(ClassRec_t **)b)->name);
   }
}

//______________________________________________________________________________
char *TClassTable::Next()
{
    // Returns next class from sorted class table.

    if (fgCursor < fgTally) {
       ClassRec_t *r = fgSortedTable[fgCursor++];
       return r->name;
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

   Printf("");
   Printf("Defined classes");
   Printf("class                              version  bits  initialized");
   Printf("=============================================================");
   for (int i = 0; i < fgTally; i++) {
      ClassRec_t *r = fgSortedTable[i];
      n++;
      if (gROOT->GetClass(r->name, kFALSE)) {
         ninit++;
         Printf("%-32s %6d %7d       Yes", r->name, r->id, r->bits);
      } else
         Printf("%-32s %6d %7d       No",  r->name, r->id, r->bits);
   }
   Printf("-------------------------------------------------------------");
   Printf("Total classes: %4d   initialized: %4d", n, ninit);
   Printf("=============================================================");

   Printf("");
}

//______________________________________________________________________________
void TClassTable::SortTable()
{
   // Sort the class table by ascending class ID's.

   if (!fgSorted) {
      if (fgSortedTable) delete [] fgSortedTable;
      fgSortedTable = new ClassRec_t* [fgTally];

      int j = 0;
      for (int i = 0; i < fgSize; i++)
         for (ClassRec_t *r = fgTable[i]; r; r = r->next)
            fgSortedTable[j++] = r;

      ::qsort(fgSortedTable, fgTally, sizeof(ClassRec_t *), ::ClassComp);
      fgSorted = kTRUE;
   }
}

//______________________________________________________________________________
void TClassTable::Terminate()
{
   // Deletes the class table (this static class function calls the dtor).

   if (gClassTable) {
      for (int i = 0; i < fgSize; i++)
         for (ClassRec_t *r = fgTable[i]; r; ) {
            ClassRec_t *t = r;
            r = r->next;
            fgIdMap->Remove(r->info->name());
            delete [] t->name;
            delete t;
         }
      delete [] fgTable; fgTable = 0;
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
void ROOT::ResetClassVersion(TClass* cl, const char* cname, Short_t newid)
{
   // Update the version number.  This is called via the RootClassVersion macro

   if (cname) {
      ClassRec_t *r = TClassTable::FindElement(cname,kFALSE);
      if (r) r->id = newid;
   }
   if (cl) {
      if (cl->fVersionUsed) {
         // Problem, the reset is called after the first usage!
         Error("ResetClassVersion","Version number of %s can not be changed after first usage!",
               cl->GetName());
      } else {
         cl->SetClassVersion(newid);
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
        TClass *cl = gROOT->GetClass(cname, kFALSE);
        if (cl) cl->SetUnloaded();
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

