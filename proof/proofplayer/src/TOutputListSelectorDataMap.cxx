// @(#)root/proofplayer:$Id$
// Author: Axel Naumann, 2010-06-09

/*************************************************************************
 * Copyright (C) 1995-2010, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TOutputListSelectorDataMap                                           //
//                                                                      //
// Set the selector's data members to the corresponding elements of the //
// output list.                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TOutputListSelectorDataMap.h"

#include "TClass.h"
#include "TDataMember.h"
#include "TExMap.h"
#include "THashTable.h"
#include "TList.h"
#include "TMemberInspector.h"
#include "TProofDebug.h"
#include "TSelector.h"
#include "TSelectorCint.h"

#include <cstddef>

namespace {

   static TClass* IsSettableDataMember(TDataMember* dm) {
      if (!dm || !dm->IsaPointer() || dm->IsBasic()) return 0;
      TString dtTypeName = dm->GetFullTypeName();
      if (!dtTypeName.EndsWith("*")) return 0;
      dtTypeName.Remove(dtTypeName.Length()-1);
      return TClass::GetClass(dtTypeName);
   }

   class TSetSelDataMembers: public TMemberInspector {
   public:
      TSetSelDataMembers(const TOutputListSelectorDataMap& owner, TCollection* dmInfo, TList* output);
      void Inspect(TClass *cl, const char *parent, const char *name, const void *addr);
      Ssiz_t GetNumSet() const { return fNumSet; }
   private:
      TCollection* fDMInfo; // output list object name / member name pairs for output list entries
      TList* fOutputList; // merged output list
      Ssiz_t fNumSet; // number of initialized data members
      const TOutputListSelectorDataMap& fOwner; // owner, used for messaging
   };
}

//______________________________________________________________________________
TSetSelDataMembers::TSetSelDataMembers(const TOutputListSelectorDataMap& owner,
                                       TCollection* dmInfo, TList* output):
   fDMInfo(dmInfo), fOutputList(output), fNumSet(0), fOwner(owner)
{}

//______________________________________________________________________________
void TSetSelDataMembers::Inspect(TClass *cl, const char* parent, const char *name, const void *addr)
{
   // This method is called by the ShowMembers() method for each
   // data member to recursively collect all base classes' members.
   //
   //    cl     is the pointer to the current class
   //    parent is the parent name (in case of composed objects)
   //    name   is the data member name
   //    addr   is the data member address

   while (name[0] == '*') ++name;

   TObject* mapping = fDMInfo->FindObject(name);
   if (!mapping) return;

   PDB(kOutput,1) fOwner.Info("SetDataMembers()",
                              "data member `%s%s::%s' maps to output list object `%s'",
                              cl->GetName(), parent, name, mapping->GetTitle());

   TObject* outputObj = fOutputList->FindObject(mapping->GetTitle());
   if (!outputObj) {
      PDB(kOutput,1) fOwner.Warning("SetDataMembers()",
                                    "object `%s' not found in output list!",
                                    mapping->GetTitle());
      return;
   }

   // Check data member type
   TDataMember *dm = cl->GetDataMember(name);
   TClass* cldt = IsSettableDataMember(dm);
   if (!cldt) {
      PDB(kOutput,1) fOwner.Warning("SetDataMembers()",
                                    "unusable data member `%s' should have been detected by TCollectDataMembers!",
                                    name);
      return;
   }

   char *pointer = (char*)addr;
   char **ppointer = (char**)(pointer);
   if (*ppointer) {
      // member points to something - replace instead of delete to not crash on deleting uninitialized values.
      fOwner.Warning("SetDataMembers()", "potential memory leak: replacing data member `%s' != 0. "
                     "Please initialize %s to 0 in constructor %s::%s()",
                     name, name, cl->GetName(), cl->GetName());
   }
   *ppointer = (char*)outputObj;
   ++fNumSet;
}


namespace {
   class TCollectDataMembers: public TMemberInspector {
   public:
      TCollectDataMembers(const TOutputListSelectorDataMap& owner): fOwner(owner) { }
      ~TCollectDataMembers();
      void Inspect(TClass *cl, const char *parent, const char *name, const void *addr);
      TExMap& GetMemberPointers() { return fMap; }
   private:
      TExMap fMap; //map of data member's value to TDataMember
      const TOutputListSelectorDataMap& fOwner; //owner, used for messaging
   };
}

//______________________________________________________________________________
void TCollectDataMembers::Inspect(TClass *cl, const char* /*parent*/, const char *name, const void *addr)
{
   // This method is called by the ShowMembers() method for each
   // data member to recursively collect all base classes' members.
   //
   //    cl     is the pointer to the current class
   //    parent is the parent name (in case of composed objects)
   //    name   is the data member name
   //    addr   is the data member address
   TDataMember *dm = cl->GetDataMember(name);
   if (!IsSettableDataMember(dm)) return;

   char *pointer = (char*)addr;
   char **ppointer = (char**)(pointer);
   char **p3pointer = (char**)(*ppointer);
   if (p3pointer) {
      // The data member points to something.
      // Handle multiple pointers to the same output list object:
      TObject* prev = (TObject*) (ptrdiff_t)fMap.GetValue((Long64_t)(ptrdiff_t)p3pointer);
      if (prev) {
         // We have a previous entry - is it a data member or already a TList (of data members)?
         if (prev->InheritsFrom(TDataMember::Class())) {
            fMap.Remove((Long64_t)(ptrdiff_t)p3pointer);
            TList* dmList = new TList;
            dmList->Add(prev);
            dmList->Add(dm);
            fMap.Add((Long64_t)(ptrdiff_t)p3pointer, (Long64_t)(ptrdiff_t)dmList);
         } else {
            TList* prevList = (TList*) prev;
            prevList->Add(dm);
         }
      } else {
         fMap.Add((Long64_t)(ptrdiff_t)p3pointer, (Long64_t)(ptrdiff_t)dm);
      }
      if (name[0] == '*') ++name;
      PDB(kOutput,1) fOwner.Info("Init()", "considering data member `%s'", name);
   }
}

TCollectDataMembers::~TCollectDataMembers() {
   // Destructor

   // Clean up collection of TDataMembers in fMap
   TExMapIter iMembers(&fMap);
   Long64_t key;
   Long64_t value;
   while (iMembers.Next(key, value)) {
      TObject* obj = (TObject*) (ptrdiff_t) value;
      if (obj->InheritsFrom(TList::Class())) {
         delete obj;
      }
   }
}

ClassImp(TOutputListSelectorDataMap);

//______________________________________________________________________________
TOutputListSelectorDataMap::TOutputListSelectorDataMap(TSelector* sel /*= 0*/):
   fMap(0)
{
   // Create a mapper between output list items and TSelector data members.
   if (sel) Init(sel);
}

//______________________________________________________________________________
const char* TOutputListSelectorDataMap::GetName() const
{
   // Return static name for TOutputListSelectorDataMap objects.
   return "PROOF_TOutputListSelectorDataMap_object";
}

//______________________________________________________________________________
Bool_t TOutputListSelectorDataMap::Init(TSelector* sel)
{
   // Initialize the data member <-> output list mapping from a selector.
   if (!sel) {
      PDB(kOutput,1) Warning("Init","Leave (no selector!)");
      return kFALSE;
   }
   TCollection* outList = sel->GetOutputList();
   if (!outList) {
      PDB(kOutput,1) Info("Init()","Leave (no output)");
      return kFALSE;
   }

   if (outList->FindObject(GetName())) {
      // mapping already exists?!
      PDB(kOutput,1) Warning("Init","Mapping already exists!");
      return kFALSE;
   }

   if (fMap) delete fMap;
   fMap = new THashTable;
   fMap->SetOwner();

   TCollectDataMembers cdm(*this);
   TClass* cl = sel->IsA();
   if (cl && cl->InheritsFrom(TSelectorCint::Class())) {
      // we don't want to set TSelectorCint's data members, but
      // the data members that it represents!
      TSelectorCint* selCINT = dynamic_cast<TSelectorCint*>(sel);
      if (selCINT) {
         cl = selCINT->GetInterpretedClass();
         sel = selCINT->GetInterpretedSelector();
      } else {
         cl = 0;
         Error("Init", "failed to get TSelectorCint interpreted class!");
      }
   }
   if (!cl || (cl && !cl->CallShowMembers(sel, cdm))) {
      // failed to map
      PDB(kOutput,1) Warning("Init","Failed to determine mapping!");
      return kFALSE;
   }
   PDB(kOutput,1) Info("Init()","Found %d data members.",
                        cdm.GetMemberPointers().GetSize());

   // Iterate over output list entries and find data members pointing to the
   // same value. Store that mapping (or a miss).
   TIter iOutput(outList);
   TObject* output;
   TList oneDM;
   while ((output = iOutput())) {
      TObject* obj = (TObject*) (ptrdiff_t)cdm.GetMemberPointers().GetValue((Long64_t)(ptrdiff_t)output);
      if (!obj) continue;

      TList* addAllDM = 0;
      if (obj->InheritsFrom(TDataMember::Class())) {
         oneDM.Add(obj);
         addAllDM = &oneDM;
      } else {
         addAllDM = (TList*) obj;
      }
      TIter iDM(addAllDM);
      TDataMember* dm = 0;
      while ((dm = (TDataMember*) iDM())) {
         fMap->Add(new TNamed(dm->GetName(), output->GetName()));
         PDB(kOutput,1) Info("Init()","Data member `%s' corresponds to output `%s'",
                              dm->GetName(), output->GetName());
      }
      oneDM.Clear();
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TOutputListSelectorDataMap::SetDataMembers(TSelector* sel) const
{
   // Given an output list, set the data members of a TSelector.
   TList* output = sel->GetOutputList();
   if (!output || output->IsEmpty()) return kTRUE;

   // Set fSelector's data members
   TSetSelDataMembers ssdm(*this, fMap, output);
   TClass* cl = sel->IsA();
   if (cl->InheritsFrom(TSelectorCint::Class())) {
      // we don't want to set TSelectorCint's data members, but
      // the data members that it represents!
      TSelectorCint* selCINT = dynamic_cast<TSelectorCint*>(sel);
      if (selCINT) {
         cl = selCINT->GetInterpretedClass();
         sel = selCINT->GetInterpretedSelector();
      } else {
         cl = 0;
         Error("Init", "failed to get TSelectorCint interpreted class!");
         return kFALSE;
      }
   }
   Bool_t res = cl->CallShowMembers(sel, ssdm);
   PDB(kOutput,1) Info("SetDataMembers()","%s, set %d data members.",
                       (res ? "success" : "failure"), ssdm.GetNumSet());
   return res;
}

//______________________________________________________________________________
Bool_t TOutputListSelectorDataMap::Merge(TObject* obj)
{
   // Merge another TOutputListSelectorDataMap object, check
   // consistency.
   TOutputListSelectorDataMap* other = dynamic_cast<TOutputListSelectorDataMap*>(obj);
   if (!other) return kFALSE;

   // check for consistency
   TIter iMapping(other->GetMap());
   TNamed* mapping = 0;
   while ((mapping = (TNamed*)iMapping())) {
      TObject* oldMap = fMap->FindObject(mapping->GetName());
      if (!oldMap) {
         fMap->Add(new TNamed(*mapping));
      } else {
         if (strcmp(oldMap->GetTitle(), mapping->GetTitle())) {
            // ouch, contradicting maps!
            PDB(kOutput,1)
               Warning("Merge()",
                       "contradicting mapping for data member `%s' (output list entry `%s' vs. `%s'). "
                       "Cancelling automatic TSelector data member setting!",
                       mapping->GetName(), oldMap->GetTitle(), mapping->GetTitle());
            fMap->Clear();
            return kFALSE;
         }
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
TOutputListSelectorDataMap* TOutputListSelectorDataMap::FindInList(TCollection* coll)
{
   // Find a TOutputListSelectorDataMap in a collection
   TIter iOutput(coll);
   TObject* out = 0;
   TOutputListSelectorDataMap* olsdm = 0;
   while ((out = iOutput())) {
      if (out->InheritsFrom(TOutputListSelectorDataMap::Class())) {
         olsdm = dynamic_cast<TOutputListSelectorDataMap*>(out);
         if (olsdm) break;
      }
   }
   return olsdm;
}
