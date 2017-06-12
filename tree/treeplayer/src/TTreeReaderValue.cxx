// @(#)root/treeplayer:$Id$
// Author: Axel Naumann, 2011-09-28

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTreeReaderValue.h"

#include "TTreeReader.h"
#include "TBranchClones.h"
#include "TBranchElement.h"
#include "TBranchRef.h"
#include "TBranchSTL.h"
#include "TBranchProxyDirector.h"
#include "TClassEdit.h"
#include "TLeaf.h"
#include "TTreeProxyGenerator.h"
#include "TTreeReaderValue.h"
#include "TRegexp.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TNtuple.h"
#include <vector>

/** \class TTreeReaderValue

Extracts data from a TTree.
*/

ClassImp(ROOT::Internal::TTreeReaderValueBase);

////////////////////////////////////////////////////////////////////////////////
/// Construct a tree value reader and register it with the reader object.

ROOT::Internal::TTreeReaderValueBase::TTreeReaderValueBase(TTreeReader* reader /*= 0*/,
                                                 const char* branchname /*= 0*/,
                                                 TDictionary* dict /*= 0*/):
   fHaveLeaf(0),
   fHaveStaticClassOffsets(0),
   fReadStatus(kReadNothingYet),
   fBranchName(branchname),
   fTreeReader(reader),
   fDict(dict)
{
   RegisterWithTreeReader();
}

////////////////////////////////////////////////////////////////////////////////
/// Copy-construct.

ROOT::Internal::TTreeReaderValueBase::TTreeReaderValueBase(const TTreeReaderValueBase& rhs):
   fHaveLeaf(rhs.fHaveLeaf),
   fHaveStaticClassOffsets(rhs.fHaveStaticClassOffsets),
   fReadStatus(rhs.fReadStatus),
   fSetupStatus(rhs.fSetupStatus),
   fBranchName(rhs.fBranchName),
   fLeafName(rhs.fLeafName),
   fTreeReader(rhs.fTreeReader),
   fDict(rhs.fDict),
   fProxy(rhs.fProxy),
   fLeaf(rhs.fLeaf),
   fStaticClassOffsets(rhs.fStaticClassOffsets)
{
   RegisterWithTreeReader();
}

////////////////////////////////////////////////////////////////////////////////
/// Copy-assign.

ROOT::Internal::TTreeReaderValueBase&
ROOT::Internal::TTreeReaderValueBase::operator=(const TTreeReaderValueBase& rhs) {
   if (&rhs != this) {
      fHaveLeaf = rhs.fHaveLeaf;
      fHaveStaticClassOffsets = rhs.fHaveStaticClassOffsets;
      fBranchName = rhs.fBranchName;
      fLeafName = rhs.fLeafName;
      if (fTreeReader != rhs.fTreeReader) {
         if (fTreeReader)
            fTreeReader->DeregisterValueReader(this);
         fTreeReader = rhs.fTreeReader;
         RegisterWithTreeReader();
      }
      fDict = rhs.fDict;
      fProxy = rhs.fProxy;
      fLeaf = rhs.fLeaf;
      fSetupStatus = rhs.fSetupStatus;
      fReadStatus = rhs.fReadStatus;
      fStaticClassOffsets = rhs.fStaticClassOffsets;
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Unregister from tree reader, cleanup.

ROOT::Internal::TTreeReaderValueBase::~TTreeReaderValueBase()
{
   if (fTreeReader) fTreeReader->DeregisterValueReader(this);
   R__ASSERT((fLeafName.Length() == 0 ) == !fHaveLeaf
          && "leafness disagreement");
   R__ASSERT(fStaticClassOffsets.empty() == !fHaveStaticClassOffsets
          && "static class offset disagreement");
}

////////////////////////////////////////////////////////////////////////////////
/// Register with tree reader.

void ROOT::Internal::TTreeReaderValueBase::RegisterWithTreeReader() {
   if (fTreeReader) {
      if (!fTreeReader->RegisterValueReader(this)) {
         fTreeReader = nullptr;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Try to read the value from the TBranchProxy, returns
/// the status of the read.

ROOT::Internal::TTreeReaderValueBase::EReadStatus
ROOT::Internal::TTreeReaderValueBase::ProxyRead() {
   if (!fProxy) return kReadNothingYet;
   if (fProxy->Read()) {
      fReadStatus = kReadSuccess;
   } else {
      fReadStatus = kReadError;
   }
   return fReadStatus;
}

////////////////////////////////////////////////////////////////////////////////
/// Stringify the template argument.
std::string ROOT::Internal::TTreeReaderValueBase::GetElementTypeName(const std::type_info& ti) {
   int err;
   char* buf = TClassEdit::DemangleTypeIdName(ti, err);
   std::string ret = buf;
   free(buf);
   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// The TTreeReader has switched to a new TTree. Update the leaf.

void ROOT::Internal::TTreeReaderValueBase::NotifyNewTree(TTree* newTree) {
   if (!fHaveLeaf)
      return;

   TBranch *myBranch = newTree->GetBranch(fBranchName);

   if (!myBranch) {
      fReadStatus = kReadError;
      Error("TTreeReaderValueBase::GetLeaf()", "Unable to get the branch from the tree");
      return;
   }

   fLeaf = myBranch->GetLeaf(fLeafName);
   if (!fLeaf) {
      Error("TTreeReaderValueBase::GetLeaf()", "Failed to get the leaf from the branch");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the memory address of the object being read.

void* ROOT::Internal::TTreeReaderValueBase::GetAddress() {
   if (ProxyRead() != kReadSuccess) return 0;

   if (fHaveLeaf){
      if (GetLeaf()){
         return fLeaf->GetValuePointer();
      }
      else {
         fReadStatus = kReadError;
         Error("TTreeReaderValueBase::GetAddress()", "Unable to get the leaf");
         return 0;
      }
   }
   if (fHaveStaticClassOffsets){ // Follow all the pointers
      Byte_t *address = (Byte_t*)fProxy->GetWhere();

      for (unsigned int i = 0; i < fStaticClassOffsets.size() - 1; ++i){
         address = *(Byte_t**)(address + fStaticClassOffsets[i]);
      }

      return address + fStaticClassOffsets.back();
   }
   return (Byte_t*)fProxy->GetWhere();
}

////////////////////////////////////////////////////////////////////////////////
/// Create the proxy object for our branch.

void ROOT::Internal::TTreeReaderValueBase::CreateProxy() {
   if (fProxy) {
      return;
   }

   fSetupStatus = kSetupInternalError; // Fallback; set to something concrete below.
   if (!fTreeReader) {
      Error("TTreeReaderValueBase::CreateProxy()", "TTreeReader object not set / available for branch %s!",
            fBranchName.Data());
      fSetupStatus = kSetupTreeDestructed;
      return;
   }
   if (!fDict) {
      TBranch* br = fTreeReader->GetTree()->GetBranch(fBranchName);
      const char* brDataType = "{UNDETERMINED}";
      if (br) {
         TDictionary* brDictUnused = 0;
         brDataType = GetBranchDataType(br, brDictUnused);
      }
      Error("TTreeReaderValueBase::CreateProxy()", "The template argument type T of %s accessing branch %s (which contains data of type %s) is not known to ROOT. You will need to create a dictionary for it.",
            GetDerivedTypeName(), fBranchName.Data(), brDataType);
      fSetupStatus = kSetupMissingDictionary;
      return;
   }

   // Search for the branchname, determine what it contains, and wire the
   // TBranchProxy representing it to us so we can access its data.

   TNamedBranchProxy* namedProxy
      = (TNamedBranchProxy*)fTreeReader->FindObject(fBranchName);
   if (namedProxy && namedProxy->GetDict() == fDict) {
      fProxy = namedProxy->GetProxy();
      fSetupStatus = kSetupMatch;
      return;
   }

   TBranch* branch = fTreeReader->GetTree()->GetBranch(fBranchName);
   TLeaf *myLeaf = NULL;
   TDictionary* branchActualType = 0;

   if (!branch) {
      if (fBranchName.Contains(".")){
         TRegexp leafNameExpression ("\\.[a-zA-Z0-9_]+$");
         TString leafName (fBranchName(leafNameExpression));
         TString branchName = fBranchName(0, fBranchName.Length() - leafName.Length());
         branch = fTreeReader->GetTree()->GetBranch(branchName);
         if (!branch){
            std::vector<TString> nameStack;
            nameStack.push_back(TString()); //Trust me
            nameStack.push_back(leafName.Strip(TString::kBoth, '.'));
            leafName = branchName(leafNameExpression);
            branchName = branchName(0, branchName.Length() - leafName.Length());

            branch = fTreeReader->GetTree()->GetBranch(branchName);
            if (!branch) branch = fTreeReader->GetTree()->GetBranch(branchName + ".");
            if (leafName.Length()) nameStack.push_back(leafName.Strip(TString::kBoth, '.'));

            while (!branch && branchName.Contains(".")){
               leafName = branchName(leafNameExpression);
               branchName = branchName(0, fBranchName.Length() - leafName.Length());
               branch = fTreeReader->GetTree()->GetBranch(branchName);
               if (!branch) branch = fTreeReader->GetTree()->GetBranch(branchName + ".");
               nameStack.push_back(leafName.Strip(TString::kBoth, '.'));
            }

            if (branch && branch->IsA() == TBranchElement::Class()){
               TBranchElement *myBranchElement = (TBranchElement*)branch;

               TString traversingBranch = nameStack.back();
               nameStack.pop_back();

               bool found = true;

               TDataType *finalDataType = 0;

               std::vector<Long64_t> offsets;
               Long64_t offset = 0;
               TClass *elementClass = 0;

               TObjArray *myObjArray = myBranchElement->GetInfo()->GetElements();
               TVirtualStreamerInfo *myInfo = myBranchElement->GetInfo();

               while (nameStack.size() && found){
                  found = false;

                  for (int i = 0; i < myObjArray->GetEntries(); ++i){

                     TStreamerElement *tempStreamerElement = (TStreamerElement*)myObjArray->At(i);

                     if (!strcmp(tempStreamerElement->GetName(), traversingBranch.Data())){
                        offset += myInfo->GetElementOffset(i);

                        traversingBranch = nameStack.back();
                        nameStack.pop_back();

                        elementClass = tempStreamerElement->GetClass();
                        if (elementClass) {
                           myInfo = elementClass->GetStreamerInfo(0);
                           myObjArray = myInfo->GetElements();
                           // FIXME: this is odd, why is 'i' not also reset????
                        }
                        else {
                           finalDataType = TDataType::GetDataType((EDataType)tempStreamerElement->GetType());
                           if (!finalDataType) {
                              TDictionary* seType = TDictionary::GetDictionary(tempStreamerElement->GetTypeName());
                              if (seType && seType->IsA() == TDataType::Class()) {
                                 finalDataType = TDataType::GetDataType((EDataType)((TDataType*)seType)->GetType());
                              }
                           }
                        }

                        if (tempStreamerElement->IsaPointer()){
                           offsets.push_back(offset);
                           offset = 0;
                        }

                        found = true;
                        break;
                     }
                  }
               }

               offsets.push_back(offset);

               if (found){
                  fStaticClassOffsets = offsets;
                  fHaveStaticClassOffsets = 1;

                  if (fDict != finalDataType && fDict != elementClass){
                     Error("TTreeReaderValueBase::CreateProxy", "Wrong data type %s", finalDataType ? finalDataType->GetName() : elementClass ? elementClass->GetName() : "UNKNOWN");
                     fSetupStatus = kSetupMismatch;
                     fProxy = 0;
                     return;
                  }
               }
            }


            if (!fHaveStaticClassOffsets) {
               Error("TTreeReaderValueBase::CreateProxy()", "The tree does not have a branch called %s. You could check with TTree::Print() for available branches.", fBranchName.Data());
               fSetupStatus = kSetupMissingBranch;
               fProxy = 0;
               return;
            }
         }
         else {
            myLeaf = branch->GetLeaf(TString(leafName(1, leafName.Length())));
            if (!myLeaf){
               Error("TTreeReaderValueBase::CreateProxy()",
                     "The tree does not have a branch, nor a sub-branch called %s. You could check with TTree::Print() for available branches.", fBranchName.Data());
               fSetupStatus = kSetupMissingBranch;
               fProxy = 0;
               return;
            }
            else {
               TDictionary *tempDict = TDictionary::GetDictionary(myLeaf->GetTypeName());
               if (tempDict && tempDict->IsA() == TDataType::Class() && TDictionary::GetDictionary(((TDataType*)tempDict)->GetTypeName()) == fDict){
                  //fLeafOffset = myLeaf->GetOffset() / 4;
                  branchActualType = fDict;
                  fLeaf = myLeaf;
                  fBranchName = branchName;
                  fLeafName = leafName(1, leafName.Length());
                  fHaveLeaf = fLeafName.Length() > 0;
                  fSetupStatus = kSetupMatchLeaf;
               }
               else {
                  Error("TTreeReaderValueBase::CreateProxy()",
                        "Leaf of type %s cannot be read by TTreeReaderValue<%s>.", myLeaf->GetTypeName(), fDict->GetName());
                  fSetupStatus = kSetupMismatch;
               }
            }
         }
      }
      else {
         Error("TTreeReaderValueBase::CreateProxy()", "The tree does not have a branch called %s. You could check with TTree::Print() for available branches.", fBranchName.Data());
         fProxy = 0;
         return;
      }
   }

   if (!myLeaf && !fHaveStaticClassOffsets) {
      const char* branchActualTypeName = GetBranchDataType(branch, branchActualType);

      if (!branchActualType) {
         Error("TTreeReaderValueBase::CreateProxy()", "The branch %s contains data of type %s, which does not have a dictionary.",
               fBranchName.Data(), branchActualTypeName ? branchActualTypeName : "{UNDETERMINED TYPE}");
         fProxy = 0;
         return;
      }

      if (fDict != branchActualType) {
         TDataType *dictdt = dynamic_cast<TDataType*>(fDict);
         TDataType *actualdt = dynamic_cast<TDataType*>(branchActualType);
         bool complainAboutMismatch = true;
         if (dictdt && actualdt) {
            if (dictdt->GetType() > 0 && dictdt->GetType() == actualdt->GetType()) {
               // Same numerical type but different TDataType, likely Long64_t
               complainAboutMismatch = false;
            } else if ((actualdt->GetType() == kDouble32_t && dictdt->GetType() == kDouble_t)
                       || (actualdt->GetType() == kFloat16_t && dictdt->GetType() == kFloat_t)) {
               // Double32_t and Float16_t never "decay" to their underlying type;
               // we need to identify them manually here (ROOT-8731).
               complainAboutMismatch = false;
            }
         }
         if (complainAboutMismatch) {
            Error("TTreeReaderValueBase::CreateProxy()",
                  "The branch %s contains data of type %s. It cannot be accessed by a TTreeReaderValue<%s>",
                  fBranchName.Data(), branchActualType->GetName(),
                  fDict->GetName());
            return;
         }
      }
   }


   // Update named proxy's dictionary
   if (namedProxy && !namedProxy->GetDict()) {
      namedProxy->SetDict(fDict);
      fProxy = namedProxy->GetProxy();
      if (fProxy)
         fSetupStatus = kSetupMatch;
      return;
   }

   // Search for the branchname, determine what it contains, and wire the
   // TBranchProxy representing it to us so we can access its data.
   // A proxy for branch must not have been created before (i.e. check
   // fProxies before calling this function!)

   TString membername;

   bool isTopLevel = branch->GetMother() == branch;
   if (!isTopLevel) {
      membername = strrchr(branch->GetName(), '.');
      if (membername.IsNull()) {
         membername = branch->GetName();
      }
   }
   namedProxy = new TNamedBranchProxy(fTreeReader->fDirector, branch, membername);
   fTreeReader->GetProxies()->Add(namedProxy);
   fProxy = namedProxy->GetProxy();
   if (fProxy) {
      fSetupStatus = kSetupMatch;
   } else {
      fSetupStatus = kSetupMismatch;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the type of data stored by branch; put its dictionary into
/// dict, return its type name. If no dictionary is available, at least
/// its type name should be returned.

const char* ROOT::Internal::TTreeReaderValueBase::GetBranchDataType(TBranch* branch,
                                           TDictionary* &dict) const
{
   dict = 0;
   if (branch->IsA() == TBranchElement::Class()) {
      TBranchElement* brElement = (TBranchElement*)branch;

      auto ResolveTypedef = [&]() -> void {
         if (dict->IsA() != TDataType::Class())
            return;
         // Resolve the typedef.
         dict = TDictionary::GetDictionary(((TDataType*)dict)->GetTypeName());
         if (dict->IsA() != TDataType::Class()) {
            // Might be a class.
            if (dict != fDict) {
               dict = TClass::GetClass(brElement->GetTypeName());
            }
            if (dict != fDict) {
               dict = brElement->GetCurrentClass();
            }
         }
      };

      if (brElement->GetType() == TBranchElement::kSTLNode ||
            brElement->GetType() == TBranchElement::kLeafNode ||
            brElement->GetType() == TBranchElement::kObjectNode) {

         TStreamerInfo *streamerInfo = brElement->GetInfo();
         Int_t id = brElement->GetID();

         if (id >= 0){
            TStreamerElement *element = (TStreamerElement*)streamerInfo->GetElements()->At(id);
            if (element->IsA() == TStreamerSTL::Class()){
               TStreamerSTL *myStl = (TStreamerSTL*)element;
               dict = myStl->GetClass();
               return 0;
            }
         }

         if (brElement->GetType() == 3 || brElement->GetType() == 4) {
            dict = brElement->GetCurrentClass();
            return brElement->GetTypeName();
         }

         if (brElement->GetTypeName())
            dict = TDictionary::GetDictionary(brElement->GetTypeName());

         if (dict)
            ResolveTypedef();
         else
            dict = brElement->GetCurrentClass();

         return brElement->GetTypeName();
      } else if (brElement->GetType() == TBranchElement::kClonesNode) {
         dict = TClonesArray::Class();
         return "TClonesArray";
      } else if (brElement->GetType() == 31
                 || brElement->GetType() == 41) {
         // it's a member, extract from GetClass()'s streamer info
         Error("TTreeReaderValueBase::GetBranchDataType()", "Must use TTreeReaderArray to access a member of an object that is stored in a collection.");
      } else if (brElement->GetType() == -1 && brElement->GetTypeName()) {
         dict = TDictionary::GetDictionary(brElement->GetTypeName());
         ResolveTypedef();
         return brElement->GetTypeName();
      } else {
         Error("TTreeReaderValueBase::GetBranchDataType()", "Unknown type and class combination: %i, %s", brElement->GetType(), brElement->GetClassName());
      }
      return 0;
   } else if (branch->IsA() == TBranch::Class()
              || branch->IsA() == TBranchObject::Class()
              || branch->IsA() == TBranchSTL::Class()) {
      if (branch->GetTree()->IsA() == TNtuple::Class()){
         dict = TDataType::GetDataType(kFloat_t);
         return dict->GetName();
      }
      const char* dataTypeName = branch->GetClassName();
      if ((!dataTypeName || !dataTypeName[0])
          && branch->IsA() == TBranch::Class()) {
         TLeaf *myLeaf = branch->GetLeaf(branch->GetName());
         if (myLeaf){
            TDictionary *myDataType = TDictionary::GetDictionary(myLeaf->GetTypeName());
            if (myDataType && myDataType->IsA() == TDataType::Class()){
               dict = TDataType::GetDataType((EDataType)((TDataType*)myDataType)->GetType());
               return myLeaf->GetTypeName();
            }
         }


         // leaflist. Can't represent.
         Error("TTreeReaderValueBase::GetBranchDataType()", "The branch %s was created using a leaf list and cannot be represented as a C++ type. Please access one of its siblings using a TTreeReaderArray:", branch->GetName());
         TIter iLeaves(branch->GetListOfLeaves());
         TLeaf* leaf = 0;
         while ((leaf = (TLeaf*) iLeaves())) {
            Error("TTreeReaderValueBase::GetBranchDataType()", "   %s.%s", branch->GetName(), leaf->GetName());
         }
         return 0;
      }
      if (dataTypeName) dict = TDictionary::GetDictionary(dataTypeName);
      return dataTypeName;
   } else if (branch->IsA() == TBranchClones::Class()) {
      dict = TClonesArray::Class();
      return "TClonesArray";
   } else if (branch->IsA() == TBranchRef::Class()) {
      // Can't represent.
      Error("TTreeReaderValueBase::GetBranchDataType()", "The branch %s is a TBranchRef and cannot be represented as a C++ type.", branch->GetName());
      return 0;
   } else {
      Error("TTreeReaderValueBase::GetBranchDataType()", "The branch %s is of type %s - something that is not handled yet.", branch->GetName(), branch->IsA()->GetName());
      return 0;
   }

   return 0;
}

