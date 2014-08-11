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
#include "TLeaf.h"
#include "TTreeProxyGenerator.h"
#include "TTreeReaderValue.h"
#include "TRegexp.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TNtuple.h"
#include <vector>

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// TTreeReaderValue                                                        //
//                                                                            //
// Extracts data from a TTree.                                                //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

ClassImp(TTreeReaderValueBase)

//______________________________________________________________________________
ROOT::TTreeReaderValueBase::TTreeReaderValueBase(TTreeReader* reader /*= 0*/,
                                                 const char* branchname /*= 0*/,
                                                 TDictionary* dict /*= 0*/):
   fBranchName(branchname),
   fTreeReader(reader),
   fDict(dict),
   fProxy(NULL),
   fLeaf(NULL),
   fTreeLastOffset(-1),
   fSetupStatus(kSetupNotSetup),
   fReadStatus(kReadNothingYet)
{
   // Construct a tree value reader and register it with the reader object.
   if (fTreeReader) fTreeReader->RegisterValueReader(this);
}

//______________________________________________________________________________
ROOT::TTreeReaderValueBase::~TTreeReaderValueBase()
{
   // Unregister from tree reader, cleanup.
   if (fTreeReader) fTreeReader->DeregisterValueReader(this);
}

//______________________________________________________________________________
ROOT::TTreeReaderValueBase::EReadStatus
ROOT::TTreeReaderValueBase::ProxyRead() {
   // Try to read the value from the TBranchProxy, returns
   // the status of the read.

   if (!fProxy) return kReadNothingYet;
   if (fProxy->Read()) {
      fReadStatus = kReadSuccess;
   } else {
      fReadStatus = kReadError;
   }
   return fReadStatus;
}

//______________________________________________________________________________
TLeaf* ROOT::TTreeReaderValueBase::GetLeaf() {
   // If we are reading a leaf, return the corresponding TLeaf.

   if (fLeafName.Length() > 0){

      Long64_t newChainOffset = fTreeReader->GetTree()->GetChainOffset();

      if (newChainOffset != fTreeLastOffset){
         fTreeLastOffset = newChainOffset;

         TTree *myTree = fTreeReader->GetTree();

         if (!myTree) {
            fReadStatus = kReadError;
            Error("GetLeaf()", "Unable to get the tree from the TTreeReader");
            return 0;
         }

         TBranch *myBranch = myTree->GetBranch(fBranchName);

         if (!myBranch) {
            fReadStatus = kReadError;
            Error("GetLeaf()", "Unable to get the branch from the tree");
            return 0;
         }

         fLeaf = myBranch->GetLeaf(fLeafName);
         if (!fLeaf) {
            Error("GetLeaf()", "Failed to get the leaf from the branch");
         }
      }
      return fLeaf;
   }
   else {
      Error("GetLeaf()", "We are not reading a leaf");
      return 0;
   }
}

//______________________________________________________________________________
void* ROOT::TTreeReaderValueBase::GetAddress() {
   // Returns the memory address of the object being read.

   if (ProxyRead() != kReadSuccess) return 0;

   if (fLeafName.Length() > 0){
      if (GetLeaf()){
         return fLeaf->GetValuePointer();
      }
      else {
         fReadStatus = kReadError;
         Error("GetAddress()", "Unable to get the leaf");
         return 0;
      }
   }
   if (!fStaticClassOffsets.empty()){ // Follow all the pointers
      Byte_t *address = (Byte_t*)fProxy->GetWhere();

      for (unsigned int i = 0; i < fStaticClassOffsets.size() - 1; ++i){
         address = *(Byte_t**)(address + fStaticClassOffsets[i]);
      }

      return address + fStaticClassOffsets.back();
   }
   return fProxy ? (Byte_t*)fProxy->GetWhere() : 0;
}

//______________________________________________________________________________
void ROOT::TTreeReaderValueBase::CreateProxy() {
   // Create the proxy object for our branch.
   if (fProxy) {
      return;
   }
   if (!fTreeReader) {
      Error("CreateProxy()", "TTreeReader object not set / available for branch %s!",
            fBranchName.Data());
      return;
   }
   if (!fDict) {
      TBranch* br = fTreeReader->GetTree()->GetBranch(fBranchName);
      const char* brDataType = "{UNDETERMINED}";
      if (br) {
         TDictionary* brDictUnused = 0;
         brDataType = GetBranchDataType(br, brDictUnused);
      }
      Error("CreateProxy()", "The template argument type T of %s accessing branch %s (which contains data of type %s) is not known to ROOT. You will need to create a dictionary for it.",
            GetDerivedTypeName(), fBranchName.Data(), brDataType);
      return;
   }

   // Search for the branchname, determine what it contains, and wire the
   // TBranchProxy representing it to us so we can access its data.

   ROOT::TNamedBranchProxy* namedProxy
      = (ROOT::TNamedBranchProxy*)fTreeReader->FindObject(fBranchName);
   if (namedProxy && namedProxy->GetDict() == fDict) {
      fProxy = namedProxy->GetProxy();
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

                  if (fDict != finalDataType && fDict != elementClass){
                     Error("CreateProxy", "Wrong data type %s", finalDataType ? finalDataType->GetName() : elementClass ? elementClass->GetName() : "UNKNOWN");
                     fProxy = 0;
                     return;
                  }
               }
            }


            if (!fStaticClassOffsets.size()) {
               Error("CreateProxy()", "The tree does not have a branch called %s. You could check with TTree::Print() for available branches.", fBranchName.Data());
               fProxy = 0;
               return;
            }
         }
         else {
            myLeaf = branch->GetLeaf(TString(leafName(1, leafName.Length())));
            if (!myLeaf){
               Error("CreateProxy()", "The tree does not have a branch, nor a sub-branch called %s. You could check with TTree::Print() for available branches.", fBranchName.Data());
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
               }
               else {
                  Error("CreateProxy()", "Leaf of type %s cannot be read by TTreeReaderValue<%s>.", myLeaf->GetTypeName(), fDict->GetName());
               }
            }
         }
      }
      else {
         Error("CreateProxy()", "The tree does not have a branch called %s. You could check with TTree::Print() for available branches.", fBranchName.Data());
         fProxy = 0;
         return;
      }
   }

   if (!myLeaf && !fStaticClassOffsets.size()){
      const char* branchActualTypeName = GetBranchDataType(branch, branchActualType);

      if (!branchActualType) {
         Error("CreateProxy()", "The branch %s contains data of type %s, which does not have a dictionary.",
               fBranchName.Data(), branchActualTypeName ? branchActualTypeName : "{UNDETERMINED TYPE}");
         fProxy = 0;
         return;
      }

      if (fDict != branchActualType) {
         Error("CreateProxy()", "The branch %s contains data of type %s. It cannot be accessed by a TTreeReaderValue<%s>",
               fBranchName.Data(), branchActualType->GetName(), fDict->GetName());
         return;
      }
   }


   // Update named proxy's dictionary
   if (namedProxy && !namedProxy->GetDict()) {
      namedProxy->SetDict(fDict);
      fProxy = namedProxy->GetProxy();
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
   namedProxy = new ROOT::TNamedBranchProxy(fTreeReader->fDirector, branch, membername);
   fTreeReader->GetProxies()->Add(namedProxy);
   fProxy = namedProxy->GetProxy();
}

//______________________________________________________________________________
const char* ROOT::TTreeReaderValueBase::GetBranchDataType(TBranch* branch,
                                           TDictionary* &dict) const
{
   // Retrieve the type of data stored by branch; put its dictionary into
   // dict, return its type name. If no dictionary is available, at least
   // its type name should be returned.

   dict = 0;
   if (branch->IsA() == TBranchElement::Class()) {
      TBranchElement* brElement = (TBranchElement*)branch;
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

         if (brElement->GetTypeName()) dict = TDictionary::GetDictionary(brElement->GetTypeName());
         if (dict && dict->IsA() == TDataType::Class()){
            dict = TDictionary::GetDictionary(((TDataType*)dict)->GetTypeName());
            if (dict != fDict){
               dict = TClass::GetClass(brElement->GetTypeName());
            }
            if (dict != fDict){
               dict = brElement->GetCurrentClass();
            }
         }
         else if (!dict) {
            dict = brElement->GetCurrentClass();
         }

         return brElement->GetTypeName();
      } else if (brElement->GetType() == TBranchElement::kClonesNode) {
         dict = TClonesArray::Class();
         return "TClonesArray";
      } else if (brElement->GetType() == 31
                 || brElement->GetType() == 41) {
         // it's a member, extract from GetClass()'s streamer info
         Error("GetBranchDataType()", "Must use TTreeReaderValueArray to access a member of an object that is stored in a collection.");
      }
      else {
         Error("GetBranchDataType()", "Unknown type and class combination: %i, %s", brElement->GetType(), brElement->GetClassName());
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
         Error("GetBranchDataType()", "The branch %s was created using a leaf list and cannot be represented as a C++ type. Please access one of its siblings using a TTreeReaderValueArray:", branch->GetName());
         TIter iLeaves(branch->GetListOfLeaves());
         TLeaf* leaf = 0;
         while ((leaf = (TLeaf*) iLeaves())) {
            Error("GetBranchDataType()", "   %s.%s", branch->GetName(), leaf->GetName());
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
      Error("GetBranchDataType()", "The branch %s is a TBranchRef and cannot be represented as a C++ type.", branch->GetName());
      return 0;
   } else {
      Error("GetBranchDataType()", "The branch %s is of type %s - something that is not handled yet.", branch->GetName(), branch->IsA()->GetName());
      return 0;
   }

   return 0;
}

