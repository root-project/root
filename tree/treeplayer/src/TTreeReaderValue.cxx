// @(#)root/treeplayer:$Id$
// Author: Axel Naumann, 2011-09-28

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers and al.        *
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
   fTreeReader(reader),
   fBranchName(branchname),
   fDict(dict),
   fProxy(0),
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
   if (!fProxy) return kReadNothingYet;
   if (fProxy->Read()) {
      fReadStatus = kReadSuccess;
   } else {
      fReadStatus = kReadError;
   }
   return fReadStatus;
}

//______________________________________________________________________________
void ROOT::TTreeReaderValueBase::CreateProxy() {
   // Create the proxy object for our branch.
   if (fProxy) {
      Error("CreateProxy()", "Proxy object for branch %s already exists!",
            fBranchName.Data());
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
            IsA()->GetName(), fBranchName.Data(), brDataType);
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
   if (!branch) {
      Error("CreateProxy()", "The tree does not have a branch called %s. You could check with TTree::Print() for available branches.", fBranchName.Data());
      fProxy = 0;
      return;
   }

   TDictionary* branchActualType = 0;
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
      if (brElement->GetType() == 4) {
         dict = brElement->GetClass();
         return brElement->GetClassName();
      } else if (brElement->GetType() == 3) {
         dict = TClonesArray::Class();
         return "TClonesArray";
      } else if (brElement->GetType() == 31
                 || brElement->GetType() == 41) {
         // it's a member, extract from GetClass()'s streamer info
         Error("GetBranchDataType()", "Must use TTreeReaderValueArray to access a member of an object that is stored in a collection.");
      }
      return 0;
   } else if (branch->IsA() == TBranch::Class()
              || branch->IsA() == TBranchObject::Class()
              || branch->IsA() == TBranchSTL::Class()) {
      const char* dataTypeName = branch->GetClassName();
      if ((!dataTypeName || !dataTypeName[0])
          && branch->IsA() == TBranch::Class()) {
         // leaflist. Can't represent.
         Error("GetBranchDataType()", "The branch %s was created using a leaf list and cannot be represented as a C++ type. Please access one of its siblings using a TTreeReaderValueArray:", branch->GetName());
         TIter iLeaves(branch->GetListOfLeaves());
         TLeaf* leaf = 0;
         while ((leaf = (TLeaf*) iLeaves())) {
            Error("GetBranchDataType()", "   %s.%s", branch->GetName(), leaf->GetName());
         }
         return 0;
      }
      dict = TDictionary::GetDictionary(dataTypeName);
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

