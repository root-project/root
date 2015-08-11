// @(#)root/treeplayer:$Id$
// Author: Axel Naumann, 2011-09-28

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTreeReaderArray.h"

#include "TBranchClones.h"
#include "TBranchElement.h"
#include "TBranchRef.h"
#include "TBranchSTL.h"
#include "TBranchProxyDirector.h"
#include "TClassEdit.h"
#include "TLeaf.h"
#include "TROOT.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TTreeReader.h"
#include "TGenCollectionProxy.h"
#include "TRegexp.h"

// pin vtable
ROOT::TVirtualCollectionReader::~TVirtualCollectionReader() {}

namespace {
   // Reader interface for clones arrays
   class TClonesReader: public ROOT::TVirtualCollectionReader {
   public:
      ~TClonesReader() {}
      TClonesArray* GetCA(ROOT::TBranchProxy* proxy) {
         if (!proxy->Read()){
            fReadStatus = ROOT::TTreeReaderValueBase::kReadError;
            Error("GetCA()", "Read error in TBranchProxy.");
            return 0;
         }
         fReadStatus = ROOT::TTreeReaderValueBase::kReadSuccess;
         return (TClonesArray*) proxy->GetWhere();
      }
      virtual size_t GetSize(ROOT::TBranchProxy* proxy) {
         TClonesArray *myClonesArray = GetCA(proxy);
         if (myClonesArray){
            return myClonesArray->GetEntries();
         }
         else return 0;
      }
      virtual void* At(ROOT::TBranchProxy* proxy, size_t idx) {
         TClonesArray *myClonesArray = GetCA(proxy);
         if (myClonesArray){
            return myClonesArray->UncheckedAt(idx);
         }
         else return 0;
      }
   };

   // Reader interface for STL
   class TSTLReader: public ROOT::TVirtualCollectionReader {
   public:
      ~TSTLReader() {}
      TVirtualCollectionProxy* GetCP(ROOT::TBranchProxy* proxy) {
         if (!proxy->Read()){
            fReadStatus = ROOT::TTreeReaderValueBase::kReadError;
            Error("GetCP()", "Read error in TBranchProxy.");
            return 0;
         }
         fReadStatus = ROOT::TTreeReaderValueBase::kReadSuccess;
         return (TVirtualCollectionProxy*) proxy->GetCollection();
      }

      virtual size_t GetSize(ROOT::TBranchProxy* proxy) {
         if (!CheckProxy(proxy)) return -1;
         if (!proxy->ReadEntries()) return -1;
         TVirtualCollectionProxy *myCollectionProxy = GetCP(proxy);
         if (!myCollectionProxy) return 0;
         return myCollectionProxy->Size();
      }

      Bool_t CheckProxy(ROOT::TBranchProxy *proxy) {
         if (!proxy->Read()) return false;
         if (proxy->IsaPointer()) {
            if (proxy->GetWhere() && *(void**)proxy->GetWhere()){
               ((TGenCollectionProxy*)proxy->GetCollection())->PopProxy();
               ((TGenCollectionProxy*)proxy->GetCollection())->PushProxy(*(void**)proxy->GetWhere());
            }
            else return false;
         }
         return true;
      }

      virtual void* At(ROOT::TBranchProxy* proxy, size_t idx) {
         if (!CheckProxy(proxy)) return 0;
         if (!proxy->Read()) return 0;
         if (!proxy->GetWhere()) return 0;

         if (proxy->GetCollection()->HasPointers()){
            return *(void**)proxy->GetCollection()->At(idx);
         }
         else {
            return proxy->GetCollection()->At(idx);
         }
      }
   };

   class TCollectionLessSTLReader : public ROOT::TVirtualCollectionReader {
   private:
      TVirtualCollectionProxy *localCollection;
   public:
      TCollectionLessSTLReader(TVirtualCollectionProxy *proxy) : localCollection(proxy) {}

      TVirtualCollectionProxy* GetCP(ROOT::TBranchProxy* proxy) {
         if (!proxy->Read()){
            fReadStatus = ROOT::TTreeReaderValueBase::kReadError;
            Error("GetCP()", "Read error in TBranchProxy.");
            return 0;
         }
         fReadStatus = ROOT::TTreeReaderValueBase::kReadSuccess;
         return localCollection;
      }

      virtual size_t GetSize(ROOT::TBranchProxy* proxy) {
         if (!proxy->ReadEntries()) return -1;
         TVirtualCollectionProxy *myCollectionProxy = GetCP(proxy);
         if (!myCollectionProxy) return 0;
         if (!proxy->GetWhere()) return 0;
         TVirtualCollectionProxy::TPushPop ppRaii(myCollectionProxy, proxy->GetWhere());
         return myCollectionProxy->Size();
      }

      virtual void* At(ROOT::TBranchProxy* proxy, size_t idx) {
         if (!proxy->Read()) return 0;
         if (!proxy->GetWhere()) return 0;

         TVirtualCollectionProxy *myCollectionProxy = GetCP(proxy);
         if (!myCollectionProxy) return 0;
         TVirtualCollectionProxy::TPushPop ppRaii(myCollectionProxy, proxy->GetWhere());
         if (myCollectionProxy->HasPointers()){
            return *(void**)myCollectionProxy->At(idx);
         }
         else {
            return myCollectionProxy->At(idx);
         }
      }
   };


   // Reader interface for leaf list
   // SEE TTreeProxyGenerator.cxx:1319: '//We have a top level raw type'
   class TObjectArrayReader: public ROOT::TVirtualCollectionReader {
   private:
      Int_t basicTypeSize;
   public:
      TObjectArrayReader() : basicTypeSize(-1) { }
      ~TObjectArrayReader() {}
      TVirtualCollectionProxy* GetCP(ROOT::TBranchProxy* proxy) {
         if (!proxy->Read()){
            fReadStatus = ROOT::TTreeReaderValueBase::kReadError;
            Error("GetCP()", "Read error in TBranchProxy.");
            return 0;
         }
         fReadStatus = ROOT::TTreeReaderValueBase::kReadSuccess;
         return (TVirtualCollectionProxy*) proxy->GetCollection();
      }
      virtual size_t GetSize(ROOT::TBranchProxy* proxy) {
         TVirtualCollectionProxy *myCollectionProxy = GetCP(proxy);
         if (!myCollectionProxy) return 0;
         return myCollectionProxy->Size();
      }
      virtual void* At(ROOT::TBranchProxy* proxy, size_t idx) {
         if (!proxy->Read()) return 0;

         Int_t objectSize;
         void *array = (void*)proxy->GetStart();

         if (basicTypeSize == -1){
            TClass *myClass = proxy->GetClass();
            if (!myClass){
               Error("At()", "Cannot get class info from branch proxy.");
               return 0;
            }
            objectSize = myClass->GetClassSize();
         }
         else {
            objectSize = basicTypeSize;
         }
         return (void*)((Byte_t*)array + (objectSize * idx));
      }

      void SetBasicTypeSize(Int_t size){
         basicTypeSize = size;
      }
   };

   class TArrayParameterSizeReader : public TObjectArrayReader {
   private:
      TTreeReaderValue<Int_t> indexReader;
   public:
      TArrayParameterSizeReader(TTreeReader *treeReader, const char *branchName) : indexReader(*treeReader, branchName) {}

      virtual size_t GetSize(ROOT::TBranchProxy* /*proxy*/){ return *indexReader; }
   };

   // Reader interface for fixed size arrays
   class TArrayFixedSizeReader : public TObjectArrayReader {
   private:
      Int_t size;

   public:
      TArrayFixedSizeReader(Int_t sizeArg) : size(sizeArg) {}

      virtual size_t GetSize(ROOT::TBranchProxy* /*proxy*/) { return size; }
   };

   class TBasicTypeArrayReader : public ROOT::TVirtualCollectionReader {
   public:
      ~TBasicTypeArrayReader() {}

      TVirtualCollectionProxy* GetCP (ROOT::TBranchProxy *proxy) {
         if (!proxy->Read()){
            fReadStatus = ROOT::TTreeReaderValueBase::kReadError;
            Error("GetCP()", "Read error in TBranchProxy.");
            return 0;
         }
         fReadStatus = ROOT::TTreeReaderValueBase::kReadSuccess;
         return (TVirtualCollectionProxy*) proxy->GetCollection();
      }

      virtual size_t GetSize(ROOT::TBranchProxy* proxy){
         TVirtualCollectionProxy *myCollectionProxy = GetCP(proxy);
         if (!myCollectionProxy) return 0;
         return myCollectionProxy->Size();
      }

      virtual void* At(ROOT::TBranchProxy* proxy, size_t idx){
         TVirtualCollectionProxy *myCollectionProxy = GetCP(proxy);
         if (!myCollectionProxy) return 0;
         return (Byte_t*)myCollectionProxy->At(idx) + proxy->GetOffset();
      }
   };

   class TBasicTypeClonesReader : public TClonesReader {
   private:
      Int_t offset;
   public:
      TBasicTypeClonesReader(Int_t offsetArg) : offset(offsetArg) {}

      virtual void* At(ROOT::TBranchProxy* proxy, size_t idx){
         TClonesArray *myClonesArray = GetCA(proxy);
         if (!myClonesArray) return 0;
         return (Byte_t*)myClonesArray->At(idx) + offset;
      }
   };

   class TLeafReader : public ROOT::TVirtualCollectionReader {
   private:
      ROOT::TTreeReaderValueBase *valueReader;
      Int_t elementSize;
   public:
      TLeafReader(ROOT::TTreeReaderValueBase *valueReaderArg) : valueReader(valueReaderArg), elementSize(-1) {}

      virtual size_t GetSize(ROOT::TBranchProxy* /*proxy*/){
         TLeaf *myLeaf = valueReader->GetLeaf();
         return myLeaf ? myLeaf->GetLen() : 0; // Error will be printed by GetLeaf
      }

      virtual void* At(ROOT::TBranchProxy* /*proxy*/, size_t idx){
         ProxyRead();
         void *address = valueReader->GetAddress();
         if (elementSize == -1){
            TLeaf *myLeaf = valueReader->GetLeaf();
            if (!myLeaf) return 0; // Error will be printed by GetLeaf
            elementSize = myLeaf->GetLenType();
         }
         return (Byte_t*)address + (elementSize * idx);
      }

   protected:
      void ProxyRead(){
         valueReader->ProxyRead();
      }
   };

   class TLeafParameterSizeReader : public TLeafReader {
   private:
      TTreeReaderValue<Int_t> sizeReader;
   public:
      TLeafParameterSizeReader(TTreeReader *treeReader, const char *leafName, ROOT::TTreeReaderValueBase *valueReaderArg) : TLeafReader(valueReaderArg), sizeReader(*treeReader, leafName) {}

      virtual size_t GetSize(ROOT::TBranchProxy* /*proxy*/){
         ProxyRead();
         return *sizeReader;
      }
   };
}

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// TTreeReaderArray                                                        //
//                                                                            //
// Extracts array data from a TTree.                                          //
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

ClassImp(TTreeReaderArrayBase)

//______________________________________________________________________________
void ROOT::TTreeReaderArrayBase::CreateProxy()
{
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
         TDictionary* dictUnused = 0;
         brDataType = GetBranchDataType(br, dictUnused);
      }
      Error("CreateProxy()", "The template argument type T of %s accessing branch %s (which contains data of type %s) is not known to ROOT. You will need to create a dictionary for it.",
            GetDerivedTypeName(), fBranchName.Data(), brDataType);
      return;
   }

   // Access a branch's collection content (not the collection itself)
   // through a proxy.
   // Search for the branchname, determine what it contains, and wire the
   // TBranchProxy representing it to us so we can access its data.

   ROOT::TNamedBranchProxy* namedProxy = fTreeReader->FindProxy(fBranchName);
   if (namedProxy && namedProxy->GetContentDict() == fDict) {
      fProxy = namedProxy->GetProxy();
      if (!fImpl){
         Fatal("CreateProxy()", "No fImpl set!");
      }
      return;
   }


   TDictionary* branchActualType = 0;
   TBranch* branch = fTreeReader->GetTree()->GetBranch(fBranchName);
   TLeaf *myLeaf = NULL;
   if (!branch) {
      if (fBranchName.Contains(".")){
         TRegexp leafNameExpression ("\\.[a-zA-Z0-9_]+$");
         TString leafName (fBranchName(leafNameExpression));
         TString branchName = fBranchName(0, fBranchName.Length() - leafName.Length());
         branch = fTreeReader->GetTree()->GetBranch(branchName);
         if (!branch){
            Error("CreateProxy()", "The tree does not have a branch called %s. You could check with TTree::Print() for available branches.", fBranchName.Data());
            fProxy = 0;
            return;
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
               if (!tempDict){
                  Error("CreateProxy()", "Failed to get the dictionary for %s.", myLeaf->GetTypeName());
                  fProxy = 0;
                  return;
               }
               else if (tempDict->IsA() == TDataType::Class() && TDictionary::GetDictionary(((TDataType*)tempDict)->GetTypeName()) == fDict){
                  //fLeafOffset = myLeaf->GetOffset() / 4;
                  branchActualType = fDict;
                  fLeaf = myLeaf;
                  fBranchName = branchName;
                  fLeafName = leafName(1, leafName.Length());
               }
               else {
                  Error("CreateProxy()", "Leaf of type %s cannot be read by TTreeReaderValue<%s>.", myLeaf->GetTypeName(), fDict->GetName());
                  fProxy = 0;
                  return;
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

   // Update named proxy's dictionary
   if (namedProxy && !namedProxy->GetContentDict()) {
      namedProxy->SetContentDict(fDict);
      fProxy = namedProxy->GetProxy();
   }
   else {
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

   if (!myLeaf){
      TString branchActualTypeName;
      const char* nonCollTypeName = GetBranchContentDataType(branch, branchActualTypeName, branchActualType);
      if (nonCollTypeName) {
         Error("CreateContentProxy()", "The branch %s contains data of type %s, which should be accessed through a TTreeReaderValue< %s >.",
               fBranchName.Data(), nonCollTypeName, nonCollTypeName);
         fProxy = 0;
         return;
      }
      if (!branchActualType) {
         if (branchActualTypeName.IsNull()) {
            Error("CreateContentProxy()", "Cannot determine the type contained in the collection of branch %s. That's weird - please report!",
                  fBranchName.Data());
         } else {
            Error("CreateContentProxy()", "The branch %s contains data of type %s, which does not have a dictionary.",
                  fBranchName.Data(), branchActualTypeName.Data());
         }
         fProxy = 0;
         return;
      }

      if (fDict != branchActualType) {
         Error("CreateContentProxy()", "The branch %s contains data of type %s. It cannot be accessed by a TTreeReaderArray<%s>",
               fBranchName.Data(), branchActualType->GetName(), fDict->GetName());

         // Update named proxy's dictionary
         if (namedProxy && !namedProxy->GetContentDict()) {
            namedProxy->SetContentDict(fDict);
         }

         // fProxy = 0;
         // return;
      }
   }



   // Access a branch's collection content (not the collection itself)
   // through a proxy.
   // Search for the branchname, determine what it contains, and wire the
   // TBranchProxy representing it to us so we can access its data.
   // A proxy for branch must not have been created before (i.e. check
   // fProxies before calling this function!)

   if (myLeaf){
      if (!myLeaf->GetLeafCount()){
         fImpl = new TLeafReader(this);
      }
      else {
         TString leafFullName = myLeaf->GetBranch()->GetName();
         leafFullName += ".";
         leafFullName += myLeaf->GetLeafCount()->GetName();
         fImpl = new TLeafParameterSizeReader(fTreeReader, leafFullName.Data(), this);
      }
   }
   else if (branch->IsA() == TBranchElement::Class()) {
      TBranchElement* branchElement = ((TBranchElement*)branch);

      TStreamerInfo *streamerInfo = branchElement->GetInfo();
      Int_t id = branchElement->GetID();

      if (id >= 0){ // Not root node?
         // Int_t offset = streamerInfo->GetOffsets()[id];
         TStreamerElement *element = (TStreamerElement*)streamerInfo->GetElements()->At(id);
         // Bool_t isPointer = element->IsaPointer();
         // TClass *classPointer = element->GetClassPointer();

         if (element->IsA() == TStreamerSTL::Class()){
            fImpl = new TSTLReader();
         }
         else if (element->IsA() == TStreamerObject::Class()){
            //fImpl = new TObjectArrayReader(); // BArray[12]

            if (element->GetClass() == TClonesArray::Class()){
               fImpl = new TClonesReader();
            }
            else {
               fImpl = new TArrayFixedSizeReader(element->GetArrayLength());
            }
         }
         else if (element->IsA() == TStreamerLoop::Class()) {
            fImpl = new TArrayParameterSizeReader(fTreeReader, branchElement->GetBranchCount()->GetName());
         }
         else if (element->IsA() == TStreamerBasicType::Class()){
            if (branchElement->GetType() == TBranchElement::kSTLMemberNode){
               fImpl = new TBasicTypeArrayReader();
            }
            else if (branchElement->GetType() == TBranchElement::kClonesMemberNode){
               fImpl = new TBasicTypeClonesReader(element->GetOffset());
            }
            else {
               fImpl = new TArrayFixedSizeReader(element->GetArrayLength());
               ((TObjectArrayReader*)fImpl)->SetBasicTypeSize(((TDataType*)fDict)->Size());
            }
         }
         else if (element->IsA() == TStreamerBase::Class()){
            fImpl = new TClonesReader();
         }
      }
      else { // We are at root node?
         if (branchElement->GetClass()->GetCollectionProxy()){
            fImpl = new TCollectionLessSTLReader(branchElement->GetClass()->GetCollectionProxy());
         }
      }
   } else if (branch->IsA() == TBranch::Class()) {
      TLeaf *topLeaf = branch->GetLeaf(branch->GetName());
      if (!topLeaf) {
         Error("CreateProxy", "Failed to get the top leaf from the branch");
         return;
      }
      Int_t size = 0;
      TLeaf *sizeLeaf = topLeaf->GetLeafCounter(size);
      if (!sizeLeaf) {
         fImpl = new TArrayFixedSizeReader(size);
      }
      else {
         fImpl = new TArrayParameterSizeReader(fTreeReader, sizeLeaf->GetName());
      }
      ((TObjectArrayReader*)fImpl)->SetBasicTypeSize(((TDataType*)fDict)->Size());
   } else if (branch->IsA() == TBranchClones::Class()) {
      Error("CreateProxy", "Support for branches of type TBranchClones not implemented");
   } else if (branch->IsA() == TBranchObject::Class()) {
      Error("CreateProxy", "Support for branches of type TBranchObject not implemented");
   } else if (branch->IsA() == TBranchSTL::Class()) {
      Error("CreateProxy", "Support for branches of type TBranchSTL not implemented");
      fImpl = new TSTLReader();
   } else if (branch->IsA() == TBranchRef::Class()) {
      Error("CreateProxy", "Support for branches of type TBranchRef not implemented");
   }
}

//______________________________________________________________________________
const char* ROOT::TTreeReaderArrayBase::GetBranchContentDataType(TBranch* branch,
                                                                 TString& contentTypeName,
                                                                 TDictionary* &dict) const
{
   // Access a branch's collection content (not the collection itself)
   // through a proxy.
   // Retrieve the type of data contained in the collection stored by branch;
   // put its dictionary into dict, If there is no dictionary, put its type
   // name into contentTypeName.
   // The contentTypeName is set to NULL if the branch does not
   // contain a collection; in that case, the type of the branch is returned.
   // In all other cases, NULL is returned.

   dict = 0;
   contentTypeName = "";
   if (branch->IsA() == TBranchElement::Class()) {
      TBranchElement* brElement = (TBranchElement*)branch;
      if (brElement->GetType() == 4
          || brElement->GetType() == 3) {
         TVirtualCollectionProxy* collProxy = brElement->GetCollectionProxy();
         if (collProxy) {
            TClass *myClass = collProxy->GetValueClass();
            if (!myClass){
               Error("GetBranchContentDataType()", "Could not get value class.");
               return 0;
            }
            dict = TDictionary::GetDictionary(myClass->GetName());
            if (!dict) dict = TDataType::GetDataType(collProxy->GetType());
         }
         if (!dict) {
            // We don't know the dictionary, thus we need the content's type name.
            // Determine it.
            if (brElement->GetType() == 3) {
               contentTypeName = brElement->GetClonesName();
               dict = TDictionary::GetDictionary(brElement->GetClonesName());
               return 0;
            }
            // STL:
            TClassEdit::TSplitType splitType(brElement->GetClassName());
            int isSTLCont = splitType.IsSTLCont();
            if (!isSTLCont) {
               Error("GetBranchContentDataType()", "Cannot determine STL collection type of %s stored in branch %s", brElement->GetClassName(), branch->GetName());
               return brElement->GetClassName();
            }
            bool isMap = isSTLCont == ROOT::kSTLmap
               || isSTLCont == ROOT::kSTLmultimap;
            if (isMap) contentTypeName = "std::pair< ";
            contentTypeName += splitType.fElements[1];
            if (isMap) {
               contentTypeName += splitType.fElements[2];
               contentTypeName += " >";
            }
            return 0;
         }
         return 0;
      } else if (brElement->GetType() == 31
                 || brElement->GetType() == 41) {
         // it's a member, extract from GetClass()'s streamer info
         TClass* clData = 0;
         EDataType dtData = kOther_t;
         int ExpectedTypeRet = brElement->GetExpectedType(clData, dtData);
         if (ExpectedTypeRet == 0) {
            dict = clData;
            if (!dict) {
               dict = TDataType::GetDataType(dtData);
            }
            if (!dict) {
               Error("GetBranchContentDataType()", "The branch %s contains a data type %d for which the dictionary cannot be retrieved.",
                     branch->GetName(), (int)dtData);
               contentTypeName = TDataType::GetTypeName(dtData);
               return 0;
            }
            return 0;
         } else if (ExpectedTypeRet == 1) {
            int brID = brElement->GetID();
            if (brID == -1) {
               // top
               Error("GetBranchContentDataType()", "The branch %s contains data of type %s for which the dictionary does not exist. It's needed.",
                     branch->GetName(), brElement->GetClassName());
               contentTypeName = brElement->GetClassName();
               return 0;
            }
            // Either the data type name doesn't have an EDataType entry
            // or the streamer info doesn't have a TClass* attached.
            TStreamerElement* element =
               (TStreamerElement*) brElement->GetInfo()->GetElement(brID);
            contentTypeName = element->GetTypeName();
            return 0;
         }
         /* else (ExpectedTypeRet == 2)*/
         // The streamer info entry cannot be found.
         // TBranchElement::GetExpectedType() has already complained.
         return "{CANNOT DETERMINE TBranchElement DATA TYPE}";
      }
      else if (brElement->GetType() == TBranchElement::kLeafNode){
         TStreamerInfo *streamerInfo = brElement->GetInfo();
         Int_t id = brElement->GetID();

         if (id >= 0){
            TStreamerElement *element = (TStreamerElement*)streamerInfo->GetElements()->At(id);

            if (element->IsA() == TStreamerSTL::Class()){
               TClass *myClass = brElement->GetCurrentClass();
               if (!myClass){
                  Error("GetBranchDataType()", "Could not get class from branch element.");
                  return 0;
               }
               TVirtualCollectionProxy *myCollectionProxy = myClass->GetCollectionProxy();
               if (!myCollectionProxy){
                  Error("GetBranchDataType()", "Could not get collection proxy from STL class");
                  return 0;
               }
               // Try getting the contained class
               dict = myCollectionProxy->GetValueClass();
               // If it fails, try to get the contained type as a primitive type
               if (!dict) dict = TDataType::GetDataType(myCollectionProxy->GetType());
               if (!dict){
                  Error("GetBranchDataType()", "Could not get valueClass from collectionProxy.");
                  return 0;
               }
               contentTypeName = dict->GetName();
               return 0;
            }
            else if (element->IsA() == TStreamerObject::Class() && !strcmp(element->GetTypeName(), "TClonesArray")){
               if (!fProxy->Setup() || !fProxy->Read()){
                  Error("GetBranchContentDataType()", "Failed to get type from proxy, unable to check type");
                  contentTypeName = "UNKNOWN";
                  dict = 0;
                  return contentTypeName;
               }
               TClonesArray *myArray = (TClonesArray*)fProxy->GetWhere();
               dict = myArray->GetClass();
               contentTypeName = dict->GetName();
               return 0;
            }
            else {
               dict = brElement->GetCurrentClass();
               if (!dict) {
                  TDictionary *myDataType = TDictionary::GetDictionary(brElement->GetTypeName());
                  dict = TDataType::GetDataType((EDataType)((TDataType*)myDataType)->GetType());
               }
               contentTypeName = brElement->GetTypeName();
               return 0;
            }
         }
         if (brElement->GetCurrentClass() == TClonesArray::Class()){
            contentTypeName = "TClonesArray";
            Warning("GetBranchContentDataType()", "Not able to check type correctness, ignoring check");
            dict = fDict;
         }
         else if (!dict && (branch->GetSplitLevel() == 0 || brElement->GetClass()->GetCollectionProxy())){
            // Try getting the contained class
            dict = brElement->GetClass()->GetCollectionProxy()->GetValueClass();
            // If it fails, try to get the contained type as a primitive type
            if (!dict) dict = TDataType::GetDataType(brElement->GetClass()->GetCollectionProxy()->GetType());
            if (dict) contentTypeName = dict->GetName();
            return 0;
         }
         else if (!dict){
            dict = brElement->GetClass();
            contentTypeName = dict->GetName();
            return 0;
         }

         return 0;
      }
      return 0;
   } else if (branch->IsA() == TBranch::Class()
              || branch->IsA() == TBranchObject::Class()
              || branch->IsA() == TBranchSTL::Class()) {
      const char* dataTypeName = branch->GetClassName();
      if ((!dataTypeName || !dataTypeName[0])
          && branch->IsA() == TBranch::Class()) {
         TLeaf *myLeaf = branch->GetLeaf(branch->GetName());
         if (myLeaf){
            TDictionary *myDataType = TDictionary::GetDictionary(myLeaf->GetTypeName());
            if (myDataType && myDataType->IsA() == TDataType::Class()){
               dict = TDataType::GetDataType((EDataType)((TDataType*)myDataType)->GetType());
               contentTypeName = myLeaf->GetTypeName();
               return 0;
            }
         }

         // leaflist. Can't represent.
         Error("GetBranchContentDataType()", "The branch %s was created using a leaf list and cannot be represented as a C++ type. Please access one of its siblings using a TTreeReaderValueArray:", branch->GetName());
         TIter iLeaves(branch->GetListOfLeaves());
         TLeaf* leaf = 0;
         while ((leaf = (TLeaf*) iLeaves())) {
            Error("GetBranchContentDataType()", "   %s.%s", branch->GetName(), leaf->GetName());
         }
         return 0;
      }
      if (dataTypeName) dict = TDictionary::GetDictionary(dataTypeName);
      if (branch->IsA() == TBranchSTL::Class()){
         Warning("GetBranchContentDataType()", "Not able to check type correctness, ignoring check");
         dict = fDict;
         return 0;
      }
      return dataTypeName;
   } else if (branch->IsA() == TBranchClones::Class()) {
      dict = TClonesArray::Class();
      return "TClonesArray";
   } else if (branch->IsA() == TBranchRef::Class()) {
      // Can't represent.
      Error("GetBranchContentDataType()", "The branch %s is a TBranchRef and cannot be represented as a C++ type.", branch->GetName());
      return 0;
   } else {
      Error("GetBranchContentDataType()", "The branch %s is of type %s - something that is not handled yet.", branch->GetName(), branch->IsA()->GetName());
      return 0;
   }

   return 0;
}
