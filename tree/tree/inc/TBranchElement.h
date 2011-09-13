// @(#)root/tree:$Id$
// Author: Rene Brun   14/01/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBranchElement
#define ROOT_TBranchElement


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBranchElement                                                       //
//                                                                      //
// A Branch for the case of an object.                                  //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TBranch
#include "TBranch.h"
#endif

#ifndef ROOT_TClassRef
#include "TClassRef.h"
#endif

#ifndef ROOT_TTree
#include "TTree.h"
#endif

#ifndef ROOT_TError
#include "TError.h"
#endif

#include <vector>

class TFolder;
class TStreamerInfo;
class TVirtualCollectionProxy;
class TVirtualCollectionIterators;
class TVirtualCollectionPtrIterators;
class TVirtualArray;

namespace TStreamerInfoActions { class TActionSequence; }


class TBranchElement : public TBranch {

// Friends
   friend class TTreeCloner;

// Types
protected:
   enum {
      kBranchFolder = BIT(14),
      kDeleteObject = BIT(16),  //  We are the owner of fObject.
      kCache        = BIT(18),  //  Need to pushd/pop fOnfileObject.
      kOwnOnfileObj = BIT(19),  //  We are the owner of fOnfileObject.
      kAddressSet   = BIT(20),  //  The addressing set have been called for this branch
      kMakeClass    = BIT(21),  //  This branch has been switched to using the MakeClass Mode
      kDecomposedObj= BIT(21)   //  More explicit alias for kMakeClass.
   };

// Data Members
protected:
   TString                  fClassName;     //  Class name of referenced object
   TString                  fParentName;    //  Name of parent class
   TString                  fClonesName;    //  Name of class in TClonesArray (if any)
   TVirtualCollectionProxy *fCollProxy;     //! collection interface (if any)
   UInt_t                   fCheckSum;      //  CheckSum of class
   Int_t                    fClassVersion;  //  Version number of class
   Int_t                    fID;            //  element serial number in fInfo
   Int_t                    fType;          //  branch type
   Int_t                    fStreamerType;  //  branch streamer type
   Int_t                    fMaximum;       //  Maximum entries for a TClonesArray or variable array
   Int_t                    fSTLtype;       //! STL container type
   Int_t                    fNdata;         //! Number of data in this branch
   TBranchElement          *fBranchCount;   //  pointer to primary branchcount branch
   TBranchElement          *fBranchCount2;  //  pointer to secondary branchcount branch
   TStreamerInfo           *fInfo;          //! Pointer to StreamerInfo
   char                    *fObject;        //! Pointer to object at *fAddress
   TVirtualArray           *fOnfileObject;  //! Place holder for the onfile representation of data members.
   Bool_t                   fInit;          //! Initialization flag for branch assignment
   Bool_t                   fInitOffsets;   //! Initialization flag to not endlessly recalculate offsets
   TClassRef                fTargetClass;   //! Reference to the target in-memory class
   TClassRef                fCurrentClass;  //! Reference to current (transient) class definition
   TClassRef                fParentClass;   //! Reference to class definition in fParentName
   TClassRef                fBranchClass;   //! Reference to class definition in fClassName
   TClassRef                fClonesClass;   //! Reference to class definition in fClonesName
   Int_t                   *fBranchOffset;  //! Sub-Branch offsets with respect to current transient class
   Int_t                    fBranchID;      //! ID number assigned by a TRefTable.
   std::vector<Int_t>       fIDs;           //! List of the serial number of all the StreamerInfo to be used.
   TStreamerInfoActions::TActionSequence *fReadActionSequence;  //! Set of actions to be executed to extract the data from the basket.
   TStreamerInfoActions::TActionSequence *fFillActionSequence; //! Set of actions to be executed to write the data to the basket.
   TVirtualCollectionIterators           *fIterators;     //! holds the iterators when the branch is of fType==4.
   TVirtualCollectionPtrIterators        *fPtrIterators;  //! holds the iterators when the branch is of fType==4 and it is a split collection of pointers.

// Not implemented
private:
   TBranchElement(const TBranchElement&);            // not implemented
   TBranchElement& operator=(const TBranchElement&); // not implemented

   static void SwitchContainer(TObjArray *);

// Implementation use only functions.
protected:
   void                     BuildTitle(const char* name);
   virtual void             InitializeOffsets();
   virtual void             InitInfo();
   Bool_t                   IsMissingCollection() const;
   TClass                  *GetCurrentClass(); // Class referenced by transient description
   TClass                  *GetParentClass(); // Class referenced by fParentName
   TStreamerInfo           *GetInfoImp() const;
   void                     ReleaseObject();
   void                     SetBranchCount(TBranchElement* bre);
   void                     SetBranchCount2(TBranchElement* bre) { fBranchCount2 = bre; }
   Int_t                    Unroll(const char* name, TClass* cltop, TClass* cl, char* ptr, Int_t basketsize, Int_t splitlevel, Int_t btype);
   inline void              ValidateAddress() const;

   void Init(TTree *tree, TBranch *parent, const char* name, TStreamerInfo* sinfo, Int_t id, char* pointer, Int_t basketsize = 32000, Int_t splitlevel = 0, Int_t btype = 0);
   void Init(TTree *tree, TBranch *parent, const char* name, TClonesArray* clones, Int_t basketsize = 32000, Int_t splitlevel = 0, Int_t compress = -1);
   void Init(TTree *tree, TBranch *parent, const char* name, TVirtualCollectionProxy* cont, Int_t basketsize = 32000, Int_t splitlevel = 0, Int_t compress = -1);

   void ReadLeavesImpl(TBuffer& b);
   void ReadLeavesMakeClass(TBuffer& b);
   void ReadLeavesCollection(TBuffer& b);
   void ReadLeavesCollectionSplitPtrMember(TBuffer& b);
   void ReadLeavesCollectionSplitVectorPtrMember(TBuffer& b);
   void ReadLeavesCollectionMember(TBuffer& b);
   void ReadLeavesClones(TBuffer& b);
   void ReadLeavesClonesMember(TBuffer& b);
   void ReadLeavesCustomStreamer(TBuffer& b);
   void ReadLeavesMember(TBuffer& b);
   void ReadLeavesMemberBranchCount(TBuffer& b);
   void ReadLeavesMemberCounter(TBuffer& b);
   void SetReadLeavesPtr();
   void SetReadActionSequence();
   void SetupAddressesImpl();
   
   void FillLeavesImpl(TBuffer& b);
   void FillLeavesMakeClass(TBuffer& b);
   void FillLeavesCollection(TBuffer& b);
   void FillLeavesCollectionSplitVectorPtrMember(TBuffer& b);
   void FillLeavesCollectionSplitPtrMember(TBuffer& b);
   void FillLeavesCollectionMember(TBuffer& b);
   void FillLeavesClones(TBuffer& b);
   void FillLeavesClonesMember(TBuffer& b);
   void FillLeavesCustomStreamer(TBuffer& b);
   void FillLeavesMemberBranchCount(TBuffer& b);
   void FillLeavesMemberCounter(TBuffer& b);
   void FillLeavesMember(TBuffer& b);
   void SetFillLeavesPtr();
   void SetFillActionSequence();
// Public Interface.
public:
   TBranchElement();
   TBranchElement(TTree *tree, const char* name, TStreamerInfo* sinfo, Int_t id, char* pointer, Int_t basketsize = 32000, Int_t splitlevel = 0, Int_t btype = 0);
   TBranchElement(TTree *tree, const char* name, TClonesArray* clones, Int_t basketsize = 32000, Int_t splitlevel = 0, Int_t compress = -1);
   TBranchElement(TTree *tree, const char* name, TVirtualCollectionProxy* cont, Int_t basketsize = 32000, Int_t splitlevel = 0, Int_t compress = -1);
   TBranchElement(TBranch *parent, const char* name, TStreamerInfo* sinfo, Int_t id, char* pointer, Int_t basketsize = 32000, Int_t splitlevel = 0, Int_t btype = 0);
   TBranchElement(TBranch *parent, const char* name, TClonesArray* clones, Int_t basketsize = 32000, Int_t splitlevel = 0, Int_t compress = -1);
   TBranchElement(TBranch *parent, const char* name, TVirtualCollectionProxy* cont, Int_t basketsize = 32000, Int_t splitlevel = 0, Int_t compress = -1);

   virtual                  ~TBranchElement();

   virtual void             Browse(TBrowser* b);
   virtual Int_t            Fill();
   virtual TBranch         *FindBranch(const char *name);
   virtual TLeaf           *FindLeaf(const char *name);
   virtual char            *GetAddress() const;
           TBranchElement  *GetBranchCount() const { return fBranchCount; }
           TBranchElement  *GetBranchCount2() const { return fBranchCount2; }
           Int_t           *GetBranchOffset() const { return fBranchOffset; }
           UInt_t           GetCheckSum() { return fCheckSum; }
   virtual const char      *GetClassName() const { return fClassName.Data(); }
   virtual TClass          *GetClass() const { return fBranchClass; }
   virtual const char      *GetClonesName() const { return fClonesName.Data(); }
   TVirtualCollectionProxy *GetCollectionProxy();
   virtual Int_t            GetEntry(Long64_t entry = 0, Int_t getall = 0);
   virtual Int_t            GetExpectedType(TClass *&clptr,EDataType &type);
           const char      *GetIconName() const;
           Int_t            GetID() const { return fID; }
           TStreamerInfo   *GetInfo() const;
           Bool_t           GetMakeClass() const;
           char            *GetObject() const;
   virtual const char      *GetParentName() const { return fParentName.Data(); }
   virtual Int_t            GetMaximum() const;
           Int_t            GetNdata() const { return fNdata; }
           Int_t            GetType() const { return fType; }
           Int_t            GetStreamerType() const { return fStreamerType; }
   virtual TClass          *GetTargetClass() { return fTargetClass; }
   virtual const char      *GetTypeName() const;
           Double_t         GetValue(Int_t i, Int_t len, Bool_t subarr = kFALSE) const;
   virtual void            *GetValuePointer() const;
           Int_t            GetClassVersion() { return fClassVersion; }
           Bool_t           IsBranchFolder() const { return TestBit(kBranchFolder); }
           Bool_t           IsFolder() const;
   virtual Bool_t           IsObjectOwner() const { return TestBit(kDeleteObject); }
   virtual Bool_t           Notify() { if (fAddress) { ResetAddress(); } return 1; }
   virtual void             Print(Option_t* option = "") const;
           void             PrintValue(Int_t i) const;
   virtual void             Reset(Option_t* option = "");
   virtual void             ResetAfterMerge(TFileMergeInfo *);
   virtual void             ResetAddress();
   virtual void             ResetDeleteObject();
   virtual void             SetAddress(void* addobj);
   virtual Bool_t           SetMakeClass(Bool_t decomposeObj = kTRUE);
   virtual void             SetObject(void *objadd);
   virtual void             SetBasketSize(Int_t buffsize);
   virtual void             SetBranchFolder() { SetBit(kBranchFolder); }
   virtual void             SetClassName(const char* name) { fClassName = name; }
   virtual void             SetOffset(Int_t offset);
   inline  void             SetParentClass(TClass* clparent);
   virtual void             SetParentName(const char* name) { fParentName = name; }
   virtual void             SetTargetClass(const char *name);
   virtual void             SetupAddresses();
   virtual void             SetType(Int_t btype) { fType = btype; }
   virtual void             UpdateFile();

   ClassDef(TBranchElement,9)  // Branch in case of an object
};

inline void TBranchElement::SetParentClass(TClass* clparent)
{
   fParentClass = clparent;
   fParentName = clparent ? clparent->GetName() : "";
}

inline void TBranchElement::ValidateAddress() const
{
   // Check to see if the user changed the object pointer without telling us.

   if (fID < 0) {
      // We are a top-level branch.
      if (!fTree->GetMakeClass() && fAddress && (*((char**) fAddress) != fObject)) {
         // The semantics of fAddress and fObject are violated.
         // Assume the user changed the pointer on us.
         // Note: The cast is here because we want to be able to
         //       be called from the constant get functions.

         // FIXME: Disable the check/warning TTree until we add a missing interface.
         if (TestBit(kDeleteObject)) {
            // This should never happen!
            Error("ValidateAddress", "We owned an object whose address changed!  our ptr: %p  new ptr: %p", fObject, *((char**) fAddress));
            const_cast<TBranchElement*>(this)->ResetBit(kDeleteObject);
         }
         const_cast<TBranchElement*>(this)->SetAddress(fAddress);
      }
   }
}

#endif // ROOT_TBranchElement
