// @(#)root/tree:$Name:  $:$Id: TBranchElement.h,v 1.53 2006/12/08 17:33:28 pcanal Exp $
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

class TFolder;
class TStreamerInfo;
class TVirtualCollectionProxy;

class TBranchElement : public TBranch {

// Friends
   friend class TTreeCloner;

// Types
protected:
   enum {
      kBranchFolder = BIT(14),
      kDeleteObject = BIT(16)               //  We are the owner of fObject.
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
   Bool_t                   fInit;          //! Initialization flag for branch assignment
   Bool_t                   fInitOffsets;   //! Initialization flag to not endlessly recalculate offsets
   TClassRef                fCurrentClass;  //! Reference to current (transient) class definition
   TClassRef                fParentClass;   //! Reference to class definition in fParentName
   TClassRef                fBranchClass;   //! Reference to class definition in fClassName
   Int_t                   *fBranchOffset;  //! Sub-Branch offsets with respect to current transient class

// Not implemented
private:
   TBranchElement(const TBranchElement&);            // not implemented
   TBranchElement& operator=(const TBranchElement&); // not implemented

// Implementation use only functions.
protected:
   void                     BuildTitle(const char* name);
   virtual void             InitializeOffsets();
   virtual void             InitInfo();
   Bool_t                   IsMissingCollection() const;
   TClass                  *GetCurrentClass(); // Class referenced by transient description
   TClass                  *GetParentClass(); // Class referenced by fParentName
   TVirtualCollectionProxy *GetCollectionProxy();
   void                     ReleaseObject();
   void                     SetBranchCount(TBranchElement* bre);
   void                     SetBranchCount2(TBranchElement* bre) { fBranchCount2 = bre; }
   Int_t                    Unroll(const char* name, TClass* cltop, TClass* cl, char* ptr, Int_t basketsize, Int_t splitlevel, Int_t btype);
   void                     ValidateAddress() const;

// Public Interface.
public:
   TBranchElement();
   TBranchElement(const char* name, TStreamerInfo* sinfo, Int_t id, char* pointer, Int_t basketsize = 32000, Int_t splitlevel = 0, Int_t btype = 0);
   TBranchElement(const char* name, TClonesArray* clones, Int_t basketsize = 32000, Int_t splitlevel = 0, Int_t compress = -1);
   TBranchElement(const char* name, TVirtualCollectionProxy* cont, Int_t basketsize = 32000, Int_t splitlevel = 0, Int_t compress = -1);

   virtual                  ~TBranchElement();

   virtual void             Browse(TBrowser* b);
   virtual Int_t            Fill();
   virtual void             FillLeaves(TBuffer& b);
   virtual char            *GetAddress() const;
           TBranchElement  *GetBranchCount() const { return fBranchCount; }
           TBranchElement  *GetBranchCount2() const { return fBranchCount2; }
           Int_t           *GetBranchOffset() const { return fBranchOffset; }
           UInt_t           GetCheckSum() { return fCheckSum; }
   virtual const char      *GetClassName() const { return fClassName.Data(); }
   virtual const char      *GetClonesName() const { return fClonesName.Data(); }
   virtual Int_t            GetEntry(Long64_t entry = 0, Int_t getall = 0);
           const char      *GetIconName() const;
           Int_t            GetID() const { return fID; }
           TStreamerInfo   *GetInfo() const;
           char            *GetObject() const;
   virtual const char      *GetParentName() const { return fParentName.Data(); }
   virtual Int_t            GetMaximum() const;
           Int_t            GetNdata() const { return fNdata; }
           Int_t            GetType() const { return fType; }
           Int_t            GetStreamerType() const { return fStreamerType; }
   virtual const char      *GetTypeName() const;
           Double_t         GetValue(Int_t i, Int_t len, Bool_t subarr = kFALSE) const;
   virtual void            *GetValuePointer() const;
           Bool_t           IsBranchFolder() const { return TestBit(kBranchFolder); }
           Bool_t           IsFolder() const;
   virtual Bool_t           IsObjectOwner() const { return TestBit(kDeleteObject); }
   virtual Bool_t           Notify() { if (fAddress) { SetAddress(0); } return 1; }
   virtual void             Print(Option_t* option = "") const;
           void             PrintValue(Int_t i) const;
   virtual void             ReadLeaves(TBuffer& b);
   virtual void             Reset(Option_t* option = "");
   virtual void             ResetAddress();
   virtual void             ResetDeleteObject();
   virtual void             SetAddress(void* addobj);
   virtual void             SetBasketSize(Int_t buffsize);
   virtual void             SetBranchFolder() { SetBit(kBranchFolder); }
   virtual void             SetClassName(const char* name) { fClassName = name; }
   inline  void             SetParentClass(TClass* clparent);
   virtual void             SetParentName(const char* name) { fParentName = name; }
   virtual void             SetupAddresses();
   virtual void             SetType(Int_t btype) { fType = btype; }

   ClassDef(TBranchElement,8)  // Branch in case of an object
};

inline void TBranchElement::SetParentClass(TClass* clparent)
{
   fParentClass = clparent;
   fParentName = clparent ? clparent->GetName() : "";
}

#endif // ROOT_TBranchElement
