// @(#)root/tree:$Id$
// Author: Rene Brun   11/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TChainElement
#define ROOT_TChainElement


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TChainElement                                                        //
//                                                                      //
// Describes a component of a TChain.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TBranch;

class TChainElement : public TNamed {

   // TChainElement status bits
   enum EStatusBits {
      kHasBeenLookedUp = BIT(15)
   };

protected:
   Long64_t      fEntries;          //Number of entries in the tree of this chain element
   Int_t         fNPackets;         //Number of packets
   Int_t         fPacketSize;       //Number of events in one packet for parallel root
   Int_t         fStatus;           //branch status when used as a branch
   void         *fBaddress;         //!branch address when used as a branch
   TString       fBaddressClassName;//!Name of the class pointed to by fBaddress
   UInt_t        fBaddressType;     //!Type of the value pointed to by fBaddress
   Bool_t        fBaddressIsPtr;    //!True if the address is a pointer to an address
   char         *fPackets;          //!Packet descriptor string
   TBranch     **fBranchPtr;        //!Address of user branch pointer (to updated upon loading a file)

public:
   TChainElement();
   TChainElement(const char *title, const char *filename);
   virtual ~TChainElement();
   virtual void        CreatePackets();
   virtual void       *GetBaddress() const {return fBaddress;}
   virtual const char *GetBaddressClassName() const { return fBaddressClassName; }
   virtual Bool_t      GetBaddressIsPtr() const { return fBaddressIsPtr; }
   virtual UInt_t      GetBaddressType() const { return fBaddressType; }
   virtual TBranch   **GetBranchPtr() const { return fBranchPtr; }
   virtual Long64_t    GetEntries() const {return fEntries;}
   virtual char       *GetPackets() const {return fPackets;}
   virtual Int_t       GetPacketSize() const {return fPacketSize;}
   virtual Int_t       GetStatus() const {return fStatus;}
   virtual Bool_t      HasBeenLookedUp() { return TestBit(kHasBeenLookedUp); }
   virtual void        ls(Option_t *option="") const;
   virtual void        SetBaddress(void *add) {fBaddress = add;}
   virtual void        SetBaddressClassName(const char* clname) { fBaddressClassName = clname; }
   virtual void        SetBaddressIsPtr(Bool_t isptr) { fBaddressIsPtr = isptr; }
   virtual void        SetBaddressType(UInt_t type) { fBaddressType = type; }
   virtual void        SetBranchPtr(TBranch **ptr) { fBranchPtr = ptr; }
   virtual void        SetLookedUp(Bool_t y = kTRUE);
   virtual void        SetNumberEntries(Long64_t n) {fEntries=n;}
   virtual void        SetPacketSize(Int_t size = 100);
   virtual void        SetStatus(Int_t status) {fStatus = status;}

   ClassDef(TChainElement,2);  //A chain element
};

#endif

