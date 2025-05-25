// @(#)root/tree:$Id$
// Author: Rene Brun   27/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLeafObject
#define ROOT_TLeafObject


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLeafObject                                                          //
//                                                                      //
// A TLeaf for a general object derived from TObject.                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TLeaf.h"
#include "TClassRef.h"

class TClass;
class TMethodCall;

class TLeafObject : public TLeaf {

protected:
   TClassRef    fClass;          ///<! pointer to class
   void       **fObjAddress;     ///<! Address of Pointer to object
   bool         fVirtual;        ///<  Support for polymorphism, when set classname is written with object.

public:
   enum EStatusBits {
      kWarn = BIT(14)
   };

   /// In version of ROOT older then v6.12, kWarn was set to BIT(12)
   /// which overlaps with TBranch::kBranchObject.  Since it stored
   /// in ROOT files as part of the TBranchObject and that we want
   /// to reset in TBranchObject::Streamer, we need to keep track
   /// of the old value.
   enum EStatusBitsOldValues {
      kOldWarn = BIT(12)
   };

   TLeafObject();
   TLeafObject(TBranch *parent, const char *name, const char *type);
   ~TLeafObject() override;

   bool            CanGenerateOffsetArray() override { return false; }
   void            FillBasket(TBuffer &b) override;
   virtual Int_t  *GenerateOffsetArrayBase(Int_t /*base*/, Int_t /*events*/) { return nullptr; }
   TClass         *GetClass() const {return fClass;}
   TMethodCall    *GetMethodCall(const char *name);
   TObject        *GetObject() const {return fObjAddress ? (TObject*)(*fObjAddress) : nullptr;}
   const char     *GetTypeName() const override;
   void           *GetValuePointer() const override {return fObjAddress;}
   bool            IsOnTerminalBranch() const override;
   bool            IsVirtual() const {return fVirtual;}
   bool            Notify() override;
   void            PrintValue(Int_t i=0) const override;
   void            ReadBasket(TBuffer &b) override;
   void            SetAddress(void *add=nullptr) override;
   virtual void    SetVirtual(bool virt=true) {fVirtual=virt;}

   ClassDefOverride(TLeafObject,4);  //A TLeaf for a general object derived from TObject.
};

#endif
