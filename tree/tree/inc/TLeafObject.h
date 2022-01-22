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
   Bool_t       fVirtual;        ///<  Support for polymorphism, when set classname is written with object.

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
   virtual ~TLeafObject();

   virtual Bool_t  CanGenerateOffsetArray() { return false; }
   virtual void    FillBasket(TBuffer &b);
   virtual Int_t  *GenerateOffsetArrayBase(Int_t /*base*/, Int_t /*events*/) { return nullptr; }
   TClass         *GetClass() const {return fClass;}
   TMethodCall    *GetMethodCall(const char *name);
   TObject        *GetObject() const {return (TObject*)(*fObjAddress);}
   const char     *GetTypeName() const ;
   virtual void   *GetValuePointer() const {return fObjAddress;}
   Bool_t          IsOnTerminalBranch() const;
   Bool_t          IsVirtual() const {return fVirtual;}
   virtual Bool_t  Notify();
   virtual void    PrintValue(Int_t i=0) const;
   virtual void    ReadBasket(TBuffer &b);
   virtual void    SetAddress(void *add=0);
   virtual void    SetVirtual(Bool_t virt=kTRUE) {fVirtual=virt;}

   ClassDef(TLeafObject,4);  //A TLeaf for a general object derived from TObject.
};

#endif
