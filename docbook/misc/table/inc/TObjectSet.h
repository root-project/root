// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   25/12/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TObjectSet
#define ROOT_TObjectSet

#include "TDataSet.h"

//////////////////////////////////////////////////////////////////////////////////////
//                                                                                  //
//  TObjectSet  - is a container TDataSet                                           //
//                  This means this object has an extra pointer to an embedded      //
//                  TObject.                                                        //
//  Terminology:    This TObjectSet may be an OWNER of the embeded TObject          //
//                  If the container is the owner it can delete the embeded object  //
//                  otherwsie it leaves that object "as is"                         //
//                                                                                  //
//////////////////////////////////////////////////////////////////////////////////////

class TObjectSet : public TDataSet {
protected:
   enum EOwnerBits { kIsOwner         = BIT(23) };
   TObject *fObj;                              // TObject to be inserted

public:
   TObjectSet(const Char_t *name, TObject *obj=0,Bool_t makeOwner=kTRUE);
   TObjectSet(TObject *obj=0,Bool_t makeOwner=kTRUE);
   virtual ~TObjectSet();
   virtual TObject *AddObject(TObject *obj,Bool_t makeOwner=kTRUE);
   virtual void     Browse(TBrowser *b);
   virtual void     Delete(Option_t *opt="");
   virtual Bool_t   DoOwner(Bool_t done=kTRUE);
   virtual Long_t   HasData() const;
   virtual TObject *GetObject() const; 
   virtual TDataSet *Instance() const;
   virtual Bool_t   IsOwner() const;
   virtual void     SetObject(TObject *obj);
   virtual TObject *SetObject(TObject *obj,Bool_t makeOwner);

   static TObjectSet *instance();

   ClassDef(TObjectSet,1) // TDataSet wrapper for TObject class objects
};

inline TObjectSet *TObjectSet::instance()
{ return new TObjectSet();}

inline Long_t   TObjectSet::HasData()   const {return fObj ? 1 : 0;}
inline TObject *TObjectSet::GetObject() const {return fObj;}
inline Bool_t   TObjectSet::IsOwner()   const {return TestBit(kIsOwner);}

inline void     TObjectSet::SetObject(TObject *obj) { SetObject(obj,kTRUE);}

#endif

