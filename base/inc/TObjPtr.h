// @(#)root/base:$Name:  $:$Id: TObjPtr.h,v 1.2 2000/12/13 15:13:45 brun Exp $
// Author: Fons Rademakers   04/05/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TObjPtr
#define ROOT_TObjPtr


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjPtr                                                              //
//                                                                      //
// Collectable generic pointer class. This is a TObject containing a    //
// void *.                                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif


class TObjPtr : public TObject {

private:
   const void  *fPtr;       //Wrapped pointer

public:
   TObjPtr(const void *p = 0) : fPtr(p) { }
   TObjPtr(const TObjPtr &p) : TObject(p),  fPtr(p.fPtr) { }
   ~TObjPtr() { }
   Int_t     Compare(const TObject *obj) const;
   ULong_t   Hash() const { return (ULong_t) fPtr >> 2; }
   Bool_t    IsSortable() const { return kTRUE; }
   Bool_t    IsEqual(const TObject *obj) const { return fPtr == obj; }
   void     *Ptr() const { return (void *)fPtr; }

   //ClassDef(TObjPtr,1)  //Collectable generic pointer class
};

#endif
