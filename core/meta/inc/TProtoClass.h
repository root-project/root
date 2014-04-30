// @(#)root/meta:$Id$
// Author: Axel Naumann 2014-04-28

/*************************************************************************
 * Copyright (C) 1995-2014, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TProtoClass
#define ROOT_TProtoClass

#ifndef ROOT_TList
#include "TList.h"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProtoClass                                                          //
//                                                                      //
// Stores enough information to create a TClass from a dictionary       //
// without interpreter information.                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TProtoClass: public TNamed {
public:
   TList   *fBase;     // List of base classes
   TList   *fData;     // List of data members
   TList   *fRealData; // TRealData; must be after fData for I/O!
   Int_t    fSizeof;   // Size of the class
   Int_t    fCanSplit; // Whether this class can be split
   Long_t   fProperty; // Class properties, see EProperties

   TProtoClass(const TProtoClass&) = delete;
   TProtoClass& operator=(const TProtoClass&) = delete;

   TProtoClass():
      fBase(0), fData(0), fRealData(0), fSizeof(0), fCanSplit(0), fProperty(0)
   {}

   virtual ~TProtoClass(); // implemented in TClass.cxx to pin vtable

   void Delete(Option_t* opt = "") {
      // Delete the containers that are usually owned by their TClass.
      if (fRealData) fRealData->Delete(opt);
      if (fBase) fBase->Delete(opt);
      if (fData) fData->Delete(opt);
   }

   ClassDef(TProtoClass,2); //Persistent TClass
};

#endif
