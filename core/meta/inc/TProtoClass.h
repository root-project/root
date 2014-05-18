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

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TClass;
class TList;
class TRealData;

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
   class TProtoRealData: public TNamed {
      Long_t fOffset; // data member offset
      enum {
         kIsObject = BIT(15) // member is an object
      };
   public:
      TProtoRealData() {}
      TProtoRealData(const TRealData* rd);
      virtual ~TProtoRealData();
      TRealData* CreateRealData(TClass* currentClass) const;
      ClassDef(TProtoRealData, 2);//Persistent version of TRealData
   };

private:
   TList   *fBase;     // List of base classes
   TList   *fData;     // List of data members
   TList   *fPRealData;// List of TProtoRealData
   Int_t    fSizeof;   // Size of the class
   Int_t    fCanSplit; // Whether this class can be split
   Int_t    fStreamerType; // Which streaming method to use
   Long_t   fProperty; // Class properties, see EProperties
   Long_t   fClassProperty; // Class C++ properties, see EClassProperties
   Long_t   fOffsetStreamer; // Offset to streamer function

   TProtoClass(const TProtoClass&) = delete;
   TProtoClass& operator=(const TProtoClass&) = delete;

public:
   TProtoClass():
      fBase(0), fData(0), fPRealData(0), fSizeof(0), fCanSplit(0),
      fStreamerType(0), fProperty(0), fClassProperty(0),
      fOffsetStreamer(0)
   {}

   TProtoClass(TClass* cl);
   virtual ~TProtoClass();

   Bool_t FillTClass(TClass* pcl);
   void Delete(Option_t* opt = "");

   ClassDef(TProtoClass,2); //Persistent TClass
};

#endif
