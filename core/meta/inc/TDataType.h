// @(#)root/meta:$Id$
// Author: Rene Brun   04/02/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDataType
#define ROOT_TDataType


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDataType                                                            //
//                                                                      //
// Basic data type descriptor (datatype information is obtained from    //
// CINT).                                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDictionary
#include "TDictionary.h"
#endif


enum EDataType {
   kChar_t   = 1,  kUChar_t  = 11, kShort_t    = 2,  kUShort_t = 12,
   kInt_t    = 3,  kUInt_t   = 13, kLong_t     = 4,  kULong_t  = 14,
   kFloat_t  = 5,  kDouble_t =  8, kDouble32_t = 9,  kchar     = 10,
   kBool_t   = 18, kLong64_t = 16, kULong64_t  = 17, kOther_t  = -1,
   kNoType_t = 0,  kFloat16_t= 19,
   kCounter  =  6, kCharStar = 7,  kBits     = 15 /* for compatibility with TStreamerInfo */,
   kVoid_t   = 20,

   kDataTypeAliasUnsigned_t = 21,
   kDataTypeAliasSignedChar_t = 22,
   // could add "long int" etc
   kNumDataTypes
};


class TDataType : public TDictionary {

private:
   TypedefInfo_t    *fInfo;     //!pointer to CINT typedef info
   Int_t             fSize;     //size of type
   EDataType         fType;     //type id
   Long_t            fProperty; //The property information for the (potential) underlying class
   TString           fTrueName; //Qualified name of the (potential) underlying class, e.g. "MyClass*const*"
   Int_t             fTypeNameIdx; //Start of class name part of the (potential) underlying class in fTrueName
   Int_t             fTypeNameLen; //Strlen of class name part of the (potential) underlying class in fTrueName
   static TDataType* fgBuiltins[kNumDataTypes]; //Array of builtins

   void CheckInfo();
   void SetType(const char *name);

protected:
   TDataType(const TDataType&);
   TDataType& operator=(const TDataType&);

public:
   TDataType(TypedefInfo_t *info = 0);
   TDataType(const char *typenam);
   virtual       ~TDataType();
   Int_t          Size() const;
   Int_t          GetType() const { return (Int_t)fType; }
   TString        GetTypeName();
   const char    *GetFullTypeName() const;
   const char    *AsString(void *buf) const;
   Long_t         Property() const;

   static const char *GetTypeName(EDataType type);
   static TDataType  *GetDataType(EDataType type);
   static EDataType GetType(const type_info &typeinfo);
   static void AddBuiltins(TCollection* types);

   ClassDef(TDataType,2)  //Basic data type descriptor
};

#endif
