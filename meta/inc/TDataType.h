// @(#)root/meta:$Name:  $:$Id: TDataType.h,v 1.1.1.1 2000/05/16 17:00:43 rdm Exp $
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
#ifndef ROOT_TString
#include "TString.h"
#endif


enum EDataType {
   kChar_t  = 1, kUChar_t  = 11, kShort_t = 2,  kUShort_t = 12,
   kInt_t   = 3, kUInt_t   = 13, kLong_t  = 4,  kULong_t  = 14,
   kFloat_t = 5, kDouble_t = 8,  kchar  = 10, kOther_t  = -1
};

class G__TypedefInfo;


class TDataType : public TDictionary {

private:
   G__TypedefInfo   *fInfo;     //pointer to CINT typedef info
   TString           fName;     //name of basic type
   Int_t             fSize;     //size of basic type (in case fInfo==0)
   EDataType         fType;     //type id

   void SetType(const char *name);

public:
   TDataType(G__TypedefInfo *info = 0);
   TDataType(const char *typenam);
   virtual       ~TDataType();
   Int_t          Size() const;
   Int_t          GetType() const { return (Int_t)fType; }
   const char    *GetTypeName() const;
   const char    *GetFullTypeName() const;
   const char    *GetName() const;
   const char    *GetTitle() const;
   Int_t          Compare(const TObject *obj) const;
   ULong_t        Hash() const;
   const char    *AsString(void *buf) const;
   Long_t         Property() const;

   ClassDef(TDataType,0)  //Basic data type descriptor
};

#endif

