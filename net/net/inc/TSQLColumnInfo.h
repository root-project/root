// @(#)root/net:$Id$
// Author: Sergey Linev   31/05/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSQLColumnInfo
#define ROOT_TSQLColumnInfo

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TSQLColumnInfo : public TNamed {

protected:
   // Database specific fields 
   TString  fTypeName;   //! sql type name, as reported by DB. Should be as much as close to declaration of column in CREATE TABLE query
   
   // Database independent fields
   Int_t    fSQLType;    //! datatype code (see TSQLServer::ESQLDataTypes constants), -1 if not defeined
   Int_t    fSize;       //! size of column in bytes, -1 if not defing
   Int_t    fLength;     //! datatype length definition, for instance VARCHAR(len) or FLOAT(len), -1 if not defined
   Int_t    fScale;      //! datatype scale factor, used for instance in NUMBER(len,scale) definition. -1 if not defined
   Int_t    fSigned;     //! if datatype signed or not, 0 - kFALSE, 1 - kTRUE, -1 - unknown
   Bool_t   fNullable;   //! identify if value can be NULL 

public:
   TSQLColumnInfo();
   TSQLColumnInfo(const char* columnname,
                  const char* sqltypename = "unknown",
                  Bool_t nullable = kFALSE,
                  Int_t sqltype = -1,
                  Int_t size = -1,
                  Int_t length = -1,
                  Int_t scale = -1,
                  Int_t sign = -1);
   virtual ~TSQLColumnInfo() {}
   
   const char* GetTypeName() const { return fTypeName.Data(); }
   Bool_t      IsNullable()  const { return fNullable; }
   Int_t       GetSQLType()  const { return fSQLType; }
   Int_t       GetSize()     const { return fSize; }
   Int_t       GetLength()   const { return fLength; }
   Int_t       GetScale()    const { return fScale; }
   Int_t       GetSigned()   const { return fSigned; }
   Bool_t      IsSigned()    const { return fSigned==1; }
   Bool_t      IsUnsigned()  const { return fSigned==0; }
   
   virtual void Print(Option_t* option = "") const;

   ClassDef(TSQLColumnInfo, 0) // Summury information about column from SQL table
};

#endif
