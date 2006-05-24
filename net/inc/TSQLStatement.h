// @(#)root/net:$Name:  $:$Id: TSQLStatement.h,v 1.2 2006/05/22 08:55:30 brun Exp $
// Author: Sergey Linev   6/02/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSQLStatement
#define ROOT_TSQLStatement

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TSQLStatement : public TObject {

protected:
   TSQLStatement() : TObject() { ClearError(); }

   Int_t     fErrorCode;  // error code of last operation
   TString   fErrorMsg;   // error message of last operation

   void                ClearError();
   void                SetError(Int_t code, const char* msg, const char* method = 0);

public:
   virtual ~TSQLStatement() {}

   virtual Int_t       GetBufferLength() const = 0;
   virtual Int_t       GetNumParameters() = 0;

   virtual Bool_t      NextIteration() = 0;

   virtual Bool_t      SetInt(Int_t, Int_t) { return kFALSE; }
   virtual Bool_t      SetUInt(Int_t, UInt_t) { return kFALSE; }
   virtual Bool_t      SetLong(Int_t, Long_t) { return kFALSE; }
   virtual Bool_t      SetLong64(Int_t, Long64_t) { return kFALSE; }
   virtual Bool_t      SetULong64(Int_t, ULong64_t) { return kFALSE; }
   virtual Bool_t      SetDouble(Int_t, Double_t) { return kFALSE; }
   virtual Bool_t      SetString(Int_t, const char*, Int_t = 256) { return kFALSE; }

   virtual Bool_t      Process() = 0;
   virtual Int_t       GetNumAffectedRows() { return 0; }

   virtual Bool_t      StoreResult() = 0;
   virtual Int_t       GetNumFields() = 0;
   virtual const char *GetFieldName(Int_t) = 0;
   virtual Bool_t      NextResultRow() = 0;

   virtual Int_t       GetInt(Int_t) { return 0; }
   virtual UInt_t      GetUInt(Int_t) { return 0; }
   virtual Long_t      GetLong(Int_t) { return 0; }
   virtual Long64_t    GetLong64(Int_t) { return 0; }
   virtual ULong64_t   GetULong64(Int_t) { return 0; }
   virtual Double_t    GetDouble(Int_t) { return 0.; }
   virtual const char *GetString(Int_t) { return 0; }

   virtual Bool_t      IsError() const { return GetErrorCode()!=0; }
   virtual Int_t       GetErrorCode() const;
   virtual const char* GetErrorMsg() const;

   ClassDef(TSQLStatement, 0) //SQL statement
};

#endif
