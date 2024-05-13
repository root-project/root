// @(#)root/net:$Id$
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

#include "TObject.h"
#include "TString.h"
#include "TDatime.h"
#include "TTimeStamp.h"
#include <vector>

class TSQLStatement : public TObject {

protected:
   TSQLStatement(Bool_t errout = kTRUE) { fErrorOut = errout; }

   Int_t     fErrorCode{0};      // error code of last operation
   TString   fErrorMsg;          // error message of last operation
   Bool_t    fErrorOut{kFALSE};  // enable error output

   void                ClearError();
   void                SetError(Int_t code, const char* msg, const char *method = nullptr);

public:
   virtual ~TSQLStatement() = default;

   virtual Int_t       GetBufferLength() const = 0;
   virtual Int_t       GetNumParameters() = 0;

   virtual Bool_t      NextIteration() = 0;

   virtual void        Close(Option_t * = "") {}

   virtual Bool_t      SetNull(Int_t) { return kFALSE; }
   virtual Bool_t      SetInt(Int_t, Int_t) { return kFALSE; }
   virtual Bool_t      SetUInt(Int_t, UInt_t) { return kFALSE; }
   virtual Bool_t      SetLong(Int_t, Long_t) { return kFALSE; }
   virtual Bool_t      SetLong64(Int_t, Long64_t) { return kFALSE; }
   virtual Bool_t      SetULong64(Int_t, ULong64_t) { return kFALSE; }
   virtual Bool_t      SetDouble(Int_t, Double_t) { return kFALSE; }
   virtual Bool_t      SetString(Int_t, const char*, Int_t = 256) { return kFALSE; }
   virtual Bool_t      SetDate(Int_t, Int_t, Int_t, Int_t) { return kFALSE; }
           Bool_t      SetDate(Int_t, const TDatime&);
   virtual Bool_t      SetTime(Int_t, Int_t, Int_t, Int_t) { return kFALSE; }
           Bool_t      SetTime(Int_t, const TDatime&);
   virtual Bool_t      SetDatime(Int_t, Int_t, Int_t, Int_t, Int_t, Int_t, Int_t) { return kFALSE; }
           Bool_t      SetDatime(Int_t, const TDatime&);
   virtual Bool_t      SetTimestamp(Int_t, Int_t, Int_t, Int_t, Int_t, Int_t, Int_t, Int_t = 0);
   virtual Bool_t      SetTimestamp(Int_t, const TTimeStamp&);
           Bool_t      SetTimestamp(Int_t, const TDatime&);
   virtual void        SetTimeFormating(const char*) {}
   virtual Bool_t      SetBinary(Int_t, void*, Long_t, Long_t = 0x1000) { return kFALSE; }
   virtual Bool_t      SetLargeObject(Int_t col, void* mem, Long_t size, Long_t maxsize = 0x1000) { return SetBinary(col, mem, size, maxsize); }

   virtual Bool_t      SetVInt(Int_t, const std::vector<Int_t>, const char*, const char*) { return kFALSE; }
   virtual Bool_t      SetVUInt(Int_t, const std::vector<UInt_t>, const char*, const char*) { return kFALSE; }
   virtual Bool_t      SetVLong(Int_t, const std::vector<Long_t>, const char*, const char*) { return kFALSE; }
   virtual Bool_t      SetVLong64(Int_t, const std::vector<Long64_t>, const char*, const char*) { return kFALSE; }
   virtual Bool_t      SetVULong64(Int_t, const std::vector<ULong64_t>, const char*, const char*) { return kFALSE; }
   virtual Bool_t      SetVDouble(Int_t, const std::vector<Double_t>, const char*, const char*) { return kFALSE; }

   virtual Bool_t      Process() = 0;
   virtual Int_t       GetNumAffectedRows() { return 0; }

   virtual Bool_t      StoreResult() = 0;
   virtual Int_t       GetNumFields() = 0;
   virtual const char *GetFieldName(Int_t) = 0;
   virtual Bool_t      SetMaxFieldSize(Int_t, Long_t) { return kFALSE; }
   virtual Bool_t      NextResultRow() = 0;

   virtual Bool_t      IsNull(Int_t) { return kTRUE; }
   virtual Int_t       GetInt(Int_t) { return 0; }
   virtual UInt_t      GetUInt(Int_t) { return 0; }
   virtual Long_t      GetLong(Int_t) { return 0; }
   virtual Long64_t    GetLong64(Int_t) { return 0; }
   virtual ULong64_t   GetULong64(Int_t) { return 0; }
   virtual Double_t    GetDouble(Int_t) { return 0.; }
   virtual const char *GetString(Int_t) { return nullptr; }
   virtual Bool_t      GetBinary(Int_t, void* &, Long_t&) { return kFALSE; }
   virtual Bool_t      GetLargeObject(Int_t col, void* &mem, Long_t& size) { return GetBinary(col, mem, size); }

   virtual Bool_t      GetDate(Int_t, Int_t&, Int_t&, Int_t&) { return kFALSE; }
   virtual Bool_t      GetTime(Int_t, Int_t&, Int_t&, Int_t&) { return kFALSE; }
   virtual Bool_t      GetDatime(Int_t, Int_t&, Int_t&, Int_t&, Int_t&, Int_t&, Int_t&) { return kFALSE; }
           TDatime     GetDatime(Int_t);
           Int_t       GetYear(Int_t);
           Int_t       GetMonth(Int_t);
           Int_t       GetDay(Int_t);
           Int_t       GetHour(Int_t);
           Int_t       GetMinute(Int_t);
           Int_t       GetSecond(Int_t);
           Int_t       GetSecondsFraction(Int_t);
   virtual Bool_t      GetTimestamp(Int_t, Int_t&, Int_t&, Int_t&, Int_t&, Int_t&, Int_t&, Int_t&);
   virtual Bool_t      GetTimestamp(Int_t, TTimeStamp&);
           TDatime     GetTimestamp(Int_t);
   virtual Bool_t      GetVInt(Int_t, std::vector<Int_t>&) { return kFALSE; }
   virtual Bool_t      GetVUInt(Int_t, std::vector<UInt_t>&) { return kFALSE; }
   virtual Bool_t      GetVLong(Int_t, std::vector<Long_t>&) { return kFALSE; }
   virtual Bool_t      GetVLong64(Int_t, std::vector<Long64_t>&) { return kFALSE; }
   virtual Bool_t      GetVULong64(Int_t, std::vector<ULong64_t>&) { return kFALSE; }
   virtual Bool_t      GetVDouble(Int_t, std::vector<Double_t>&) { return kFALSE; }

   virtual Bool_t      IsError() const { return GetErrorCode()!=0; }
   virtual Int_t       GetErrorCode() const;
   virtual const char* GetErrorMsg() const;
   virtual void        EnableErrorOutput(Bool_t on = kTRUE) { fErrorOut = on; }

   ClassDefOverride(TSQLStatement, 0) //SQL statement
};

#endif
