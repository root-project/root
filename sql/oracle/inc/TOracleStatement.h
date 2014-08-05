// @(#)root/oracle:$Id$
// Author: Sergey Linev   6/02/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TOracleStatement
#define ROOT_TOracleStatement

#ifndef ROOT_TSQLStatement
#include "TSQLStatement.h"
#endif

#if !defined(__CINT__)
#include <occi.h>
#ifdef CONST
#undef CONST
#endif
#else
namespace oracle { namespace occi {
class Environment;
class Connection;
class Statement;
class ResultSet;
class MetaData;
   }}
#endif

class TOracleStatement : public TSQLStatement {

protected:

   struct TBufferRec {
      char* strbuf;
      Long_t strbufsize;
      char* namebuf;
   };

   oracle::occi::Environment *fEnv;         // environment
   oracle::occi::Connection  *fConn;        // connection to Oracle
   oracle::occi::Statement   *fStmt;        // executed statement
   oracle::occi::ResultSet   *fResult;      // query result (rows)
   std::vector<oracle::occi::MetaData> *fFieldInfo;   // info for each field in the row
   TBufferRec            *fBuffer;       // buffer of values and field names
   Int_t                  fBufferSize;   // size of fBuffer
   Int_t                  fNumIterations;  // size of internal statement buffer
   Int_t                  fIterCounter; //counts nextiteration calls and process iterations, if required
   Int_t                  fWorkingMode; // 1 - settingpars, 2 - getting results
   TString                fTimeFmt;     // format for date to string conversion, default "MM/DD/YYYY, HH24:MI:SS"

   Bool_t      IsParSettMode() const { return fWorkingMode==1; }
   Bool_t      IsResultSet() const { return (fWorkingMode==2) && (fResult!=0); }

   void        SetBufferSize(Int_t size);
   void        CloseBuffer();

public:
   TOracleStatement(oracle::occi::Environment* env,
                    oracle::occi::Connection* conn,
                    oracle::occi::Statement* stmt,
                    Int_t niter, Bool_t errout = kTRUE);
   virtual ~TOracleStatement();

   virtual void        Close(Option_t * = "");

   virtual Int_t       GetBufferLength() const { return fNumIterations; }
   virtual Int_t       GetNumParameters();

   virtual Bool_t      SetNull(Int_t npar);
   virtual Bool_t      SetInt(Int_t npar, Int_t value);
   virtual Bool_t      SetUInt(Int_t npar, UInt_t value);
   virtual Bool_t      SetLong(Int_t npar, Long_t value);
   virtual Bool_t      SetLong64(Int_t npar, Long64_t value);
   virtual Bool_t      SetULong64(Int_t npar, ULong64_t value);
   virtual Bool_t      SetDouble(Int_t npar, Double_t value);
   virtual Bool_t      SetString(Int_t npar, const char* value, Int_t maxsize = 256);
   virtual Bool_t      SetBinary(Int_t npar, void* mem, Long_t size, Long_t maxsize = 0x1000);
   virtual Bool_t      SetDate(Int_t npar, Int_t year, Int_t month, Int_t day);
   virtual Bool_t      SetTime(Int_t npar, Int_t hour, Int_t min, Int_t sec);
   virtual Bool_t      SetDatime(Int_t npar, Int_t year, Int_t month, Int_t day, Int_t hour, Int_t min, Int_t sec);
   virtual Bool_t      SetTimestamp(Int_t npar, Int_t year, Int_t month, Int_t day, Int_t hour, Int_t min, Int_t sec, Int_t frac = 0);
   virtual void        SetTimeFormating(const char* fmt) { fTimeFmt = fmt; }
   virtual Bool_t      SetVInt(Int_t npar, const std::vector<Int_t> value, const char* schemaName, const char* typeName);
   virtual Bool_t      SetVUInt(Int_t npar, const std::vector<UInt_t> value, const char* schemaName, const char* typeName);
   virtual Bool_t      SetVLong(Int_t npar, const std::vector<Long_t> value, const char* schemaName, const char* typeName);
   virtual Bool_t      SetVLong64(Int_t npar, const std::vector<Long64_t> value, const char* schemaName, const char* typeName);
   virtual Bool_t      SetVULong64(Int_t npar, const std::vector<ULong64_t> value, const char* schemaName, const char* typeName);
   virtual Bool_t      SetVDouble(Int_t npar, const std::vector<Double_t> value, const char* schemaName, const char* typeName);

   virtual Bool_t      NextIteration();

   virtual Bool_t      Process();
   virtual Int_t       GetNumAffectedRows();

   virtual Bool_t      StoreResult();
   virtual Int_t       GetNumFields();
   virtual const char *GetFieldName(Int_t nfield);
   virtual Bool_t      SetMaxFieldSize(Int_t nfield, Long_t maxsize);
   virtual Bool_t      NextResultRow();

   virtual Bool_t      IsNull(Int_t);
   virtual Int_t       GetInt(Int_t npar);
   virtual UInt_t      GetUInt(Int_t npar);
   virtual Long_t      GetLong(Int_t npar);
   virtual Long64_t    GetLong64(Int_t npar);
   virtual ULong64_t   GetULong64(Int_t npar);
   virtual Double_t    GetDouble(Int_t npar);
   virtual const char *GetString(Int_t npar);
   virtual Bool_t      GetBinary(Int_t npar, void* &mem, Long_t& size);
   virtual Bool_t      GetDate(Int_t npar, Int_t& year, Int_t& month, Int_t& day);
   virtual Bool_t      GetTime(Int_t npar, Int_t& hour, Int_t& min, Int_t& sec);
   virtual Bool_t      GetDatime(Int_t npar, Int_t& year, Int_t& month, Int_t& day, Int_t& hour, Int_t& min, Int_t& sec);
   virtual Bool_t      GetTimestamp(Int_t npar, Int_t& year, Int_t& month, Int_t& day, Int_t& hour, Int_t& min, Int_t& sec, Int_t& frac);
   virtual Bool_t      GetVInt(Int_t npar, std::vector<Int_t> &value);
   virtual Bool_t      GetVUInt(Int_t npar, std::vector<UInt_t> &value);
   virtual Bool_t      GetVLong(Int_t npar, std::vector<Long_t> &value);
   virtual Bool_t      GetVLong64(Int_t npar, std::vector<Long64_t> &value);
   virtual Bool_t      GetVULong64(Int_t npar, std::vector<ULong64_t> &value);
   virtual Bool_t      GetVDouble(Int_t npar, std::vector<Double_t> &value);

   ClassDef(TOracleStatement, 0); // SQL statement class for Oracle
};

#endif
