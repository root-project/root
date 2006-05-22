// @(#)root/oracle:$Name:  $:$Id: TOracleStatement.h,v 1.1 2006/04/12 20:53:45 rdm Exp $
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
using namespace oracle::occi;
#else
class Connection;
class Statement;
class ResultSet;
class MetaData;
#endif

class TOracleStatement : public TSQLStatement {

protected:

   struct TBufferRec {
      char* strbuf;
      char* namebuf;
   };

   Connection            *fConn;        // connection to Oracle
   Statement             *fStmt;        // executed statement
   ResultSet             *fResult;      // query result (rows)
   std::vector<MetaData> *fFieldInfo;   // info for each field in the row
   TBufferRec            *fBuffer;       // buffer of values and field names
   Int_t                  fBufferSize;   // size of fBuffer
   Int_t                  fNumIterations;  // size of internal statement buffer
   Int_t                  fIterCounter; //counts nextiteration calls and process iterations, if required
   Int_t                  fWorkingMode; // 1 - settingpars, 2 - getting results

   Bool_t      IsParSettMode() const { return fWorkingMode==1; }
   Bool_t      IsResultSet() const { return (fWorkingMode==2) && (fResult!=0); }

   void        SetBufferSize(Int_t size);
   void        CloseBuffer();
   
public:
   TOracleStatement(Connection* conn, Statement* stmt, Int_t niter);
   virtual ~TOracleStatement();

   virtual void        Close(Option_t * = "");

   virtual Int_t       GetBufferLength() const { return fNumIterations; }
   virtual Int_t       GetNumParameters();

   virtual Bool_t      SetInt(Int_t npar, Int_t value);
   virtual Bool_t      SetUInt(Int_t npar, UInt_t value);
   virtual Bool_t      SetLong(Int_t npar, Long_t value);
   virtual Bool_t      SetLong64(Int_t npar, Long64_t value);
   virtual Bool_t      SetULong64(Int_t npar, ULong64_t value);
   virtual Bool_t      SetDouble(Int_t npar, Double_t value);
   virtual Bool_t      SetString(Int_t npar, const char* value, Int_t maxsize = 256);

   virtual Bool_t      NextIteration();

   virtual Bool_t      Process();
   virtual Int_t       GetNumAffectedRows();

   virtual Bool_t      StoreResult();
   virtual Int_t       GetNumFields();
   virtual const char *GetFieldName(Int_t nfield);
   virtual Bool_t      NextResultRow();

   virtual Int_t       GetInt(Int_t npar);
   virtual UInt_t      GetUInt(Int_t npar);
   virtual Long_t      GetLong(Int_t npar);
   virtual Long64_t    GetLong64(Int_t npar);
   virtual ULong64_t   GetULong64(Int_t npar);
   virtual Double_t    GetDouble(Int_t npar);
   virtual const char *GetString(Int_t npar);

   ClassDef(TOracleStatement, 0); // Statement class for Oracle
};

#endif
