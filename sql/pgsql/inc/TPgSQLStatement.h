// @(#)root/mysql:$Id$
// Author: Dennis Box (dbox@fnal.gov)  3/12/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPgSQLStatement
#define ROOT_TPgSQLStatement

#include "TSQLStatement.h"

#include <libpq-fe.h>
#include <pg_config.h> // to get PG_VERSION_NUM

#define pgsql_success(x) (((x) == PGRES_EMPTY_QUERY) \
                        || ((x) == PGRES_COMMAND_OK) \
                        || ((x) == PGRES_TUPLES_OK))

struct PgSQL_Stmt_t {
   PGconn   *fConn;
   PGresult *fRes;
};


class TPgSQLStatement : public TSQLStatement {

private:
   PgSQL_Stmt_t         *fStmt;          //! executed statement
   Int_t                 fNumBuffers;    //! number of statement parameters
   char                **fBind;          //! array of data for input
   char                **fFieldName;     //! array of column names
   Int_t                 fWorkingMode;   //! 1 - setting parameters, 2 - retrieving results
   Int_t                 fIterationCount;//! number of iteration
   int                  *fParamLengths;  //! length of column
   int                  *fParamFormats;  //! data type (OID)
   Int_t                 fNumResultRows;
   Int_t                 fNumResultCols;

   Bool_t      IsSetParsMode() const { return fWorkingMode==1; }
   Bool_t      IsResultSetMode() const { return fWorkingMode==2; }

   Bool_t      SetSQLParamType(Int_t npar, int sqltype, bool sig, int sqlsize = 0);

   long double ConvertToNumeric(Int_t npar);
   const char *ConvertToString(Int_t npar);

   void        FreeBuffers();
   void        SetBuffersNumber(Int_t n);

   void        ConvertTimeToUTC(const TString &PQvalue, Int_t& year, Int_t& month, Int_t& day, Int_t& hour, Int_t& min, Int_t& sec);

public:
   TPgSQLStatement(PgSQL_Stmt_t* stmt, Bool_t errout = kTRUE);
   virtual ~TPgSQLStatement();

   virtual void        Close(Option_t * = "");

   virtual Int_t       GetBufferLength() const { return 1; }
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
   virtual Bool_t      SetLargeObject(Int_t npar, void* mem, Long_t size, Long_t maxsize = 0x1000);
   virtual Bool_t      SetDate(Int_t npar, Int_t year, Int_t month, Int_t day);
   virtual Bool_t      SetTime(Int_t npar, Int_t hour, Int_t min, Int_t sec);
   virtual Bool_t      SetDatime(Int_t npar, Int_t year, Int_t month, Int_t day, Int_t hour, Int_t min, Int_t sec);
   virtual Bool_t      SetTimestamp(Int_t npar, Int_t year, Int_t month, Int_t day, Int_t hour, Int_t min, Int_t sec, Int_t frac = 0);

   virtual Bool_t      NextIteration();

   virtual Bool_t      Process();
   virtual Int_t       GetNumAffectedRows();

   virtual Bool_t      StoreResult();
   virtual Int_t       GetNumFields();
   virtual const char *GetFieldName(Int_t nfield);
   virtual Bool_t      NextResultRow();

   virtual Bool_t      IsNull(Int_t npar);
   virtual Int_t       GetInt(Int_t npar);
   virtual UInt_t      GetUInt(Int_t npar);
   virtual Long_t      GetLong(Int_t npar);
   virtual Long64_t    GetLong64(Int_t npar);
   virtual ULong64_t   GetULong64(Int_t npar);
   virtual Double_t    GetDouble(Int_t npar);
   virtual const char *GetString(Int_t npar);
   virtual Bool_t      GetBinary(Int_t npar, void* &mem, Long_t& size);
   virtual Bool_t      GetLargeObject(Int_t npar, void* &mem, Long_t& size);
   virtual Bool_t      GetDate(Int_t npar, Int_t& year, Int_t& month, Int_t& day);
   virtual Bool_t      GetTime(Int_t npar, Int_t& hour, Int_t& min, Int_t& sec);
   virtual Bool_t      GetDatime(Int_t npar, Int_t& year, Int_t& month, Int_t& day, Int_t& hour, Int_t& min, Int_t& sec);
   virtual Bool_t      GetTimestamp(Int_t npar, Int_t& year, Int_t& month, Int_t& day, Int_t& hour, Int_t& min, Int_t& sec, Int_t&);

   ClassDef(TPgSQLStatement, 0);  // SQL statement class for PgSQL DB
};

#endif
