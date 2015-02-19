// @(#)root/mysql:$Id$
// Author: Sergey Linev   6/02/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMySQLStatement
#define ROOT_TMySQLStatement

#ifndef ROOT_TSQLStatement
#include "TSQLStatement.h"
#endif


#include <mysql.h>

#if MYSQL_VERSION_ID < 40100
typedef struct { int dummy; } MYSQL_STMT;
typedef struct { int dummy; } MYSQL_BIND;
#endif

class TMySQLStatement : public TSQLStatement {

protected:

   struct TParamData {
      void*         fMem;        //! allocated data buffer
      Int_t         fSize;       //! size of allocated data
      Int_t         fSqlType;     //! sqltype of parameter
      Bool_t        fSign;        //! signed - not signed type
      ULong_t       fResLength;  //! length argument
      my_bool       fResNull;    //! indicates if argument is null
      char*         fStrBuffer;  //! special buffer to be used for string conversions
      char*         fFieldName;  //! buffer for field name
   };

   MYSQL_STMT           *fStmt;          //! executed statement
   Int_t                 fNumBuffers; //! number of statement parameters
   MYSQL_BIND           *fBind;          //! array of bind data
   TParamData           *fBuffer;         //! parameter definition structures
   Int_t                 fWorkingMode;   //! 1 - setting parameters, 2 - retrieving results
   Int_t                 fIterationCount;//! number of iteration
   Bool_t                fNeedParBind;   //! indicates when parameters bind should be called

   Bool_t      IsSetParsMode() const { return fWorkingMode==1; }
   Bool_t      IsResultSetMode() const { return fWorkingMode==2; }

   Bool_t      SetSQLParamType(Int_t npar, int sqltype, Bool_t sig, ULong_t sqlsize = 0);

   long double ConvertToNumeric(Int_t npar);
   const char *ConvertToString(Int_t npar);

   void        FreeBuffers();
   void        SetBuffersNumber(Int_t n);

   void       *BeforeSet(const char* method, Int_t npar, Int_t sqltype, Bool_t sig = kTRUE, ULong_t size = 0);

   static ULong64_t fgAllocSizeLimit;

private:
   TMySQLStatement(const TMySQLStatement&);            // Not implemented.
   TMySQLStatement &operator=(const TMySQLStatement&); // Not implemented.

public:
   TMySQLStatement(MYSQL_STMT* stmt, Bool_t errout = kTRUE);
   virtual ~TMySQLStatement();

   static ULong_t GetAllocSizeLimit() { return fgAllocSizeLimit; }
   static void SetAllocSizeLimit(ULong_t sz) { fgAllocSizeLimit = sz; }

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
   virtual Bool_t      GetDate(Int_t npar, Int_t& year, Int_t& month, Int_t& day);
   virtual Bool_t      GetTime(Int_t npar, Int_t& hour, Int_t& min, Int_t& sec);
   virtual Bool_t      GetDatime(Int_t npar, Int_t& year, Int_t& month, Int_t& day, Int_t& hour, Int_t& min, Int_t& sec);
   virtual Bool_t      GetTimestamp(Int_t npar, Int_t& year, Int_t& month, Int_t& day, Int_t& hour, Int_t& min, Int_t& sec, Int_t&);

   ClassDef(TMySQLStatement, 0);  // SQL statement class for MySQL DB
};

#endif
