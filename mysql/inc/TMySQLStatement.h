// @(#)root/mysql:$Name:  $:$Id: TMySQLStatement.h,v 1.3 2006/06/02 14:02:03 brun Exp $
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

#if !defined(__CINT__)
#ifndef R__WIN32
#include <sys/time.h>
#endif
#include <mysql.h>

#if MYSQL_VERSION_ID < 40100
typedef struct { int dummy; } MYSQL_STMT;
typedef struct { int dummy; } MYSQL_BIND;
#endif

#else
struct MYSQL_STMT;
struct MYSQL_BIND;
typedef char my_bool;
#endif

class TMySQLStatement : public TSQLStatement {

private:

   struct TParamData {
      void*         buffer;     //! allocated data buffer
      Int_t         fSize;       //! size of allocated data
      Int_t         sqltype;     //! sqltype of parameter
      Bool_t        sign;        //! signed - not signed type
      unsigned long fResLength;  //! length argument
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

   Bool_t      SetSQLParamType(Int_t npar, int sqltype, bool sig, int sqlsize = 0);

   long double ConvertToNumeric(Int_t npar);
   const char *ConvertToString(Int_t npar);

   void        FreeBuffers();
   void        SetBuffersNumber(Int_t n);

   void       *BeforeSet(Int_t npar, Int_t sqltype, Bool_t sig = kTRUE, Int_t size = 0);

public:
   TMySQLStatement(MYSQL_STMT* stmt, Bool_t errout = kTRUE);
   virtual ~TMySQLStatement();

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

   ClassDef(TMySQLStatement, 0);
};

#endif
