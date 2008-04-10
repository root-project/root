// @(#)root/odbc:$Id$
// Author: Sergey Linev   6/02/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TODBCStatement
#define ROOT_TODBCStatement

#ifndef ROOT_TSQLStatement
#include "TSQLStatement.h"
#endif


#ifdef __CINT__
typedef void *   SQLHSTMT;
typedef UShort_t SQLUSMALLINT;
typedef UInt_t   SQLUINTEGER;
typedef Short_t  SQLSMALLINT;
typedef Short_t  SQLRETURN;
#else
#ifdef WIN32
#include "windows.h"
#endif
#include <sql.h>
#endif

class TODBCStatement : public TSQLStatement {

protected:
    #ifdef __CINT__
    struct ODBCBufferRec_t;
    #else
    struct ODBCBufferRec_t {
       Int_t       fBroottype;
       Int_t       fBsqltype;
       Int_t       fBsqlctype;
       void       *fBbuffer;
       Int_t       fBelementsize;
       SQLLEN     *fBlenarray;
       char       *fBstrbuffer;
       char       *fBnamebuffer;
    };
    #endif

protected:
   SQLHSTMT         fHstmt;
   Int_t            fBufferPreferredSize;
   ODBCBufferRec_t *fBuffer;
   Int_t            fNumBuffers;
   Int_t            fBufferLength;     // number of entries for each parameter/column
   Int_t            fBufferCounter;    // used to indicate position in buffers
   SQLUSMALLINT    *fStatusBuffer;
   Int_t            fWorkingMode;      // 1 - setting parameters, 2 - reading results, 0 - unknown
   SQLUINTEGER      fNumParsProcessed; // contains number of parameters, affected by last operation
   SQLUINTEGER      fNumRowsFetched;   // indicates number of fetched rows
   ULong64_t        fLastResultRow;    // stores values of row number after last fetch operation

   void       *GetParAddr(Int_t npar, Int_t roottype = 0, Int_t length = 0);
   long double ConvertToNumeric(Int_t npar);
   const char *ConvertToString(Int_t npar);

   Bool_t      BindColumn(Int_t ncol, SQLSMALLINT sqltype, SQLUINTEGER size);
   Bool_t      BindParam(Int_t n, Int_t type, Int_t size = 1024);

   Bool_t      ExtractErrors(SQLRETURN retcode, const char* method);

   void        SetNumBuffers(Int_t isize, Int_t ilen);
   void        FreeBuffers();

   Bool_t      IsParSettMode() const { return fWorkingMode==1; }
   Bool_t      IsResultSet() const { return fWorkingMode==2; }

public:
   TODBCStatement(SQLHSTMT stmt, Int_t rowarrsize, Bool_t errout = kTRUE);
   virtual ~TODBCStatement();

   virtual void        Close(Option_t * = "");

   virtual Int_t       GetBufferLength() const { return fBufferLength; }
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
   virtual Bool_t      GetTimestamp(Int_t npar, Int_t& year, Int_t& month, Int_t& day, Int_t& hour, Int_t& min, Int_t& sec, Int_t&);

   ClassDef(TODBCStatement, 0); //ODBC implementation of TSQLStatement
};

#endif
