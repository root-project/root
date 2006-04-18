// @(#)root/odbc:$Name:  $:$Id: TODBCRow.cxx,v 1.1 2006/04/17 14:12:52 rdm Exp $
// Author: Sergey Linev   6/02/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TODBCRow.h"
#include "TODBCTypes.h"

#include <sqlext.h>


ClassImp(TODBCRow)

//______________________________________________________________________________
TODBCRow::TODBCRow(SQLHSTMT stmt, Int_t fieldcount)
{
   // Single row of query result.
   fHstmt = stmt;
   fFieldCount = fieldcount;

   fBuffer = 0;

   if (fFieldCount>0) {
      fBuffer = new char*[fFieldCount];
      for (Int_t n = 0; n < fFieldCount; n++)
         fBuffer[n] = 0;
   }
}

//______________________________________________________________________________
TODBCRow::~TODBCRow()
{
   // Destroy row object.

   Close();
}

//______________________________________________________________________________
void TODBCRow::Close(Option_t *)
{
   // Close row.

   if (fBuffer!=0) {
      for (Int_t n = 0; n < fFieldCount; n++)
         delete[] fBuffer[n];
     delete[] fBuffer;
     fBuffer = 0;
   }

}

//______________________________________________________________________________
ULong_t TODBCRow::GetFieldLength(Int_t)
{
   // Get length in bytes of specified field.

   return 0;
}

//______________________________________________________________________________
const char *TODBCRow::GetField(Int_t field)
{
   // Get specified field from row (0 <= field < GetFieldCount()).

   if ((field<0) || (field>=fFieldCount)) return 0;

   if (fBuffer[field]!=0) return fBuffer[field];

   #define buffer_len 2048

   fBuffer[field] = new char[buffer_len];

   ODBCInt_t  ressize;

   SQLGetData(fHstmt, field+1, SQL_C_CHAR, fBuffer[field], buffer_len, &ressize);

   return fBuffer[field];
}
