// @(#)root/oracle:$Name:  $:$Id: TOracleRow.cxx,v 1.2 2005/04/25 17:21:11 rdm Exp $
// Author: Yan Liu and Shaowen Wang   23/11/04

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TOracleRow.h"
#include <Riostream.h>

using namespace std;


ClassImp(TOracleRow);

//______________________________________________________________________________
TOracleRow::TOracleRow(ResultSet *rs, vector<MetaData> *fieldMetaData)
{
   // Single row of query result.

   fResult      = rs;
   fFieldInfo   = fieldMetaData;
   fFieldCount  = fFieldInfo->size();
   fFields      = new vector<string>(fFieldCount, "");
   GetRowData();
   fUpdateCount = 0;
   fResultType  = 1;
}

//______________________________________________________________________________
TOracleRow::TOracleRow(UInt_t updateCount)
{
   fUpdateCount = updateCount;
   fFieldCount  = 0;
   fFields      = 0;
   fResultType  = 0;
}

//______________________________________________________________________________
TOracleRow::~TOracleRow()
{
   // Destroy row object.

   if (fResultType >= 0) {
      Close();
   }
}

//______________________________________________________________________________
void TOracleRow::Close(Option_t *)
{
   // Close row.

   if (fResultType == -1)
      return;

   fFieldInfo   = 0;
   fFieldCount  = 0;
   fResult      = 0;
   fResultType  = 0;
   if (fFields)
      delete fFields;
}

//______________________________________________________________________________
Bool_t TOracleRow::IsValid(Int_t field)
{
   // Check if row is open and field index within range.

   if (!fResult) {
      Error("IsValid", "row closed");
      return kFALSE;
   }
   if (field < 0 || field >= (Int_t)fFieldInfo->size()) {
      Error("IsValid", "field index out of bounds");
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
ULong_t TOracleRow::GetFieldLength(Int_t field)
{
   // Get length in bytes of specified field.

   if (!IsValid(field) || fFieldInfo->size() <= 0)
      return 0;

   MetaData fieldMD = (*fFieldInfo)[field];

   return fieldMD.getInt(MetaData::ATTR_DATA_SIZE);
}

//______________________________________________________________________________
int TOracleRow::GetRowData()
{
   // Fetch a row from current resultset into fFields vector as ASCII
   // supported Oracle internal types conversion:
   // NUMBER -> int, float, double -> (char *)
   // CHAR   -> (char *)
   // VARCHAR2 -> (char *)
   // TIMESTAMP -> Timestamp -> (char *)
   // DATE -> (char *)
   // NOTE: above types are tested to work and this is the default impl.

   if (!fFields || !fResult || !fFieldInfo) {
      Error("GetRowData()", "Empty row, resultset or MetaData");
      return 0;
   }
   int fDataType, fDataSize, fPrecision, fScale;
   char str_number[1024];
   int int_val; double double_val; float float_val;
   try {
      for (UInt_t i=0; i<fFieldCount; i++) {
         if (fResult->isNull(i+1)) {
            (*fFields)[i] = "";
         } else {
            fDataType = (*fFieldInfo)[i].getInt(MetaData::ATTR_DATA_TYPE);
            fDataSize = (*fFieldInfo)[i].getInt(MetaData::ATTR_DATA_SIZE);
            fPrecision = (*fFieldInfo)[i].getInt(MetaData::ATTR_PRECISION);
            fScale = (*fFieldInfo)[i].getInt(MetaData::ATTR_SCALE);
            switch (fDataType) {
               case 2: //NUMBER
                  if (fScale == 0 || fPrecision == 0) {
                     (*fFields)[i] = fResult->getString(i+1);
                     break;
                  } else if (fPrecision != 0 && fScale == -127) {
                     double_val = fResult->getDouble(i+1);
                     sprintf(str_number, "%lf", double_val);
                  } else if (fScale > 0) {
                     float_val = fResult->getFloat(i+1);
                     sprintf(str_number, "%f", float_val);
                  }
                  (*fFields)[i] = str_number;
                  break;
               case 96:  // CHAR
               case 1:   // VARCHAR2
                  (*fFields)[i] = fResult->getString(i+1);
                  break;
               case 187: // TIMESTAMP
               case 232: // TIMESTAMP WITH LOCAL TIMEZONE
               case 188: // TIMESTAMP WITH TIMEZONE
                  (*fFields)[i] = (fResult->getTimestamp(i+1)).toText("MM/DD/YYYY, HH24:MI:SS",0);
                  break;
               case 12: // DATE
                  //to fetch DATE, getDate() does NOT work in occi
                  (*fFields)[i] = fResult->getString(i+1);
                  break;
               default:
                  Error("GetRowData()","Oracle type %d not supported.", fDataType);
                  return 0;
            }
         }
      }
      return 1;
   } catch (SQLException &oraex) {
      Error("GetRowData()", (oraex.getMessage()).c_str());
      return 0;
   }
}

//______________________________________________________________________________
int TOracleRow::GetRowData2()
{
   // Fetch a row from current resultset into fFields vector as ASCII.
   // This impl only use getString() to fetch column value.
   // Pro: support all type conversions, indicated by OCCI API doc
   // Con: not tested for each Oracle internal type. it has problem on
   //      Timestamp or Date type conversion.
   // NOTE: This is supposed to be the easiest and direct implemention.

   if (!fFields || !fResult || !fFieldInfo) {
      Error("GetRowData2()", "Empty row, resultset or MetaData");
      return 0;
   }

   try {
      for (UInt_t i=0; i<fFieldCount; i++) {
         if (fResult->isNull(i+1)) {
            (*fFields)[i] = "";
         } else {
            (*fFields)[i] = fResult->getString(i+1);
         }
      }
      return 1;
   } catch (SQLException &oraex) {
      Error("GetRowData2()", (oraex.getMessage()).c_str());
      return 0;
   }
}

//______________________________________________________________________________
const char *TOracleRow::GetField(Int_t field)
{
   if (!IsValid(field) || !fResult || !fFields) {
      Error("TOracleRow","GetField(): out-of-range or No RowData/ResultSet/MetaData");
      return 0;
   }
   return (*fFields)[field].c_str();
}
