// @(#)root/oracle:$Name: v4-00-08 $:$Id: TOracleRow.cxx,v 1.0 2004/12/04 17:00:45 rdm Exp $
// Author: Yan Liu and Shaowen Wang   23/11/04

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TOracleRow.h"

ClassImp(TOracleRow);

//______________________________________________________________________________
TOracleRow::TOracleRow(ResultSet *rs, vector<MetaData> *fieldMetaData)
{
   // Single row of query result.

   fResult      = rs;
   fFieldInfo   = fieldMetaData;
   fUpdateCount = 0;
   fResultType  = 1;
}

//______________________________________________________________________________
TOracleRow::TOracleRow(UInt_t updateCount)
{
   fUpdateCount = updateCount;
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
   fResult      = 0;
   fResultType  = 0;
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
const char *TOracleRow::GetField(Int_t field)
{
   // Note: Index starts from 0, not 1 as oracle call does.
   // Data Type conversion:
   // C++type   OracleType
   // Uint       2:0:x [type:precision:scale]
   // float     2:noneZero:nonZero
   // double    2:nonZero:-127
   // string    1:x:x
   // string    188:0:x - Timestamp

   // Get specified field from row (0 <= field < GetFieldCount()).

   if (!IsValid(field) || !fResult || !fFieldInfo) {
      Error("TOracleRow","GetField(): out-of-range or No ResultSet/MetaData");
      return 0;
   }
   int fDataType, fDataSize, fPrecision, fScale;
   fDataType = (*fFieldInfo)[field].getInt(MetaData::ATTR_DATA_TYPE);
   fDataSize = (*fFieldInfo)[field].getInt(MetaData::ATTR_DATA_SIZE);
   fPrecision = (*fFieldInfo)[field].getInt(MetaData::ATTR_PRECISION);
   fScale = (*fFieldInfo)[field].getInt(MetaData::ATTR_SCALE);

   switch (fDataType) {
      case 2: //NUMBER
         if (fScale == 0) {
            return (const char *)(new unsigned int(fResult->getUInt(field+1)));
         } else if (fPrecision != 0 && fScale == -127) {
            return (const char *)(new double(fResult->getDouble(field+1)));
         } else if (fScale > 0) {
            return (const char *)(new float(fResult->getFloat(field+1)));
         }
         break;
      case 1: //VARCHAR2
         return (const char *)(new string(fResult->getString(field+1)));
      case 188: //Timestamp - oracle's date/time class. convert to string
         {
            Timestamp ts = fResult->getTimestamp(field+1);
            string s = ts.toText("MM/dd/YYYY, HH:MM:SS",0);
            return (const char *)s.c_str();
         }
      default:
         return 0;
   }
   return 0;
}
