// @(#)root/sql:$Name:  $:$Id: TSQLObjectData.h,v 1.3 2005/11/24 16:57:23 pcanal Exp $
// Author: Sergey Linev  20/11/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSQLObjectData
#define ROOT_TSQLObjectData


/////////////////////////////////////////////////////////////////////////
//                                                                     //
// TSQLObjectData keeps data, requested from SQL server for object     //
//                                                                     //
/////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

class TObjArray;
class TSQLClassInfo;
class TSQLResult;
class TSQLRow;

class TSQLObjectData : public TObject {

public:
   TSQLObjectData();
   
   TSQLObjectData(TSQLClassInfo* sqlinfo,
                  Int_t          objid,
                  TSQLResult*    classdata,
                  TSQLResult*    blobdata);
   
   virtual ~TSQLObjectData();
   
   Int_t             GetObjId() const { return fObjId; }
   
   Bool_t            LocateColumn(const char* colname, Bool_t isblob = kFALSE);
   Bool_t            IsBlobData() const { return fCurrentBlob || (fUnpack!=0); }
   void              ShiftToNextValue();
   
   void              AddUnpack(const char* tname, const char* value);
   void              AddUnpackInt(const char* tname, Int_t value);
   
   const char*       GetValue() const { return fLocatedValue; }
   const char*       GetColumnName() const { return fLocatedField; }
   const char*       GetBlobName1() const { return fBlobName1.Data(); }
   const char*       GetBlobName2() const { return fBlobName2.Data(); }
   
   Bool_t            VerifyDataType(const char* tname, Bool_t errormsg = kTRUE);
   Bool_t            PrepareForRawData();
   
protected: 
   Bool_t            ExtractBlobValues();
   
   Int_t             GetNumClassFields();
   const char*       GetClassFieldName(Int_t n);
   
   TSQLClassInfo*    fInfo;          //!
   Int_t             fObjId;         //!
   TSQLResult*       fClassData;     //!
   TSQLResult*       fBlobData;      //!
   Int_t             fLocatedColumn; //!
   Int_t             fLocatedBlob;   //!
   TSQLRow*          fClassRow;      //!
   TSQLRow*          fBlobRow;       //!
   const char*       fLocatedField;  //!
   const char*       fLocatedValue;  //!
   Bool_t            fCurrentBlob;   //!
   TString           fBlobName1;     //!
   TString           fBlobName2;     //!
   TObjArray*        fUnpack;        //! 
   
   ClassDef(TSQLObjectData, 1); // Keeps the data requested from the SQL server for an object.
      
};

#endif
