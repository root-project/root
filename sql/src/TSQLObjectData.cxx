#include "TSQLObjectData.h"

#include "TObjArray.h"
#include "TNamed.h"
#include "TSQLRow.h"
#include "TSQLResult.h"
#include "TSQLClassInfo.h"

ClassImp(TSQLObjectData)


TSQLObjectData::TSQLObjectData() :
   TObject(),
   fInfo(0),
   fObjId(0),
   fClassData(0),
   fBlobData(0),
   fLocatedColumn(-1),
   fClassRow(0),
   fBlobRow(0),
   fLocatedField(0),
   fLocatedValue(0),
   fCurrentBlob(kFALSE),
   fBlobName1(),
   fBlobName2(),
   fUnpack(0)
{
}

TSQLObjectData::TSQLObjectData(TSQLClassInfo* sqlinfo,
                               Int_t          objid,
                               TSQLResult*    classdata,
                               TSQLResult*    blobdata) :
   TObject(),
   fInfo(sqlinfo),
   fObjId(objid),
   fClassData(classdata),
   fBlobData(blobdata),
   fLocatedColumn(-1),
   fClassRow(0),
   fBlobRow(0),
   fLocatedField(0),
   fLocatedValue(0),
   fCurrentBlob(kFALSE),
   fBlobName1(),
   fBlobName2(),
   fUnpack(0)
{
   if (fClassData!=0)
     fClassRow = fClassData->Next();
   if (fBlobData!=0)
     fBlobRow = fBlobData->Next();
     
}

TSQLObjectData::~TSQLObjectData()
{
   if (fClassRow!=0) delete fClassRow; 
   if (fBlobRow!=0) delete fBlobRow;
   if (fClassData!=0) delete fClassData;
   if (fBlobData!=0) delete fBlobData;
   if (fUnpack!=0) { fUnpack->Delete(); delete fUnpack; }
}

Int_t TSQLObjectData::GetNumClassFields()
{
   if (fClassData!=0) return fClassData->GetFieldCount(); 
   return 0; 
}

const char* TSQLObjectData::GetClassFieldName(Int_t n)
{
   if (fClassData!=0) return fClassData->GetFieldName(n); 
   return 0;
}


Bool_t TSQLObjectData::LocateColumn(const char* colname, Bool_t isblob)
{
   if (fUnpack!=0) { 
     fUnpack->Delete(); 
     delete fUnpack; 
     fUnpack = 0;
   } 
    
   fLocatedField = 0;
   fLocatedValue = 0;
   fCurrentBlob = kFALSE;

   if ((fClassData==0) || (fClassRow==0)) return kFALSE;
   
   Int_t numfields = GetNumClassFields();
   
   for (Int_t ncol=1;ncol<numfields;ncol++) {
      const char* fieldname = GetClassFieldName(ncol); 
      if (strcmp(colname, fieldname)==0) {
         fLocatedColumn = ncol; 
         fLocatedField = fieldname;
         fLocatedValue = fClassRow->GetField(ncol);
         break; 
      }
   }
   
   if (fLocatedField==0) return kFALSE;   
   
   if (!isblob) return kTRUE;
   
   if (fBlobRow==0) return kFALSE;
   
   fCurrentBlob = kTRUE;
   
   ExtractBlobValues();
      
   return kTRUE;
}

Bool_t TSQLObjectData::ExtractBlobValues()
{
   if (fBlobRow==0) return kFALSE;
   
   fLocatedValue = fBlobRow->GetField(1);
   
   const char* name = fBlobRow->GetField(0);
   const char* separ = strstr(name, ":"); //SQLNameSeparator()
        
   if (separ==0) {
      fBlobName1 = "";
      fBlobName2 = name;
   } else {
      fBlobName1 = "";
      fBlobName1.Append(name, separ-name);   
      separ+=strlen(":"); //SQLNameSeparator()
      fBlobName2 = separ;   
   }
   
   return kTRUE;
}

void TSQLObjectData::AddUnpack(const char* tname, const char* value)
{
   TNamed* str = new TNamed(tname, value); 
   if (fUnpack==0) {
      fUnpack = new TObjArray();
      fBlobName1 = "";
      fBlobName2 = str->GetName();
      fLocatedValue = str->GetTitle();
   }
   
   fUnpack->Add(str);
}

void TSQLObjectData::AddUnpackInt(const char* tname, Int_t value)
{
   TString sbuf;
   sbuf.Form("%d", value);
   AddUnpack(tname, sbuf.Data()); 
}


void TSQLObjectData::ShiftToNextValue()
{
   Bool_t doshift = kTRUE; 
    
   if (fUnpack!=0) {
      TObject* prev = fUnpack->First();
      fUnpack->Remove(prev);
      delete prev; 
      fUnpack->Compress();
      if (fUnpack->GetLast()>=0) {
         TNamed* curr = (TNamed*) fUnpack->First();
         fBlobName1 = "";
         fBlobName2 = curr->GetName();
         fLocatedValue = curr->GetTitle();
         return; 
      }
      delete fUnpack;
      fUnpack = 0;
      doshift = kFALSE;
   }
    
   if (fCurrentBlob>0) {
      if (doshift) {
         delete fBlobRow;
         fBlobRow = fBlobData->Next();
      }
      ExtractBlobValues();
   } else 
   if (fClassData!=0) {
      if (doshift) fLocatedColumn++;
      if (fLocatedColumn<GetNumClassFields()) {
         fLocatedField = GetClassFieldName(fLocatedColumn); 
         fLocatedValue = fClassRow->GetField(fLocatedColumn);
      } else {
         fLocatedField = 0;
         fLocatedValue = 0;
      }
   }
}

Bool_t TSQLObjectData::VerifyDataType(const char* tname, Bool_t errormsg)
{
   if (tname==0) {
      if (errormsg)
        Error("VerifyDataType","Data type not specified"); 
      return kFALSE;    
   }
   
   // here maybe type of column can be checked
   if (!IsBlobData()) return kTRUE;
   
   if (fBlobName2!=tname) {
      if (errormsg) 
         Error("VerifyDataType","Data type meissmatch %s - %s", fBlobName2.Data(), tname); 
      return kFALSE;    
   }
   
   return kTRUE;
}

Bool_t TSQLObjectData::PrepareForRawData()
{
   if (!ExtractBlobValues()) return kFALSE;
   
   fCurrentBlob = kTRUE;
   
   return kTRUE;
}

