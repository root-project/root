// @(#)root/net:$Name:  $:$Id: TSQLStructure.h,v 1.2 2005/11/22 11:30:00 brun Exp $
// Author: Sergey Linev  20/11/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSQLStructure
#define ROOT_TSQLStructure


/////////////////////////////////////////////////////////////////////////
//                                                                     //
// TSQLStructure is special class, used in TSQLBuffer for data convers //
//                                                                     //
/////////////////////////////////////////////////////////////////////////



#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

#ifndef ROOT_TAttAxis
#include "TAttAxis.h"
#endif

class TStreamerInfo;
class TStreamerInfo;
class TStreamerElement;
class TSQLFile;
class TSqlRegistry;
class TSQLObjectData;

class TSQLColumnData : public TObject {
    
protected:   
   TString     fName;             //!  name of the table column
   TString     fType;             //!  type of the table column
   TString     fValue;            //!  value of the table column
   Bool_t      fNumeric;          //!  for numeric quotes (double quotes) are not required
public:
   TSQLColumnData();
   TSQLColumnData(const char* name,            
                  const char* sqltype, 
                  const char* value, 
                  Bool_t numeric);

   TSQLColumnData(const char* name, Int_t value);
   virtual ~TSQLColumnData();
  
   virtual const char* GetName() const { return fName.Data(); }
   const char* GetType() const { return fType.Data(); }
   const char* GetValue() const { return fValue.Data(); }
   Bool_t IsNumeric() const { return fNumeric; }
  
   ClassDef(TSQLColumnData, 1); // Single SQL column data.
};

//______________________________________________________________________

class TSQLStructure : public TObject {
protected:   

   TString          MakeArrayIndex(TStreamerElement* elem, Int_t n);
   Bool_t           CheckNormalClassPair(TSQLStructure* vers, TSQLStructure* info);

   Int_t            FindMaxRef();
   void             PerformConversion(TSqlRegistry* reg, TObjArray* blobs, const char* topname, Bool_t useblob = kFALSE);
   Bool_t           StoreObject(TSqlRegistry* reg, const char* objid, TClass* cl, Bool_t registerobj = kTRUE);
   Bool_t           StoreObjectInNormalForm(TSqlRegistry* reg);
   Bool_t           StoreClassInNormalForm(TSqlRegistry* reg);
   Bool_t           StoreElementInNormalForm(TSqlRegistry* reg, TObjArray* columns);
   Bool_t           TryConvertObjectArray(TSqlRegistry* reg, TObjArray* blobs);

   Bool_t           StoreTObject(TSqlRegistry* reg);
   Bool_t           StoreTString(TSqlRegistry* reg);
   Bool_t           RecognizeTString(const char* &value);

   void             AddCmd(TObjArray* cmds, const char* name, const char* value, const char* topname = 0, const char* ns = 0);

   TSQLStructure*   fParent;     //!
   Int_t            fType;       //!
   const void*      fPointer;    //!
   TString          fValue;      //!
   Int_t            fArrayIndex; //!
   Int_t            fRepeatCnt;  //!
   TObjArray        fChilds;     //!

public:
   TSQLStructure();
   virtual ~TSQLStructure();
  
   TSQLStructure*   GetParent() const { return fParent; }
   void             SetParent(TSQLStructure* p) { fParent = p; }
   Int_t            NumChilds() const;
   TSQLStructure*   GetChild(Int_t n) const;
 
   void             SetType(Int_t typ) { fType = typ; }
   Int_t            GetType() const { return fType; }
   
   // this part requried for writing to SQL tables
   void             SetObjectRef(Int_t refid, const TClass* cl);
   void             SetObjectPointer(Int_t ptrid);
   void             SetVersion(const TClass* cl, Int_t version = -100);
   void             SetClassStreamer(const TClass* cl);
   void             SetStreamerInfo(const TStreamerInfo* info);
   void             SetStreamerElement(const TStreamerElement* elem, Int_t number);
   void             SetValue(const char* value, const char* tname = 0);
   void             SetArrayIndex(Int_t indx, Int_t cnt=1);
   void             SetArray(Int_t sz = -1);
   void             ChangeValueOnly(const char* value);
   
   TClass*          GetObjectClass() const;
   TClass*          GetVersionClass() const;
   TStreamerInfo*   GetStreamerInfo() const;
   TStreamerElement* GetElement() const;
   Int_t            GetElementNumber() const;
   const char*      GetValueType() const;
   const char*      GetValue() const;
   Int_t            GetArrayIndex() const { return fArrayIndex; }
   Int_t            GetRepeatCounter() const { return fRepeatCnt; }
  
   void             Add(TSQLStructure* child);
   void             AddVersion(const TClass* cl, Int_t version = -100);
   void             AddValue(const char* value, const char* tname = 0);
   void             ChildArrayIndex(Int_t index, Int_t cnt = 1);

   // this is part specially for reading of sql tables
  
   Int_t            DefineObjectId();
   Bool_t           IsClonesArray();
  
   void             SetObjectData(TSQLObjectData* objdata);
   void             AddObjectData(TSQLObjectData* objdata);
   TSQLObjectData*  GetObjectData(Bool_t search = false);
  
   virtual void     Print(Option_t* option = "") const;
   void             PrintLevel(Int_t level) const;
  
   Bool_t           ConvertToTables(TSQLFile* f, Int_t keyid, TObjArray* cmds);
  
   Int_t            LocateElementColumn(TSQLFile* f, TSQLObjectData* data);

   static Bool_t    UnpackTObject(TSQLFile* f, TSQLObjectData* data, Int_t objid, Int_t clversion);
   static Bool_t    UnpackTString(TSQLFile* f, TSQLObjectData* data, Int_t objid, Int_t clversion);
   static Bool_t    IsNumericType(Int_t typ);
   static const char* GetSimpleTypeName(Int_t typ);
   static Int_t     DefineElementColumnType(TStreamerElement* elem, TSQLFile* f);
  
   enum ESQLTypes {
     kSqlObject       = 10001,
     kSqlPointer      = 10002,
     kSqlVersion      = 10003,
     kSqlStreamerInfo = 10004,
     kSqlClassStreamer= 10005,
     kSqlElement      = 10006,
     kSqlValue        = 10007,
     kSqlArray        = 10008,
     kSqlObjectData   = 10009
   };
   
   enum ESQLColumns {
     kColUnknown      = 0,
     kColSimple       = 1,
     kColSimpleArray  = 2,
     kColParent       = 3,
     kColObject       = 4,
     kColObjectArray  = 5,
     kColObjectPtr    = 6,
     kColTString      = 7,
     kColClonesArray  = 8,
     kColRawData      = 9
   };   
  
   ClassDef(TSQLStructure, 1); // Table/structure description used internally by YBufferSQL.
};

// text constants, used in SQL I/O

namespace sqlio {
   extern const Int_t Ids_NullPtr;
   extern const Int_t Ids_RootDir;
   extern const Int_t Ids_StreamerInfos;
   extern const Int_t Ids_FirstKey;
   extern const Int_t Ids_FirstObject;
    
   extern const char* ObjectRef;
   extern const char* ObjectRef_Arr;
   extern const char* ObjectPtr;
   extern const char* ObjectInst;
   extern const char* Version;
   extern const char* TObjectUniqueId;
   extern const char* TObjectBits;
   extern const char* TObjectProcessId;
   extern const char* TStringValue;
   extern const char* IndexSepar;
   extern const char* RawSuffix;
   extern const char* ParentSuffix;
   extern const char* ObjectSuffix;
   extern const char* PointerSuffix;
   extern const char* StrSuffix;
   extern const char* LongStrPrefix;
    
   extern const char* Array;
   extern const char* Bool;
   extern const char* Char;
   extern const char* Short;
   extern const char* Int;
   extern const char* Long;
   extern const char* Long64;
   extern const char* Float;
   extern const char* Double;
   extern const char* UChar;
   extern const char* UShort;
   extern const char* UInt;
   extern const char* ULong;
   extern const char* ULong64;
   extern const char* CharStar;
   extern const char* True;
   extern const char* False;
    
   extern const char* KeysTable;
   extern const char* KT_Name;
   extern const char* KT_Datetime;
   extern const char* KT_Cycle;
   extern const char* KT_Class;
    
   extern const char* ObjectsTable;
   extern const char* OT_Class;
   extern const char* OT_Version;
    
   extern const char* BT_Field;
   extern const char* BT_Value;
    
   extern const char* StringsTable;
   extern const char* ST_Value;
    
   extern const char* ConfigTable;
   extern const char* CT_Field;
   extern const char* CT_Value;
    
   extern const char* cfg_Version;
   extern const char* cfg_UseSufixes;
   extern const char* cfg_ArrayLimit;
};

#endif
