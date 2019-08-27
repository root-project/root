// @(#)root/sql:$Id$
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

#include "TObjArray.h"

#ifdef Bool
#undef Bool
#endif
#ifdef True
#undef True
#endif
#ifdef False
#undef False
#endif

class TStreamerInfo;
class TStreamerInfo;
class TStreamerElement;
class TSQLFile;
class TSqlRegistry;
class TSqlRawBuffer;
class TSQLObjectData;
class TSQLClassInfo;
class TBufferSQL2;

class TSQLColumnData final : public TObject {

protected:
   TString fName;           ///<!  name of the table column
   TString fType;           ///<!  type of the table column
   TString fValue;          ///<!  value of the table column
   Bool_t fNumeric{kFALSE}; ///<!  for numeric quotes (double quotes) are not required
public:
   TSQLColumnData(const char *name, const char *sqltype, const char *value, Bool_t numeric);

   TSQLColumnData(const char *name, Long64_t value);

   const char *GetName() const final { return fName.Data(); }
   const char *GetType() const { return fType.Data(); }
   const char *GetValue() const { return fValue.Data(); }
   Bool_t IsNumeric() const { return fNumeric; }

   ClassDefOverride(TSQLColumnData, 1); // Single SQL column data.
};

//______________________________________________________________________

class TSQLTableData : public TObject {

protected:
   TSQLFile *fFile{nullptr};      ///<!
   TSQLClassInfo *fInfo{nullptr}; ///<!
   TObjArray fColumns;            ///<! collection of columns
   TObjArray *fColInfos{nullptr}; ///<! array with TSQLClassColumnInfo, used later for TSQLClassInfo

   TString DefineSQLName(const char *fullname);
   Bool_t HasSQLName(const char *sqlname);

public:
   TSQLTableData(TSQLFile *f = nullptr, TSQLClassInfo *info = nullptr);
   virtual ~TSQLTableData();

   void AddColumn(const char *name, Long64_t value);
   void AddColumn(const char *name, const char *sqltype, const char *value, Bool_t numeric);

   TObjArray *TakeColInfos();

   Int_t GetNumColumns();
   const char *GetColumn(Int_t n);
   Bool_t IsNumeric(Int_t n);

   ClassDef(TSQLTableData, 1); // Collection of columns data for single SQL table
};

//______________________________________________________________________

class TSQLStructure : public TObject {
protected:
   Bool_t CheckNormalClassPair(TSQLStructure *vers, TSQLStructure *info);

   Long64_t FindMaxObjectId();
   void PerformConversion(TSqlRegistry *reg, TSqlRawBuffer *blobs, const char *topname, Bool_t useblob = kFALSE);
   Bool_t StoreObject(TSqlRegistry *reg, Long64_t objid, TClass *cl, Bool_t registerobj = kTRUE);
   Bool_t StoreObjectInNormalForm(TSqlRegistry *reg);
   Bool_t StoreClassInNormalForm(TSqlRegistry *reg);
   Bool_t StoreElementInNormalForm(TSqlRegistry *reg, TSQLTableData *columns);
   Bool_t TryConvertObjectArray(TSqlRegistry *reg, TSqlRawBuffer *blobs);

   Bool_t StoreTObject(TSqlRegistry *reg);
   Bool_t StoreTString(TSqlRegistry *reg);
   Bool_t RecognizeTString(const char *&value);

   TSQLStructure *fParent{nullptr}; //!
   Int_t fType{0};                  //!
   const void *fPointer{nullptr};   //!
   TString fValue;                  //!
   Int_t fArrayIndex{-1};           //!
   Int_t fRepeatCnt{0};             //!
   TObjArray fChilds;               //!

public:
   TSQLStructure() = default;
   virtual ~TSQLStructure();

   TSQLStructure *GetParent() const { return fParent; }
   void SetParent(TSQLStructure *p) { fParent = p; }
   Int_t NumChilds() const;
   TSQLStructure *GetChild(Int_t n) const;

   void SetType(Int_t typ) { fType = typ; }
   Int_t GetType() const { return fType; }

   // this part requried for writing to SQL tables
   void SetObjectRef(Long64_t refid, const TClass *cl);
   void SetObjectPointer(Long64_t ptrid);
   void SetVersion(const TClass *cl, Int_t version = -100);
   void SetClassStreamer(const TClass *cl);
   void SetStreamerInfo(const TStreamerInfo *info);
   void SetStreamerElement(const TStreamerElement *elem, Int_t number);
   void SetCustomClass(const TClass *cl, Version_t version);
   void SetCustomElement(TStreamerElement *elem);
   void SetValue(const char *value, const char *tname = 0);
   void SetArrayIndex(Int_t indx, Int_t cnt = 1);
   void SetArray(Int_t sz = -1);
   void ChangeValueOnly(const char *value);

   TClass *GetObjectClass() const;
   TClass *GetVersionClass() const;
   TStreamerInfo *GetStreamerInfo() const;
   TStreamerElement *GetElement() const;
   Int_t GetElementNumber() const;
   TClass *GetCustomClass() const;
   Version_t GetCustomClassVersion() const;
   Bool_t GetClassInfo(TClass *&cl, Version_t &version);
   const char *GetValueType() const;
   const char *GetValue() const;
   Int_t GetArrayIndex() const { return fArrayIndex; }
   Int_t GetRepeatCounter() const { return fRepeatCnt; }

   void Add(TSQLStructure *child);
   void AddVersion(const TClass *cl, Int_t version = -100);
   void AddValue(const char *value, const char *tname = 0);
   void ChildArrayIndex(Int_t index, Int_t cnt = 1);

   // this is part specially for reading of sql tables

   Long64_t DefineObjectId(Bool_t recursive = kTRUE);

   void SetObjectData(TSQLObjectData *objdata);
   void AddObjectData(TSQLObjectData *objdata);
   TSQLObjectData *GetObjectData(Bool_t search = false);

   virtual void Print(Option_t *option = "") const;
   void PrintLevel(Int_t level) const;

   Bool_t ConvertToTables(TSQLFile *f, Long64_t keyid, TObjArray *cmds);

   Int_t LocateElementColumn(TSQLFile *f, TBufferSQL2 *buf, TSQLObjectData *data);

   static Bool_t UnpackTObject(TSQLFile *f, TBufferSQL2 *buf, TSQLObjectData *data, Long64_t objid, Int_t clversion);
   static Bool_t UnpackTString(TSQLFile *f, TBufferSQL2 *buf, TSQLObjectData *data, Long64_t objid, Int_t clversion);
   static Bool_t IsNumericType(Int_t typ);
   static const char *GetSimpleTypeName(Int_t typ);
   static TString MakeArrayIndex(TStreamerElement *elem, Int_t n);
   static Int_t DefineElementColumnType(TStreamerElement *elem, TSQLFile *f);
   static TString DefineElementColumnName(TStreamerElement *elem, TSQLFile *f, Int_t indx = 0);
   static void AddStrBrackets(TString &s, const char *quote);

   enum ESQLTypes {
      kSqlObject = 10001,
      kSqlPointer = 10002,
      kSqlVersion = 10003,
      kSqlStreamerInfo = 10004,
      kSqlClassStreamer = 10005,
      kSqlElement = 10006,
      kSqlValue = 10007,
      kSqlArray = 10008,
      kSqlObjectData = 10009,
      kSqlCustomClass = 10010,
      kSqlCustomElement = 10011
   };

   enum ESQLColumns {
      kColUnknown = 0,
      kColSimple = 1,
      kColSimpleArray = 2,
      kColParent = 3,
      kColObject = 4,
      kColObjectArray = 5,
      kColNormObject = 6,
      kColNormObjectArray = 7,
      kColObjectPtr = 8,
      kColTString = 9,
      kColRawData = 10
   };

   enum ESQLIdType { kIdTable = 0, kIdRawTable = 1, kIdColumn = 2 };

   ClassDef(TSQLStructure, 1); // Table/structure description used internally by TBufferSQL.
};

// text constants, used in SQL I/O

namespace sqlio {

extern Long64_t atol64(const char *value);

extern const Int_t Ids_NullPtr;
extern const Int_t Ids_RootDir;
extern const Int_t Ids_TSQLFile;
extern const Int_t Ids_StreamerInfos;
extern const Int_t Ids_FirstKey;
extern const Int_t Ids_FirstObject;

extern const char *ObjectRef;
extern const char *ObjectRef_Arr;
extern const char *ObjectPtr;
extern const char *ObjectInst;
extern const char *Version;
extern const char *TObjectUniqueId;
extern const char *TObjectBits;
extern const char *TObjectProcessId;
extern const char *TStringValue;
extern const char *IndexSepar;
extern const char *RawSuffix;
extern const char *ParentSuffix;
extern const char *ObjectSuffix;
extern const char *PointerSuffix;
extern const char *StrSuffix;
extern const char *LongStrPrefix;

extern const char *Array;
extern const char *Bool;
extern const char *Char;
extern const char *Short;
extern const char *Int;
extern const char *Long;
extern const char *Long64;
extern const char *Float;
extern const char *Double;
extern const char *UChar;
extern const char *UShort;
extern const char *UInt;
extern const char *ULong;
extern const char *ULong64;
extern const char *CharStar;
extern const char *True;
extern const char *False;

extern const char *KeysTable;
extern const char *KeysTableIndex;
extern const char *KT_Name;
extern const char *KT_Title;
extern const char *KT_Datetime;
extern const char *KT_Cycle;
extern const char *KT_Class;

extern const char *DT_Create;
extern const char *DT_Modified;
extern const char *DT_UUID;

extern const char *ObjectsTable;
extern const char *ObjectsTableIndex;
extern const char *OT_Class;
extern const char *OT_Version;

extern const char *IdsTable;
extern const char *IdsTableIndex;
extern const char *IT_TableID;
extern const char *IT_SubID;
extern const char *IT_Type;
extern const char *IT_FullName;
extern const char *IT_SQLName;
extern const char *IT_Info;

extern const char *BT_Field;
extern const char *BT_Value;

extern const char *StringsTable;
extern const char *ST_Value;

extern const char *ConfigTable;
extern const char *CT_Field;
extern const char *CT_Value;

extern const char *cfg_Version;
extern const char *cfg_UseSufixes;
extern const char *cfg_ArrayLimit;
extern const char *cfg_TablesType;
extern const char *cfg_UseTransactions;
extern const char *cfg_UseIndexes;
extern const char *cfg_LockingMode;
extern const char *cfg_ModifyCounter;
}

#endif
