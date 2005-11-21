#ifndef ROOT_TSQLStructure
#define ROOT_TSQLStructure

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
      
   protected:   
   
      TString     fName;             //!  name of the table column
      TString     fType;             //!  type of the table column
      TString     fValue;            //!  value of the table column
      Bool_t      fNumeric;          //!  for numeric quotes (double quotes) are not required
      
   ClassDef(TSQLColumnData, 1) // structure with column values and types
};

//______________________________________________________________________

class TSQLStructure : public TObject {
   public:
      TSQLStructure();
      virtual ~TSQLStructure();
      
      TSQLStructure*   GetParent() const { return fParent; }
      void             SetParent(TSQLStructure* p) { fParent = p; }
      Int_t            NumChilds() const;
      TSQLStructure*   GetChild(Int_t n) const;
      void             RemoveChild(Int_t n);

      void             SetType(Int_t typ) { fType = typ; }
      Int_t            GetType() const { return fType; }
      
      // this part requried for writing to SQL tables
      
      void             SetObjectRef(Int_t refid, const TClass* cl);
      void             SetObjectPointer(Int_t ptrid);
      void             SetVersion(const TClass* cl, Int_t version = -100);
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
      Bool_t           IsOnlyChild() const;

      // this is part specially for reading of sql tables
      
      Int_t            DefineObjectId();
      
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
      static Int_t     DefineElementColumnType(TStreamerElement* elem);
      
   protected:   
   
      TString          MakeArrayIndex(TStreamerElement* elem, Int_t n);

      Int_t            FindMaxRef();
      void             PerformConversion(TSqlRegistry* reg, TObjArray* blobs, const char* topname);
      Bool_t           StoreObject(TSqlRegistry* reg, const char* objid, TClass* cl, Bool_t registerobj = kTRUE);
      Bool_t           StoreObjectInNormalForm(TSqlRegistry* reg);
      Bool_t           StoreClassInNormalForm(TSqlRegistry* reg);
      Bool_t           StoreElementInNormalForm(TSqlRegistry* reg, TObjArray* columns);
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
      enum ESQLTypes {
         kSqlObject       = 10001,
         kSqlPointer      = 10002,
         kSqlVersion      = 10003,
         kSqlStreamerInfo = 10004,
         kSqlElement      = 10005,
         kSqlValue        = 10006,
         kSqlArray        = 10007,
         kSqlObjectData   = 10008
      };
      enum ESQLColimns {
         kColUnknown      = 0,
         kColSimple       = 1,
         kColSimpleArray  = 2,
         kColParent       = 3,
         kColObject       = 4,
         kColObjectPtr    = 5,
         kColTString      = 6,
         kColRawData      = 7
      };   
      
   ClassDef(TSQLStructure, 1); 
};

namespace sqlio {

    extern const Int_t Ids_NullPtr;
    extern const Int_t Ids_RootDir;
    extern const Int_t Ids_StreamerInfos;
    extern const Int_t Ids_FirstKey;
    extern const Int_t Ids_FirstObject;
    
    extern const char* ObjectRef;
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
    
};

#endif
