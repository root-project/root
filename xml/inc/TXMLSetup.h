// @(#)root/xml:$Name:  $:$Id: TXMLSetup.h,v 1.4 2004/05/14 14:30:46 brun Exp $
// Author: Sergey Linev  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXMLSetup
#define ROOT_TXMLSetup

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

extern const char* xmlNames_Root;
extern const char* xmlNames_Setup;
extern const char* xmlNames_ClassVersion;
extern const char* xmlNames_OnlyVersion;
extern const char* xmlNames_Ptr;
extern const char* xmlNames_Ref;
extern const char* xmlNames_Null;
extern const char* xmlNames_IdBase;
extern const char* xmlNames_Size;
extern const char* xmlNames_Xmlobject;
extern const char* xmlNames_Xmlkey;
extern const char* xmlNames_Cycle;
extern const char* xmlNames_XmlBlock;
extern const char* xmlNames_Zip;
extern const char* xmlNames_Object;
extern const char* xmlNames_ObjClass;
extern const char* xmlNames_Class;
extern const char* xmlNames_Member;
extern const char* xmlNames_Item;
extern const char* xmlNames_Name;
extern const char* xmlNames_Type;
extern const char* xmlNames_Value;
extern const char* xmlNames_v;
extern const char* xmlNames_cnt;
extern const char* xmlNames_true;
extern const char* xmlNames_false;
extern const char* xmlNames_SInfos;

extern const char* xmlNames_Array;
extern const char* xmlNames_Bool;
extern const char* xmlNames_Char;
extern const char* xmlNames_Short;
extern const char* xmlNames_Int;
extern const char* xmlNames_Long;
extern const char* xmlNames_Long64;
extern const char* xmlNames_Float;
extern const char* xmlNames_Double;
extern const char* xmlNames_UChar;
extern const char* xmlNames_UShort;
extern const char* xmlNames_UInt;
extern const char* xmlNames_ULong;
extern const char* xmlNames_ULong64;
extern const char* xmlNames_String;
extern const char* xmlNames_CharStar;

class TStreamerElement;

class TXMLSetup {
   public:
      enum EXMLLayout { kSpecialized = 2,
                        kGeneralized = 3 };

      TXMLSetup();
      TXMLSetup(const char* opt);
      TXMLSetup(const TXMLSetup& src);
      virtual ~TXMLSetup();

      TString        GetSetupAsString();

      void           PrintSetup();

      EXMLLayout     GetXmlLayout() const { return fXmlLayout; }
      Bool_t         IsStoreStreamerInfos() const { return fStoreStreamerInfos; }
      Bool_t         IsUseDtd() const { return fUseDtd; }
      Bool_t         IsUseNamespaces() const { return fUseNamespaces; }
      
      virtual void   SetXmlLayout(EXMLLayout layout) { fXmlLayout = layout; }
      virtual void   SetStoreStreamerInfos(Bool_t iConvert = kTRUE) { fStoreStreamerInfos = iConvert; }
      virtual void   SetUsedDtd(Bool_t use = kTRUE) { fUseDtd = use; }
      virtual void   SetUseNamespaces(Bool_t iUseNamespaces = kTRUE) { fUseNamespaces = iUseNamespaces; }

      const char*    XmlConvertClassName(const TClass* cl);
      const char*    XmlClassNameSpaceRef(const TClass* cl);

      Int_t          GetNextRefCounter() { return fRefCounter++; }
      
      static TString DefaultXmlSetup();
      static void    SetNameSpaceBase(const char* namespacebase);

   protected:

      TClass*        XmlDefineClass(const char* xmlClassName);
      const char*    GetElItemName(TStreamerElement* el);
      const char*    GetElName(TStreamerElement* el);

      Bool_t         IsValidXmlSetup(const char* setupstr);
      Bool_t         ReadSetupFromStr(const char* setupstr);

      Int_t          AtoI(const char* sbuf, Int_t def = 0, const char* errinfo = 0);


      EXMLLayout     fXmlLayout;
      Bool_t         fStoreStreamerInfos;
      Bool_t         fUseDtd;
      Bool_t         fUseNamespaces;

      Int_t          fRefCounter;      //!  counter , used to build id of xml references

      TString        fStrBuf;          //!  buffer, used in XmlDefineClass() function
      
      static TString fNameSpaceBase;   

   ClassDef(TXMLSetup,1) //settings to be stored in XML files
};

#endif

