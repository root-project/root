// @(#)root/xml:$Name:  $:$Id: TXMLSetup.h,v 1.0 2004/01/28 22:31:11 brun Exp $
// Author: Sergey Linev  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TXMLSETUP_H
#define TXMLSETUP_H

#include "TXMLEngine.h"
#include "TObject.h"
#include "TString.h"

extern const char* NameSpaceBase;

extern const char* xmlNames_Root;
extern const char* xmlNames_Setup;
extern const char* xmlNames_Version;
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
extern const char* xmlNames_Class;
extern const char* xmlNames_Member;
extern const char* xmlNames_Item;
extern const char* xmlNames_Name;
extern const char* xmlNames_Type;
extern const char* xmlNames_Value;

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
      enum TXMLLayout { kSpecialized = 2,
                        kGeneralized = 3 };

      TXMLSetup();
      TXMLSetup(const char* opt);
      TXMLSetup(const TXMLSetup& src);
      virtual ~TXMLSetup();
      
      void StoreSetup(xmlNodePointer node);
      Bool_t ReadSetup(xmlNodePointer node);

      void PrintSetup();
      
      TXMLLayout GetXmlLayout() const { return fXmlLayout; }
      void SetXmlLayout(TXMLLayout layout) { fXmlLayout = layout; }
      
      Bool_t IsaSolidDataBlock() const { return fSolidDataBlock; }
      void SetSolidDataBlock(Bool_t iSolid = kTRUE) { fSolidDataBlock = iSolid; }
      
      Bool_t IsConvertBasicTypes() const { return fConvertBasicTypes; }
      void SetConvertBasicTypes(Bool_t iConvert = kTRUE) { fConvertBasicTypes = iConvert; }
      
      Bool_t IsUseDtd() const { return fUseDtd; }
      void SetUsedDtd(Bool_t use = kTRUE) { fUseDtd = use; }
      
      Bool_t IsUseNamespaces() const { return fUseNamespaces; }
      void SetUseNamespaces(Bool_t iUseNamespaces = kTRUE) { fUseNamespaces = iUseNamespaces; }
      
      const char* XmlConvertClassName(const TClass* cl);
      const char* XmlClassNameSpaceRef(const TClass* cl);   
      
      Int_t GetNextRefCounter() { return fRefCounter++; }
      
   protected:
   
      TClass* XmlDefineClass(const char* xmlClassName);
      const char* GetElItemName(TStreamerElement* el);
      const char* GetElName(TStreamerElement* el);
      
      Bool_t ReadSetupFromStr(const char* setupstr);
   
      TXMLLayout fXmlLayout;
      Bool_t fSolidDataBlock;
      Bool_t fConvertBasicTypes;
      Bool_t fUseDtd;
      Bool_t fUseNamespaces;
      
      Int_t  fRefCounter;
      
      TString fStrBuf;          //!
      TString fNameBuf;         //!
      
   ClassDef(TXMLSetup,1);
};

#endif

