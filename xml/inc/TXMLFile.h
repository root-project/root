// @(#)root/xml:$Name:  $:$Id: TXMLFile.h,v 1.6 2004/06/29 14:45:38 brun Exp $
// Author: Sergey Linev  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXMLFile
#define ROOT_TXMLFile

#ifndef ROOT_TXMLEngine
#include "TXMLEngine.h"
#endif
#ifndef ROOT_TFile
#include "TFile.h"
#endif
#ifndef ROOT_TXMLSetup
#include "TXMLSetup.h"
#endif


class TXMLKey;
class TList;
class TStreamerElement;
class TStreamerInfo;


class TXMLFile : public TFile, public TXMLSetup {
   protected:
       void             InitXmlFile(Bool_t create);
       // Interface to basic system I/O routines
       virtual Int_t    SysOpen(const char*, Int_t, UInt_t) { return 0; }
       virtual Int_t    SysClose(Int_t) { return 0; }
       virtual Int_t    SysRead(Int_t, void*, Int_t) { return 0; }
       virtual Int_t    SysWrite(Int_t, const void*, Int_t) { return 0; }
       virtual Long64_t SysSeek(Int_t, Long64_t, Int_t) { return 0; }
       virtual Int_t    SysStat(Int_t, Long_t*, Long64_t*, Long_t*, Long_t*) { return 0; }
       virtual Int_t    SysSync(Int_t) { return 0; }

   private:
      //let the compiler do the job. gcc complains when the following line is activated
      //TXMLFile(const TXMLFile &) {}            //Files cannot be copied
      void operator=(const TXMLFile &);

   public:
      TXMLFile();
      TXMLFile(const char* filename, Option_t* option = "read", const char* title = "title", Int_t compression = 1);
      virtual ~TXMLFile();

      virtual void      Close(Option_t *option=""); // *MENU*
      virtual void      DrawMap(const char* ="*",Option_t* ="") {} 
      virtual void      FillBuffer(char* &) {}
      virtual void      Flush() {}

      virtual Long64_t  GetEND() const { return 0; }
      virtual Int_t     GetErrno() const { return 0; }
      virtual void      ResetErrno() const {}

      virtual Int_t     GetNfree() const { return 0; }
      virtual Int_t     GetNbytesInfo() const {return 0; }
      virtual Int_t     GetNbytesFree() const {return 0; }
      virtual Long64_t  GetSeekFree() const {return 0; }
      virtual Long64_t  GetSeekInfo() const {return 0; }
      virtual Long64_t  GetSize() const { return 0; }
      virtual TList*    GetStreamerInfoList();

      virtual Bool_t    IsOpen() const;

      virtual void      MakeFree(Long64_t, Long64_t) {}
      virtual void      MakeProject(const char *, const char* ="*", Option_t* ="new") {} // *MENU*
      virtual void      Map() {} // 
      virtual void      Paint(Option_t* ="") {}
      virtual void      Print(Option_t* ="") const {}
      virtual Bool_t    ReadBuffer(char*, Int_t) { return kFALSE; }
      virtual void      ReadFree() {}
      virtual void      ReadStreamerInfo();
      virtual Int_t     Recover() { return 0; }
      virtual Int_t     ReOpen(Option_t *mode);
      virtual void      Seek(Long64_t, ERelativeTo=kBeg) {}

      virtual void      SetEND(Long64_t) {}
      virtual Int_t     Sizeof() const { return 0; }

      virtual void      UseCache(Int_t = 10, Int_t = TCache::kDfltPageSize) {}
      virtual Bool_t    WriteBuffer(const char*, Int_t) { return kFALSE; }
      virtual Int_t     Write(const char* =0, Int_t=0, Int_t=0) { return 0; }
      virtual Int_t     Write(const char* =0, Int_t=0, Int_t=0) const { return 0; }
      virtual void      WriteFree() {}
      virtual void      WriteHeader() {}
      virtual Int_t     WriteTObject(const TObject* obj, const char* name = 0, Option_t *option="");
      virtual Int_t     WriteObjectAny(const void *obj, const char *classname, const char *name, Option_t *option="");
      virtual Int_t     WriteObjectAny(const void* obj, const TClass* cl, const char* name, Option_t *option="");
      virtual void      WriteStreamerInfo();

      // XML specific functions
      
      virtual void      SetXmlLayout(EXMLLayout layout);
      virtual void      SetStoreStreamerInfos(Bool_t iConvert = kTRUE);
      virtual void      SetUsedDtd(Bool_t use = kTRUE);
      virtual void      SetUseNamespaces(Bool_t iUseNamespaces = kTRUE);

      TXMLEngine*       XML() { return fXML; } 

   protected:
      // functions to store streamer infos
      
      void              StoreStreamerElement(xmlNodePointer node, TStreamerElement* elem);
      void              ReadStreamerElement(xmlNodePointer node, TStreamerInfo* info);

      Bool_t            ReadFromFile();

      void              SaveToFile();

      static void       ProduceFileNames(const char* filename, TString& fname, TString& dtdname);

      xmlDocPointer     fDoc;                  //!

      xmlNodePointer    fStreamerInfoNode;     //!  pointer of node with streamer info data
      
      TXMLEngine*       fXML;                  //! object for interface with xml library
      
   ClassDef(TXMLFile,1)  //ROOT file in XML format
};



#endif

