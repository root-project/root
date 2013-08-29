// @(#)root/xml:$Id$
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


class TKeyXML;
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

   // Overwrite methods for directory I/O
   virtual Long64_t DirCreateEntry(TDirectory*);
   virtual Int_t    DirReadKeys(TDirectory*);
   virtual void     DirWriteKeys(TDirectory*);
   virtual void     DirWriteHeader(TDirectory*);

private:
   //let the compiler do the job. gcc complains when the following line is activated
   //TXMLFile(const TXMLFile &) {}            //Files cannot be copied
   void operator=(const TXMLFile &);

public:
   TXMLFile();
   TXMLFile(const char* filename, Option_t* option = "read", const char* title = "title", Int_t compression = 1);
   virtual ~TXMLFile();

   virtual void      Close(Option_t *option=""); // *MENU*
   virtual TKey*     CreateKey(TDirectory* mother, const TObject* obj, const char* name, Int_t bufsize);
   virtual TKey*     CreateKey(TDirectory* mother, const void* obj, const TClass* cl, const char* name, Int_t bufsize);
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
   Int_t             GetIOVersion() const { return fIOVersion; }          

   virtual Bool_t    IsOpen() const;

   virtual void      MakeFree(Long64_t, Long64_t) {}
   virtual void      MakeProject(const char *, const char* ="*", Option_t* ="new") {} // *MENU*
   virtual void      Map() {} // 
   virtual void      Paint(Option_t* ="") {}
   virtual void      Print(Option_t* ="") const {}
   virtual Bool_t    ReadBuffer(char*, Int_t) { return kFALSE; }
   virtual Bool_t    ReadBuffer(char*, Long64_t, Int_t) { return kFALSE; }
   virtual void      ReadFree() {}
   virtual Int_t     Recover() { return 0; }
   virtual Int_t     ReOpen(Option_t *mode);
   virtual void      Seek(Long64_t, ERelativeTo=kBeg) {}

   virtual void      SetEND(Long64_t) {}
   virtual Int_t     Sizeof() const { return 0; }

   virtual void      UseCache(Int_t = 10, Int_t = 0) {}
   virtual Bool_t    WriteBuffer(const char*, Int_t) { return kFALSE; }
   virtual Int_t     Write(const char* =0, Int_t=0, Int_t=0) { return 0; }
   virtual Int_t     Write(const char* =0, Int_t=0, Int_t=0) const { return 0; }
   virtual void      WriteFree() {}
   virtual void      WriteHeader() {}
   virtual void      WriteStreamerInfo();

   // XML specific functions
   
   virtual void      SetXmlLayout(EXMLLayout layout);
   virtual void      SetStoreStreamerInfos(Bool_t iConvert = kTRUE);
   virtual void      SetUsedDtd(Bool_t use = kTRUE);
   virtual void      SetUseNamespaces(Bool_t iUseNamespaces = kTRUE);
   
   Bool_t            AddXmlComment(const char* comment);
   Bool_t            AddXmlStyleSheet(const char* href, 
                                      const char* type = "text/css",
                                      const char* title = 0,
                                      int alternate = -1,
                                      const char* media = 0,
                                      const char* charset = 0);
   Bool_t            AddXmlLine(const char* line);                                   

   TXMLEngine*       XML() { return fXML; } 

protected:
   // functions to store streamer infos
   
   void              StoreStreamerElement(XMLNodePointer_t node, TStreamerElement* elem);
   void              ReadStreamerElement(XMLNodePointer_t node, TStreamerInfo* info);

   Bool_t            ReadFromFile();
   Int_t             ReadKeysList(TDirectory* dir, XMLNodePointer_t topnode);
   TKeyXML*          FindDirKey(TDirectory* dir);
   TDirectory*       FindKeyDir(TDirectory* mother, Long64_t keyid);
   void              CombineNodesTree(TDirectory* dir, XMLNodePointer_t topnode, Bool_t dolink);

   void              SaveToFile();

   static void       ProduceFileNames(const char* filename, TString& fname, TString& dtdname);

   XMLDocPointer_t   fDoc;                  //!

   XMLNodePointer_t  fStreamerInfoNode;     //!  pointer of node with streamer info data
   
   TXMLEngine*       fXML;                  //! object for interface with xml library
   
   Int_t             fIOVersion;            //! indicates format of ROOT xml file
   
   Long64_t          fKeyCounter;           //! counter of created keys, used for keys id
   
ClassDef(TXMLFile, 2)  //ROOT file in XML format
};



#endif

