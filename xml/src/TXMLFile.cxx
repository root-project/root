// @(#)root/xml:$Name:  $:$Id: TXMLFile.cxx,v 1.5 2004/06/04 16:28:31 brun Exp $
// Author: Sergey Linev, Rene Brun  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//________________________________________________________________________
//
// The main motivation for the XML  format is to facilitate the
// communication with other non ROOT applications. Currently
// writing and reading XML files is limited to ROOT applications.
// It is our intention to develop a simple reader independent
// of the ROOT libraries that could be used as an example for
// real applications.
// The XML format should be used only for small data volumes,
// typically histogram files, pictures, geometries, calibrations.
// The XML file is built in memory before being dumped to disk.
//
// Like for normal ROOT files, XML files use the same I/O mechanism
// exploiting the ROOT/CINT dictionary. Any class having a dictionary
// can be saved in XML format.
//
// This first implementation does not support subdirectories
// or Trees.
//
// The shared library libRXML.so may be loaded dynamically
// via gSystem->Load("libRXML"). This library is automatically
// loaded by the plugin manager as soon as a XML file is created
// via, eg
//   TFile::Open("file.xml","recreate");
// TFile::Open returns a TXMLFile object. When a XML file is open in write mode,
// one can use the normal TObject::Write to write an object in the file.
// Alternatively one can use the new functions TDirectory::WriteObject and
// TDirectory::WriteObjectAny to write a TObject* or any class not deriving
// from TObject.
//
// example of a session saving a histogram to a XML file
// =====================================================
//   TFile *f = TFile::Open("Example.xml","recreate");
//   TH1F *h = new TH1F("h","test",1000,-2,2);
//   h->FillRandom("gaus");
//   h->Write();
//   delete f;
//
// example of a session reading the histogram from the file
// ========================================================
//   TFile *f = TFile::Open("Example.xml");
//   TH1F *h = (TH1F*)f->Get("h");
//   h->Draw();
//
// A new option in the canvas "File" menu is available to save
// a TCanvas as a XML file. One can also do
//   canvas->Print("Example.xml");
//
// Configuring ROOT with the option "xml"
// ======================================
// The XML package uses the public XML parser and toolkit
// from Gnome. You should download the latest version 2-6.9
// from http://www.xmlsoft.org/downloads.html.
//
// On Unix systems dowload "libxml2-2.6.9.tar.gz" and create
// $XMLDIR pointing to the directory libxml2-2.6.9. in the $ROOTSYS
// directory, run the normal
//   ./configure
//   make
// Path to $XMLDIR/.libs should be included in $LD_LIBRARY_PATH variable
//
// On Windows, from the same web site download
//   libxml2-2.6.9.win32.zip
//   iconv-1.9.1.win32.zip
// unzip the two files, then copy the file iconv.h from the iconv/include file
// to $XMLDIR/include. Also copy iconv.dll, iconv.lib and iconv_a.lib
// from the iconv/lib directory to $XMLDIR/lib.
//
// You are now ready to configure ROOT with the XML option. do:
//  ./configure -enable-xml -enable-xxxxx, etc
//
// documentation
// =============
// The "xml" package is currently under development. A more complete
// documentation will be provided shortly in the classes reference guide.
// See classes TXMLFile, TXMLKey, TXMLBuffer, TXMLEngine, TXMLSetup
// and TXMLDtdGenerator.
// An example of XML file corresponding to the small example below
//can be found at http://root.cern.ch/root/Example.xml
//
//______________________________________________________________________________

#include "TXMLFile.h"

#include "TROOT.h"
#include "TSystem.h"
#include "TList.h"
#include "TBrowser.h"
#include "TObjArray.h"
#include "TXMLBuffer.h"
#include "TXMLKey.h"
#include "TObjArray.h"
#include "TXMLDtdGenerator.h"
#include "TArrayC.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TProcessID.h"
#include "TError.h"

#include "Riostream.h"

ClassImp(TXMLFile);

//______________________________________________________________________________
TXMLFile::TXMLFile() :
    TFile(),
    TXMLSetup(),
    fDoc(0),
    fDtdGener(0),
    fStreamerInfoNode(0),
    fXML(0)
{
  // default TXMLFile constructor   
}


//______________________________________________________________________________
TXMLFile::TXMLFile(const char* filename, Option_t* option, const char* title, Int_t compression) :
    TFile(),
    TXMLSetup(),
    fDoc(0),
    fDtdGener(0),
    fStreamerInfoNode(0)
{
   // Open or creates local XML file with name filename. 
   // It is recommended to specify filename as "<file>.xml". The suffix ".xml" 
   // will be used by object browsers to automatically identify the file as
   // a XML file. If the constructor fails in any way IsZombie() will
   // return true. Use IsOpen() to check if the file is (still) open.
   //
   // If option = NEW or CREATE   create a new file and open it for writing,
   //                             if the file already exists the file is
   //                             not opened.
   //           = RECREATE        create a new file, if the file already
   //                             exists it will be overwritten.
   //           = 2xoo            create a new file with specified xml settings
   //                             for more details see TXMLSetup class 
   //           = UPDATE          open an existing file for writing.
   //                             if no file exists, it is created.
   //           = READ            open an existing file for reading.
   //
   // For more details see comments for TFile::TFile() constructor
   //
   // For a moment TXMLFile does not support TTree objects and subdirectories
   
   fXML = new TXMLEngine();

   if (!gROOT)
      ::Fatal("TFile::TFile", "ROOT system not initialized");

   if (!strncmp(filename, "xml:", 4))
      filename += 4;

   gDirectory = 0;
   SetName(filename);
   SetTitle(title);
   TDirectory::Build();
   fFile = this;
   
   fD          = -1;
   fFile       = this;
   fFree       = 0;
   fVersion    = gROOT->GetVersionInt();  //ROOT version in integer format
   fUnits      = 4;
   fOption     = option;
   SetCompressionLevel(compression);
   fWritten    = 0;
   fSumBuffer  = 0;
   fSum2Buffer = 0;
   fBytesRead  = 0;
   fBytesWrite = 0;
   fClassIndex = 0;
   fSeekInfo   = 0;
   fNbytesInfo = 0;
   fCache      = 0;
   fProcessIDs = 0;
   fNProcessIDs= 0;
   
   fOption = option;
   fOption.ToUpper();
   
   if (fOption == "NEW") fOption = "CREATE";

   Bool_t create   = (fOption == "CREATE") ? kTRUE : kFALSE;
   Bool_t recreate = (fOption == "RECREATE") ? kTRUE : kFALSE;
   Bool_t update   = (fOption == "UPDATE") ? kTRUE : kFALSE;
   Bool_t read     = (fOption == "READ") ? kTRUE : kFALSE;
   Bool_t xmlsetup = IsValidXmlSetup(option);
   if (xmlsetup) recreate = kTRUE;

   if (!create && !recreate && !update && !read) {
      read    = kTRUE;
      fOption = "READ";
   }

   Bool_t devnull = kFALSE;
   const char *fname = 0;

   if (!filename || !strlen(filename)) {
      Error("TXMLFile", "file name is not specified");
      goto zombie;
   }
   
   // support dumping to /dev/null on UNIX
   if (!strcmp(filename, "/dev/null") &&
       !gSystem->AccessPathName(filename, kWritePermission)) {
      devnull  = kTRUE;
      create   = kTRUE;
      recreate = kFALSE;
      update   = kFALSE;
      read     = kFALSE;
      fOption  = "CREATE";
      SetBit(TFile::kDevNull);
   }
  
   gROOT->cd();

   fname = gSystem->ExpandPathName(filename);
   if (fname) {
      SetName(fname);
      delete [] (char*)fname;
      fname = GetName();
   } else {
      Error("TXMLFile", "error expanding path %s", filename);
      goto zombie;
   }

   if (recreate) {
      if (!gSystem->AccessPathName(fname, kFileExists))
         gSystem->Unlink(fname);
      recreate = kFALSE;
      create   = kTRUE;
      fOption  = "CREATE";
   }
   
   if (create && !devnull && !gSystem->AccessPathName(fname, kFileExists)) {
      Error("TXMLFile", "file %s already exists", fname);
      goto zombie;
   }

   if (update) {
      if (gSystem->AccessPathName(fname, kFileExists)) {
         update = kFALSE;
         create = kTRUE;
      }
      if (update && gSystem->AccessPathName(fname, kWritePermission)) {
         Error("TXMLFile", "no write permission, could not open file %s", fname);
         goto zombie;
      }
   }
   
   if (read) {
      if (gSystem->AccessPathName(fname, kFileExists)) {
         Error("TXMLFile", "file %s does not exist", fname);
         goto zombie;
      }
      if (gSystem->AccessPathName(fname, kReadPermission)) {
         Error("TXMLFile", "no read permission, could not open file %s", fname);
         goto zombie;
      }
   }

   fRealName = fname;
   
   if (create || update) 
     SetWritable(kTRUE);
   else
     SetWritable(kFALSE);

   if (create) 
     if (xmlsetup) ReadSetupFromStr(option);
              else ReadSetupFromStr(TXMLSetup::DefaultXmlSetup());
   
   InitXmlFile(create);
   
   return;
   
zombie:
   MakeZombie();
   gDirectory = gROOT;
}

//______________________________________________________________________________
void TXMLFile::InitXmlFile(Bool_t create) 
{
   // initialize xml file and correspondent structures
   // identical to TFile::Init() function    
    
   Int_t len = gROOT->GetListOfStreamerInfo()->GetSize()+1;
   if (len<5000) len = 5000;
   fClassIndex = new TArrayC(len);
   fClassIndex->Reset(0);
    
   if (create) {
      fDoc = fXML->NewDoc(0);
      xmlNodePointer fRootNode = fXML->NewChild(0, 0, xmlNames_Root, 0);
      fXML->DocSetRootElement(fDoc, fRootNode);
   } else {
      ReadFromFile();
   } 
   
   if (IsWritable())
     fDtdGener = new TXMLDtdGenerator(*this);

   gROOT->GetListOfFiles()->Add(this);
   cd();

   fNProcessIDs = 0;
   TKey* key = 0;
   TIter iter(fKeys);
   while ((key = (TKey*)iter())!=0) {
      if (!strcmp(key->GetClassName(),"TProcessID")) fNProcessIDs++;
   }

   fProcessIDs = new TObjArray(fNProcessIDs+1);
}

//______________________________________________________________________________
void TXMLFile::Close(Option_t *option) 
{
   // Close a XML file
   // For more comments see TFile::Close() function
    
   if (!IsOpen()) return;  
   
   TString opt = option;
   if (opt.Length()>0)
     opt.ToLower();
   
   if (IsWritable()) SaveToFile();
   
   fWritable = kFALSE;
   
   if (fDoc) {
     fXML->FreeDoc(fDoc);
     fDoc = 0;
   }

   if (fDtdGener) {
     delete fDtdGener;
     fDtdGener = 0;
   }
   
   if (fClassIndex) {
      delete fClassIndex;
      fClassIndex = 0;
   }
   

   if (fStreamerInfoNode) {
     fXML->FreeNode(fStreamerInfoNode);
     fStreamerInfoNode = 0;
   }

   TDirectory *cursav = gDirectory;
   cd();

   if (cursav == this || cursav->GetFile() == this) {
      cursav = 0;
   }

   // Delete all supported directories structures from memory
   TDirectory::Close();
   cd();      // Close() sets gFile = 0

   if (cursav)
      cursav->cd();
   else {
      gFile      = 0;
      gDirectory = gROOT;
   }

   //delete the TProcessIDs
   TList pidDeleted;
   TIter next(fProcessIDs);
   TProcessID *pid;
   while ((pid = (TProcessID*)next())) {
      if (!pid->DecrementCount()) {
         if (pid != TProcessID::GetSessionProcessID()) pidDeleted.Add(pid);
      } else if(opt.Contains("r")) {
         pid->Clear();
      }
   }
   pidDeleted.Delete();

   gROOT->GetListOfFiles()->Remove(this);
}

//______________________________________________________________________________
TXMLFile::~TXMLFile() 
{
// destructor of TXMLFile object

  Close();
  
  if (fXML!=0) {
     delete fXML;
     fXML = 0;
  }
}

//______________________________________________________________________________
void TXMLFile::operator=(const TXMLFile &) 
{
// make private to exclude copy operator
}

//______________________________________________________________________________
Bool_t TXMLFile::IsOpen() const 
{
// return kTRUE if file is opened and can be accessed  
    
   return fDoc != 0;
}

//______________________________________________________________________________
Int_t TXMLFile::ReOpen(Option_t* mode) 
{
   // Reopen a file with a different access mode, like from READ to
   // See TFile::Open() for details
    
   cd();
   
   TString opt = mode;
   opt.ToUpper();

   if (opt != "READ" && opt != "UPDATE") {
      Error("ReOpen", "mode must be either READ or UPDATE, not %s", opt.Data());
      return 1;
   }
   
   if (opt == fOption || (opt == "UPDATE" && fOption == "CREATE"))
      return 1;

   if (opt == "READ") {
      // switch to READ mode
      
      if (IsOpen() && IsWritable()) 
        SaveToFile();
      fOption = opt;    
   
      SetWritable(kFALSE);

   } else {
      fOption = opt;    
    
      SetWritable(kTRUE); 
   }

   return 0;
}

//______________________________________________________________________________
Int_t TXMLFile::WriteTObject(const TObject* obj, const char* name, Option_t* option) 
{
  // write object, derived from TObject class, to file  
    
   if ((name==0) || (*name==0))
     if (obj!=0) name = obj->GetName();
    
   return WriteObjectAny(obj, obj ? obj->IsA() : 0, name, option);
}

//______________________________________________________________________________
Int_t TXMLFile::WriteObjectAny(const void *obj, const char *classname, const char *name, Option_t *option)
{
   // Write object of class classname in this directory
   // obj may not derive from TObject
   // see TDirectory::WriteObject for comments
   
   TClass *cl = gROOT->GetClass(classname);
   if (!cl) {
      Error("WriteObjectAny","Unknown class: %s",classname);
      return 0;
   }
   return WriteObjectAny(obj,cl,name,option);
}   

//______________________________________________________________________________
Int_t TXMLFile::WriteObjectAny(const void* obj, const TClass* cl, const char* name, Option_t* option) 
{
  // write object of any class with disctionary to xml file  
  // object is tranformed to xml structure, which then kept in TXMLKey
  // Data will be stored to file only when Close() or ReOpen() or destructor is called
  // For more details see TDirectory::WriteObjectAny() function
    
   if (!IsWritable()) {
      if (!TestBit(TFile::kWriteError)) {
         // Do not print the error if the file already had a SysError.
         Error("WriteObject","file %s is not writable", GetName());
      }
      return 0;
   }

   if (!obj || !cl) return 0;
   
   if (cl->InheritsFrom("TTree")) {
      Warning("WriteObject","TTree class not (yet) fully supported"); 
   }
    
   TKey *key = 0, *oldkey=0;

   TString opt = option;
   opt.ToLower();

   const char *oname;
   if (name && *name)
      oname = name;
   else
      oname = cl->GetName();

   // Remove trailing blanks in object name
   Int_t nch = strlen(oname);
   char *newName = 0;
   if (oname[nch-1] == ' ') {
      newName = new char[nch+1];
      strcpy(newName,oname);
      for (Int_t i=0;i<nch;i++) {
         if (newName[nch-i-1] != ' ') break;
         newName[nch-i-1] = 0;
      }
      oname = newName;
   }

   if (opt.Contains("overwrite")) {
      //One must use GetKey. FindObject would return the lowest cycle of the key!
      //key = (TKey*)gDirectory->GetListOfKeys()->FindObject(oname);
      key = (TKey*)GetKey(oname);
      if (key) {
         key->Delete();
         delete key;
      }
   }
   if (opt.Contains("writedelete")) {
      oldkey = (TKey*)GetKey(oname);
   }
   
   key = new TXMLKey(this, obj, cl, oname);
   
   if (newName) delete [] newName;

   if (!key->GetSeekKey()) {
      fKeys->Remove(key);
      delete key;
      return 0;
   }
   
   if (oldkey) {
      oldkey->Delete();
      delete oldkey;
   }

   return 0;
}


//______________________________________________________________________________
void TXMLFile::ProduceFileNames(const char* filename, TString& fname, TString& dtdname) 
{
   // function produces pair of xml and dtd file names
  
   fname = filename;
   dtdname = filename;

   Bool_t hasxmlext = kFALSE;

   if (fname.Length()>4) {
      TString last = fname(fname.Length()-4,4);
      last.ToLower();
      hasxmlext = (last==".xml");
   }

   if (hasxmlext) {
      dtdname.Replace(dtdname.Length()-4,4,".dtd");
   } else {
      fname+=".xml";
      dtdname+=".dtd";
   }
}

//______________________________________________________________________________
void TXMLFile::SaveToFile() 
{
   // Saves xml structures to file
   // xml elements are kept in list of TXMLKey objects 
   // When saving, all this elements are linked to root xml node
   // In the end StreamerInfo structures are added
   // After xml document is saved, all nodes will be unlinked from root node 
   // and kept in memory. 
   // Only Close() or destructor relase memory, used by xml structures
    
   if (fDoc==0) return;

   if (gDebug>1)
     cout << "TXMLFile::SaveToFile() " << fRealName << endl;

   xmlNodePointer fRootNode = fXML->DocGetRootElement(fDoc);
   
   fXML->FreeAttr(fRootNode, xmlNames_Setup);
   fXML->NewAttr(fRootNode, 0, xmlNames_Setup, GetSetupAsString());
   
   fXML->FreeAttr(fRootNode, xmlNames_Ref);
   fXML->NewAttr(fRootNode, 0, xmlNames_Ref, xmlNames_Null);

   TString fname, dtdname;
   ProduceFileNames(fRealName, fname, dtdname);

   TIter iter(GetListOfKeys());
   TXMLKey* key = 0;

   while ((key=(TXMLKey*)iter()) !=0)
      fXML->AddChild(fRootNode, key->KeyNode());

   WriteStreamerInfo();
   
   if (fStreamerInfoNode)
     fXML->AddChild(fRootNode, fStreamerInfoNode);

   if (fDtdGener && IsUseDtd())
     fXML->AssignDtd(fDoc, dtdname, xmlNames_Root);

   Int_t layout = GetCompressionLevel()>5 ? 0 : 1;

   fXML->SaveDoc(fDoc, fname, layout);

   if (fDtdGener && IsUseDtd()) {
       fDtdGener->Produce(dtdname);
   }

   iter.Reset();
   while ((key=(TXMLKey*)iter()) !=0)
      fXML->UnlinkNode(key->KeyNode());
      
   if (fStreamerInfoNode)
     fXML->UnlinkNode(fStreamerInfoNode);
}

//______________________________________________________________________________
Bool_t TXMLFile::ReadFromFile() 
{
   // read document from file
   // Now full content of docuument reads into the memory
   // Then document decomposed to separate keys and streamer info structures
   // All inrelevant data will be cleaned
   
   fDoc = fXML->ParseFile(fRealName);
   if (fDoc==0) return kFALSE;
   
   xmlNodePointer fRootNode = fXML->DocGetRootElement(fDoc);

   if (fRootNode==0) {
      fXML->FreeDoc(fDoc);
      fDoc=0;
      return kFALSE;
   }

   ReadSetupFromStr(fXML->GetAttr(fRootNode, xmlNames_Setup));
   
   fStreamerInfoNode = fXML->GetChild(fRootNode);
   fXML->SkipEmpty(fStreamerInfoNode);
   while (fStreamerInfoNode!=0) {
      if (strcmp("StreamerInfos",fXML->GetNodeName(fStreamerInfoNode))==0) break;
      fXML->ShiftToNext(fStreamerInfoNode);
   }
   fXML->UnlinkNode(fStreamerInfoNode);
   
   ReadStreamerInfo();

   if (IsUseDtd())
     if (!fXML->ValidateDocument(fDoc, gDebug>0)) {
        fXML->FreeDoc(fDoc);
        fDoc=0;
        return kFALSE;
     }

   xmlNodePointer keynode = fXML->GetChild(fRootNode);
   fXML->SkipEmpty(keynode);
   while (keynode!=0) {
      xmlNodePointer next = fXML->GetNext(keynode);
      
      if (strcmp(xmlNames_Xmlkey, fXML->GetNodeName(keynode))==0) {
         fXML->UnlinkNode(keynode);

         TXMLKey* key = new TXMLKey(this, keynode);
         AppendKey(key);
         
         if (gDebug>2)
           cout << "Adding key " << fXML->GetNodeName(keynode) << "   name = " << key->GetName() << endl;
      }

      keynode = next;
      fXML->SkipEmpty(keynode);
   }
   
   fXML->CleanNode(fRootNode);
   
   return kTRUE;
}

//______________________________________________________________________________
void TXMLFile::WriteStreamerInfo() 
{
// convert all TStreamerInfo, used in file, to xml format


   if (fStreamerInfoNode) {
     fXML->FreeNode(fStreamerInfoNode);
     fStreamerInfoNode = 0;
   }
   
   if (!IsStoreStreamerInfos()) return;
    
   TObjArray list;

   TIter iter(gROOT->GetListOfStreamerInfo());

   TStreamerInfo* info = 0;

   while ((info = (TStreamerInfo*) iter()) !=0 ) {
      Int_t uid = info->GetNumber();
      if (fClassIndex->fArray[uid])
        list.Add(info);
   }

   if (list.GetSize()==0) return;

   fStreamerInfoNode = fXML->NewChild(0, 0, "StreamerInfos");
   for (int n=0;n<=list.GetLast();n++) {
      TStreamerInfo* info  = (TStreamerInfo*) list.At(n);

      xmlNodePointer infonode = fXML->NewChild(fStreamerInfoNode, 0, "TStreamerInfo");

      fXML->NewAttr(infonode, 0, "name", info->GetName());
      fXML->NewAttr(infonode, 0, "title", info->GetTitle());

      fXML->NewIntAttr(infonode, "v", info->IsA()->GetClassVersion());
      fXML->NewIntAttr(infonode, "classversion", info->GetClassVersion());
      fXML->NewIntAttr(infonode, "checksum", info->GetCheckSum());

      TIter iter(info->GetElements());
      TStreamerElement* elem=0;
      while ((elem= (TStreamerElement*) iter()) != 0) {
         StoreStreamerElement(infonode, elem);
      }
   }
}

//______________________________________________________________________________
TList* TXMLFile::GetStreamerInfoList() 
{
// Read streamerinfo structures from xml format and provide them in the list
// It is user responsibility to destroy this list
    
   if (fStreamerInfoNode==0) return 0;
    
   TList* list = new TList();
   
   xmlNodePointer sinfonode = fXML->GetChild(fStreamerInfoNode);
   fXML->SkipEmpty(sinfonode);

   while (sinfonode!=0) {
     if (strcmp("TStreamerInfo",fXML->GetNodeName(sinfonode))==0) {
        TString fname = fXML->GetAttr(sinfonode,"name");
        TString ftitle = fXML->GetAttr(sinfonode,"title");

        TStreamerInfo* info = new TStreamerInfo(gROOT->GetClass(fname), ftitle);

        list->Add(info);

        Int_t clversion = AtoI(fXML->GetAttr(sinfonode,"classversion"));
        info->SetClassVersion(clversion);
        Int_t checksum = AtoI(fXML->GetAttr(sinfonode,"checksum"));
        info->SetCheckSum(checksum);

        xmlNodePointer node = fXML->GetChild(sinfonode);
        fXML->SkipEmpty(node);
        while (node!=0) {
           ReadStreamerElement(node, info);
           fXML->ShiftToNext(node);
        }
     }
     fXML->ShiftToNext(sinfonode);
   }
   
   list->SetOwner();
   
   return list;
}

//______________________________________________________________________________
void TXMLFile::ReadStreamerInfo() 
{
// Read the list of StreamerInfo from this file
// The corresponding TClass objects are updated.

   TList* list = GetStreamerInfoList();
   if (list==0) return;
   
   list->SetOwner(kFALSE);
   
   if (gDebug>1)
     cout << "Loop over TStreamerInfo classes num = " << list->GetSize() << endl;

   // loop on all TStreamerInfo classes
   TStreamerInfo *info;
   TIter next(list);
   while ((info = (TStreamerInfo*)next())) {
      if (info->IsA() != TStreamerInfo::Class()) {
         Warning("ReadStreamerInfo","%s: not a TStreamerInfo object", GetName());
         continue;
      }
      info->BuildCheck();
      Int_t uid = info->GetNumber();
      Int_t asize = fClassIndex->GetSize();
      if (uid >= asize && uid <100000) fClassIndex->Set(2*asize);
      if (uid >= 0 && uid < fClassIndex->GetSize()) fClassIndex->fArray[uid] = 1;
      else {
         printf("ReadStreamerInfo, class:%s, illegal uid=%d\n",info->GetName(),uid);
      }
      if (gDebug > 0)
         printf(" -class: %s version: %d info read at slot %d\n",info->GetName(), info->GetClassVersion(),uid);
   }
   fClassIndex->fArray[0] = 0;

   list->Clear(); //this will delete all TStreamerInfo objects with kCanDelete bit set
   delete list;
}


//______________________________________________________________________________
void TXMLFile::StoreStreamerElement(xmlNodePointer infonode, TStreamerElement* elem) 
{
// store data of single TStreamerElement in streamer node    
    
   TClass* cl = elem->IsA();

   xmlNodePointer node = fXML->NewChild(infonode, 0, cl->GetName());

   char sbuf[100], namebuf[100];

   fXML->NewAttr(node,0,"name",elem->GetName());
   if (strlen(elem->GetTitle())>0)
     fXML->NewAttr(node,0,"title",elem->GetTitle());

   fXML->NewIntAttr(node, "v", cl->GetClassVersion());

   fXML->NewIntAttr(node, "type", elem->GetType());

   if (strlen(elem->GetTypeName())>0)
     fXML->NewAttr(node,0,"typename", elem->GetTypeName());

   fXML->NewIntAttr(node, "size", elem->GetSize());

   if (elem->GetArrayDim()>0) {
      fXML->NewIntAttr(node, "numdim", elem->GetArrayDim());

      for (int ndim=0;ndim<elem->GetArrayDim();ndim++) {
         sprintf(namebuf, "dim%d", ndim);
         fXML->NewIntAttr(node, namebuf, elem->GetMaxIndex(ndim));
      }
   }

   if (cl == TStreamerBase::Class()) {
      TStreamerBase* base = (TStreamerBase*) elem;
      sprintf(sbuf, "%d", base->GetBaseVersion());
      fXML->NewAttr(node,0, "baseversion", sbuf);
   } else
   if (cl == TStreamerBasicPointer::Class()) {
     TStreamerBasicPointer* bptr = (TStreamerBasicPointer*) elem;
     fXML->NewIntAttr(node, "countversion", bptr->GetCountVersion());
     fXML->NewAttr(node, 0, "countname", bptr->GetCountName());
     fXML->NewAttr(node, 0, "countclass", bptr->GetCountClass());
   } else
   if (cl == TStreamerLoop::Class()) {
     TStreamerLoop* loop = (TStreamerLoop*) elem;
     fXML->NewIntAttr(node, "countversion", loop->GetCountVersion());
     fXML->NewAttr(node, 0, "countname", loop->GetCountName());
     fXML->NewAttr(node, 0, "countclass", loop->GetCountClass());
   } else
   if ((cl == TStreamerSTL::Class()) || (cl == TStreamerSTLstring::Class())) {
     TStreamerSTL* stl = (TStreamerSTL*) elem;

     fXML->NewIntAttr(node, "STLtype", stl->GetSTLtype());
     fXML->NewIntAttr(node, "Ctype", stl->GetCtype());
   }
}

//______________________________________________________________________________
void TXMLFile::ReadStreamerElement(xmlNodePointer node, TStreamerInfo* info) 
{
  // read and reconstruct single TStreamerElement from xml node   
    
   TClass* cl = gROOT->GetClass(fXML->GetNodeName(node));
   if ((cl==0) || !cl->InheritsFrom(TStreamerElement::Class())) return;

//   Int_t elemversion = fXML->GetAttr(node,"v");

   TStreamerElement* elem = (TStreamerElement*) cl->New();
   
   elem->SetName(fXML->GetAttr(node,"name"));
   elem->SetTitle(fXML->GetAttr(node,"title"));
   elem->SetType(fXML->GetIntAttr(node,"type"));
   elem->SetTypeName(fXML->GetAttr(node,"typename"));
   elem->SetSize(fXML->GetIntAttr(node,"size"));

   if (cl == TStreamerBase::Class()) {
      int basever = fXML->GetIntAttr(node,"baseversion");
      ((TStreamerBase*) elem)->SetBaseVersion(basever);
   } else
   if (cl == TStreamerBasicPointer::Class()) {
     TString countname = fXML->GetAttr(node,"countname");
     TString countclass = fXML->GetAttr(node,"countclass");
     Int_t countversion = fXML->GetIntAttr(node, "countversion");

     ((TStreamerBasicPointer*)elem)->SetCountVersion(countversion);
     ((TStreamerBasicPointer*)elem)->SetCountName(countname);
     ((TStreamerBasicPointer*)elem)->SetCountClass(countclass);
   } else
   if (cl == TStreamerLoop::Class()) {
     TString countname = fXML->GetAttr(node,"countname");
     TString countclass = fXML->GetAttr(node,"countclass");
     Int_t countversion = fXML->GetIntAttr(node,"countversion");
     ((TStreamerLoop*)elem)->SetCountVersion(countversion);
     ((TStreamerLoop*)elem)->SetCountName(countname);
     ((TStreamerLoop*)elem)->SetCountClass(countclass);
   } else
   if ((cl == TStreamerSTL::Class()) || (cl == TStreamerSTLstring::Class()))  {
     int fSTLtype = fXML->GetIntAttr(node,"STLtype");
     int fCtype = fXML->GetIntAttr(node,"Ctype");
     ((TStreamerSTL*)elem)->SetSTLtype(fSTLtype);
     ((TStreamerSTL*)elem)->SetCtype(fCtype);
   }

   char namebuf[100];

   if (fXML->HasAttr(node, "numdim") && (elem!=0)) {
     int numdim = fXML->GetIntAttr(node,"numdim");
     elem->SetArrayDim(numdim);
     for (int ndim=0;ndim<numdim;ndim++) {
         sprintf(namebuf, "dim%d", ndim);
         int maxi = fXML->GetIntAttr(node, namebuf);
         elem->SetMaxIndex(ndim, maxi);
     }
   }

   info->GetElements()->Add(elem);
}

//______________________________________________________________________________
void TXMLFile::SetXmlLayout(EXMLLayout layout) 
{
// Change layout of objects in xml file
// Can be changed only for newly created file.
//
// Currently there are two supported layouts:
//
// TXMLSetup::kSpecialized = 2 
//    This is default layout of the file, when xml nodes names class names and data member 
//    names are used. For instance: 
//          <TAttLine version="1">
//            <fLineColor v="1"/>
//            <fLineStyle v="1"/>
//            <fLineWidth v="1"/>
//          </TAttLine>
//
// TXMLSetup::kGeneralized = 3
//    For this layout all nodes name does not depend from class definitions. 
//    The same class looks like
//          <Class name="TAttLine" version="1">
//            <Member name="fLineColor" v="1"/>
//            <Member name="fLineStyle" v="1"/>
//            <Member name="fLineWidth" v="1"/>
//          </Member>
//
      
   if (IsWritable() && (GetListOfKeys()->GetSize()==0))
     TXMLSetup::SetXmlLayout(layout);
}

//______________________________________________________________________________
void TXMLFile::SetStoreStreamerInfos(Bool_t iConvert) 
{
// If true, all correspondent to file TStreamerInfo objects will be stored in file
// this allows to apply schema avolution later for this file
// may be usefull, when file used outside ROOT and TStreamerInfo objects does not required
// Can be changed only for newly created file.
    
   if (IsWritable() &&  (GetListOfKeys()->GetSize()==0))
      TXMLSetup::SetStoreStreamerInfos(iConvert);
}

//______________________________________________________________________________
void TXMLFile::SetUsedDtd(Bool_t use) 
{
// Specify usage of DTD for this file.
// Currently this option not avaliable (always false).
// Can be changed only for newly created file.
    
   if (IsWritable() &&  (GetListOfKeys()->GetSize()==0))
      TXMLSetup::SetUsedDtd(use);
}

//______________________________________________________________________________
void TXMLFile::SetUseNamespaces(Bool_t iUseNamespaces) 
{
// Specifiy usage of namespaces in xml file
// In current implementation every instrumented class in file gets its unique namespace,
// which is equal to name of class and refer to root documentation page like
// <TAttPad xmlns:TAttPad="http://root.cern.ch/root/htmldoc/TAttPad.html" version="3">
// And xml node for class member gets its name as combination of class name and member name
//            <TAttPad:fLeftMargin v="0.100000"/>
//            <TAttPad:fRightMargin v="0.100000"/>
//            <TAttPad:fBottomMargin v="0.100000"/>
//            and so on
// Usage of namespace increase size of xml file, but makes file more readable 
// and allows to produce DTD in the case, when in several classes data member has same name
// Can be changed only for newly created file.

   if (IsWritable() && (GetListOfKeys()->GetSize()==0))
       TXMLSetup::SetUseNamespaces(iUseNamespaces);
}

