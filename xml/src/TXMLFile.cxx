// @(#)root/xml:$Name:  $:$Id: TXMLFile.cxx,v 1.2 2004/05/10 23:50:27 rdm Exp $
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

#include "Riostream.h"

ClassImp(TXMLFile);

//______________________________________________________________________________
TXMLFile::TXMLFile() :
    TFile(),
    TXMLSetup(),
    fDoc(0),
    fDtdGener(0),
    fStreamerInfoNode(0) {
}


//______________________________________________________________________________
TXMLFile::TXMLFile(const char* filename, Option_t* option, const char* title, Int_t compression) :
    TFile(),
    TXMLSetup(),
    fDoc(0),
    fDtdGener(0),
    fStreamerInfoNode(0) {

   SetName(filename);
   SetTitle(title);
   SetCompressionLevel(compression);
   fOption = option;
   fOption.ToLower();
   
   fRealName = filename;
//   if (strncmp(filename,"xml:",4)==0)
//      fRealName.Remove(0,4);

   if (strlen(option) == 0) fOption = "read";
   if (fOption=="new") fOption="create";
   if (fOption=="recreate") fOption="create";
   if (fOption=="create") fOption = "2xxoo";

   gDirectory = 0;
   TDirectory::Build();
   gROOT->cd();


   int len = gROOT->GetListOfStreamerInfo()->GetSize()+1;
   if (len<5000) len = 5000;
   fClassIndex = new TArrayC(len);
   fClassIndex->Reset(0);

   fWritable = fOption != "read";
   if (!IsWritable()) {
      ReadFromFile();
   } else {
      ReadSetupFromStr(fOption);
      
      fDoc = gXML->NewDoc(0);
      xmlNodePointer fRootNode = gXML->NewChild(0, 0, xmlNames_Root, 0);
      gXML->DocSetRootElement(fDoc, fRootNode);
      StoreSetup(fRootNode);
      gXML->NewProp(fRootNode, 0, xmlNames_Ref, xmlNames_Null);
      fDtdGener = new TXMLDtdGenerator(*this);
   }

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
TXMLFile::~TXMLFile() {
   if ((fDoc!=0) && IsWritable()) SaveToFile();
   if (fDtdGener) delete fDtdGener;
   if (fDoc) gXML->FreeDoc(fDoc);
   
   if (fStreamerInfoNode) 
     gXML->FreeNode(fStreamerInfoNode);

   gROOT->GetListOfFiles()->Remove(this);

   if (gFile==this)
     gROOT->cd();
}


// redefined virtual function of TFile

//______________________________________________________________________________
void TXMLFile::Browse(TBrowser *b)
{
   if (b) {
      TObject *obj = 0;
      TXMLKey *key = 0;
      TIter next(fKeys);

      cd();

      while ((key = (TXMLKey*)next())) {
         obj = key->GetObject();
         if (obj) b->Add(obj, obj->GetName());
      }
   }
}


//______________________________________________________________________________
Bool_t TXMLFile::cd(const char*) {
	gFile = this;
	gDirectory = this;
    return kTRUE;
}


//______________________________________________________________________________
void TXMLFile::operator=(const TXMLFile &) {
}

//______________________________________________________________________________
Bool_t TXMLFile::IsOpen() const {
   return kTRUE;
}

//______________________________________________________________________________
Int_t TXMLFile::ReOpen(Option_t*) {
   return 0;
}


// **************************** XML specific functions *********************************

//______________________________________________________________________________
Int_t TXMLFile::WriteObject(const TObject* obj, const char* name, Option_t *) {
   new TXMLKey(this, obj, name);
   return 0;
}

//______________________________________________________________________________
Int_t TXMLFile::WriteObjectAny(const void* obj, const TClass* cl, const char* name,Option_t *) {
   new TXMLKey(this, obj, cl, name);
   return 0;
}


//______________________________________________________________________________
void TXMLFile::ProduceFileNames(const char* filename, TString& fname, TString& dtdname) {
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
void TXMLFile::SaveToFile() {
   if (fDoc==0) return;

   xmlNodePointer fRootNode = gXML->DocGetRootElement(fDoc);

   TString fname, dtdname;
   ProduceFileNames(fRealName, fname, dtdname);

   TIter iter(GetListOfKeys());
   TXMLKey* key = 0;

   while ((key=(TXMLKey*)iter()) !=0)
      gXML->AddChild(fRootNode, key->KeyNode());

   WriteStreamerInfo();
   
   if (fStreamerInfoNode)
     gXML->AddChild(fRootNode, fStreamerInfoNode);

   if (fDtdGener && IsUseDtd())
     gXML->AssignDtd(fDoc, dtdname, xmlNames_Root);

   Int_t layout = GetCompressionLevel()>2 ? 0 : 1;

   gXML->SaveDoc(fDoc, fname, layout);

   if (fDtdGener && IsUseDtd()) {
       fDtdGener->Produce(dtdname);
   }

   iter.Reset();
   while ((key=(TXMLKey*)iter()) !=0)
      gXML->UnlinkChild(key->KeyNode());
      
   if (fStreamerInfoNode)
     gXML->UnlinkChild(fStreamerInfoNode);
}

//______________________________________________________________________________
Bool_t TXMLFile::ReadFromFile() {
   fDoc = gXML->ParseFile(fRealName);
   if (fDoc==0) return kFALSE;

   xmlNodePointer fRootNode = gXML->DocGetRootElement(fDoc);

   if (fRootNode==0) {
      gXML->FreeDoc(fDoc);
      fDoc=0;
      return kFALSE;
   }

   ReadSetup(fRootNode);
   
   fStreamerInfoNode = gXML->GetChild(fRootNode);
   gXML->SkipEmpty(fStreamerInfoNode);
   while (fStreamerInfoNode!=0) {
      if (strcmp("StreamerInfos",gXML->GetNodeName(fStreamerInfoNode))==0) break;
      gXML->ShiftToNext(fStreamerInfoNode);
   }
   gXML->UnlinkChild(fStreamerInfoNode);

   ReadStreamerInfo();

   if (IsUseDtd())
     if (!gXML->ValidateDocument(fDoc, gDebug>0)) {
        gXML->FreeDoc(fDoc);
        fDoc=0;
        return kFALSE;
     }

   gFile = this;
   gDirectory = this;

   xmlNodePointer keynode = gXML->GetChild(fRootNode);
   gXML->SkipEmpty(keynode);
   while (keynode!=0) {
      xmlNodePointer next = gXML->GetNext(keynode);
      gXML->UnlinkChild(keynode);

      TXMLKey* key = new TXMLKey(this, keynode);

      if (gDebug>2)
        cout << "Adding key " << gXML->GetNodeName(keynode) << "   name = " << key->GetName() << endl;
      fKeys->Add(key);

      keynode = next;
      gXML->SkipEmpty(keynode);
   }

   return kTRUE;
}

//______________________________________________________________________________
TObject* TXMLFile::Get(const char* name) {
   TXMLKey* key = dynamic_cast<TXMLKey*> (GetKey(name));
   if (key==0) return 0;
   return key->GetObject();
}

//______________________________________________________________________________
void* TXMLFile::GetAny(const char* name) {
   TXMLKey* key = dynamic_cast<TXMLKey*> (GetKey(name));
   if (key==0) return 0;
   return key->GetObjectAny();
}

//______________________________________________________________________________
void TXMLFile::WriteStreamerInfo() {
   // exclude all streamer info node if exists 
   if (fStreamerInfoNode) {
     gXML->FreeNode(fStreamerInfoNode);
     fStreamerInfoNode = 0;
   }
    
   TObjArray list;

   TIter iter(gROOT->GetListOfStreamerInfo());

   TStreamerInfo* info = 0;

   while ((info = (TStreamerInfo*) iter()) !=0 ) {
      Int_t uid = info->GetNumber();
      if (fClassIndex->fArray[uid])
        list.Add(info);
   }

   if (list.GetSize()==0) return;

   fStreamerInfoNode = gXML->NewChild(0, 0, "StreamerInfos");
   for (int n=0;n<=list.GetLast();n++) {
      TStreamerInfo* info  = (TStreamerInfo*) list.At(n);

      xmlNodePointer infonode = gXML->NewChild(fStreamerInfoNode, 0, "TStreamerInfo");

      gXML->NewProp(infonode, 0, "name", info->GetName());
      gXML->NewProp(infonode, 0, "title", info->GetTitle());

      char sbuf[100];

      sprintf(sbuf, "%d", info->IsA()->GetClassVersion());
      gXML->NewProp(infonode, 0, "v", sbuf);

      sprintf(sbuf, "%d", info->GetClassVersion());
      gXML->NewProp(infonode,0,"classversion", sbuf);

      sprintf(sbuf, "%d", info->GetCheckSum());
      gXML->NewProp(infonode,0,"checksum", sbuf);

      TIter iter(info->GetElements());
      TStreamerElement* elem=0;
      while ((elem= (TStreamerElement*) iter()) != 0) {
         StoreStreamerElement(infonode, elem);
      }
   }
}

//______________________________________________________________________________
TList* TXMLFile::GetStreamerInfoList() {
   if (fStreamerInfoNode==0) return 0;
    
   TList* list = new TList(); 
   
   xmlNodePointer sinfonode = gXML->GetChild(fStreamerInfoNode);
   gXML->SkipEmpty(sinfonode);

   while (sinfonode!=0) {
     if (strcmp("TStreamerInfo",gXML->GetNodeName(sinfonode))==0) {
        TString fname = gXML->GetProp(sinfonode,"name");
        TString ftitle = gXML->GetProp(sinfonode,"title");

        TStreamerInfo* info = new TStreamerInfo(gROOT->GetClass(fname), ftitle);

        list->Add(info);

        Int_t clversion = AtoI(gXML->GetProp(sinfonode,"classversion"));
        info->SetClassVersion(clversion);
        Int_t checksum = AtoI(gXML->GetProp(sinfonode,"checksum"));
        info->SetCheckSum(checksum);

        xmlNodePointer node = gXML->GetChild(sinfonode);
        gXML->SkipEmpty(node);
        while (node!=0) {
           ReadStreamerElement(node, info);
           gXML->ShiftToNext(node);
        }
     }
     gXML->ShiftToNext(sinfonode);
   }
   
   return list;
}

//______________________________________________________________________________
void TXMLFile::ReadStreamerInfo() {

   TList* list = GetStreamerInfoList();
   if (list==0) return;
   
   delete list;
   return;
   
   cout << "Loop over all classes num = " << list->GetSize() << endl;

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

//   list->Clear();
   delete list;
}


//______________________________________________________________________________
void TXMLFile::StoreStreamerElement(xmlNodePointer infonode, TStreamerElement* elem) {
   TClass* cl = elem->IsA();

   xmlNodePointer node = gXML->NewChild(infonode, 0, cl->GetName());

   char sbuf[100], namebuf[100];

   gXML->NewProp(node,0,"name",elem->GetName());
   if (strlen(elem->GetTitle())>0)
     gXML->NewProp(node,0,"title",elem->GetTitle());

   sprintf(sbuf, "%d", cl->GetClassVersion());
   gXML->NewProp(node,0,"v",sbuf);

   sprintf(sbuf, "%d", elem->GetType());
   gXML->NewProp(node,0,"type", sbuf);

   if (strlen(elem->GetTypeName())>0)
     gXML->NewProp(node,0,"typename", elem->GetTypeName());

   sprintf(sbuf, "%d", elem->GetSize());
   gXML->NewProp(node,0,"size", sbuf);

   if (elem->GetArrayDim()>0) {
      sprintf(sbuf, "%d", elem->GetArrayDim());
      gXML->NewProp(node,0,"numdim", sbuf);

      for (int ndim=0;ndim<elem->GetArrayDim();ndim++) {
         sprintf(namebuf, "dim%d", ndim);
         sprintf(sbuf, "%d", elem->GetMaxIndex(ndim));
         gXML->NewProp(node,0, namebuf, sbuf);
      }
   }

   if (cl == TStreamerBase::Class()) {
      TStreamerBase* base = (TStreamerBase*) elem;
      sprintf(sbuf, "%d", base->GetBaseVersion());
      gXML->NewProp(node,0, "baseversion", sbuf);
   } else
   if (cl == TStreamerBasicPointer::Class()) {
     TStreamerBasicPointer* bptr = (TStreamerBasicPointer*) elem;
     sprintf(sbuf, "%d", bptr->GetCountVersion());
     gXML->NewProp(node, 0, "countversion", sbuf);
     gXML->NewProp(node, 0, "countname", bptr->GetCountName());
     gXML->NewProp(node, 0, "countclass", bptr->GetCountClass());
   } else
   if (cl == TStreamerLoop::Class()) {
     TStreamerLoop* loop = (TStreamerLoop*) elem;
     sprintf(sbuf, "%d", loop->GetCountVersion());
     gXML->NewProp(node, 0, "countversion", sbuf);
     gXML->NewProp(node, 0, "countname", loop->GetCountName());
     gXML->NewProp(node, 0, "countclass", loop->GetCountClass());
   } else
   if ((cl == TStreamerSTL::Class()) || (cl == TStreamerSTLstring::Class())) {
     TStreamerSTL* stl = (TStreamerSTL*) elem;

     sprintf(sbuf, "%d", stl->GetSTLtype());
     gXML->NewProp(node,0,"STLtype", sbuf);

     sprintf(sbuf, "%d", stl->GetCtype());
     gXML->NewProp(node,0,"Ctype", sbuf);
   }
}


//______________________________________________________________________________
void TXMLFile::ReadStreamerElement(xmlNodePointer node, TStreamerInfo* info) {
   TClass* cl = gROOT->GetClass(gXML->GetNodeName(node));
   if ((cl==0) || !cl->InheritsFrom(TStreamerElement::Class())) return;

//   Int_t elemversion = gXML->GetProp(node,"v");

   TStreamerElement* elem = (TStreamerElement*) cl->New();
   
   elem->SetName(gXML->GetProp(node,"name"));
   elem->SetTitle(gXML->GetProp(node,"title"));
   elem->SetType(AtoI(gXML->GetProp(node,"type")));
   elem->SetTypeName(gXML->GetProp(node,"typename"));
   elem->SetSize(AtoI(gXML->GetProp(node,"size")));

   if (cl == TStreamerBase::Class()) {
      int basever = AtoI(gXML->GetProp(node,"baseversion"));
      ((TStreamerBase*) elem)->SetBaseVersion(basever);
   } else
   if (cl == TStreamerBasicPointer::Class()) {
     TString countname = gXML->GetProp(node,"countname");
     TString countclass = gXML->GetProp(node,"countclass");
     Int_t countversion = AtoI(gXML->GetProp(node,"countversion"));

     ((TStreamerBasicPointer*)elem)->SetCountVersion(countversion);
     ((TStreamerBasicPointer*)elem)->SetCountName(countname);
     ((TStreamerBasicPointer*)elem)->SetCountClass(countclass);
   } else
   if (cl == TStreamerLoop::Class()) {
     TString countname = gXML->GetProp(node,"countname");
     TString countclass = gXML->GetProp(node,"countclass");
     Int_t countversion = AtoI(gXML->GetProp(node,"countversion"));
     ((TStreamerLoop*)elem)->SetCountVersion(countversion);
     ((TStreamerLoop*)elem)->SetCountName(countname);
     ((TStreamerLoop*)elem)->SetCountClass(countclass);
   } else
   if ((cl == TStreamerSTL::Class()) || (cl == TStreamerSTLstring::Class()))  {
     int fSTLtype = AtoI(gXML->GetProp(node,"STLtype"));
     int fCtype = AtoI(gXML->GetProp(node,"Ctype"));
     ((TStreamerSTL*)elem)->SetSTLtype(fSTLtype);
     ((TStreamerSTL*)elem)->SetCtype(fCtype);
   }

   char namebuf[100];

   if (gXML->HasProp(node, "numdim") && (elem!=0)) {
     int numdim = AtoI(gXML->GetProp(node,"numdim"));
     elem->SetArrayDim(numdim);
     for (int ndim=0;ndim<numdim;ndim++) {
         sprintf(namebuf, "dim%d", ndim);
         int maxi = AtoI(gXML->GetProp(node, namebuf));
         elem->SetMaxIndex(ndim, maxi);
     }
   }

   info->GetElements()->Add(elem);
}





