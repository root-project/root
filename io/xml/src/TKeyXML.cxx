// @(#)root/xml:$Id$
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
// TKeyXML is represents one block of data in TXMLFile
// Normally this block corresponds to data of single object like histogram,
// TObjArray and so on.
//________________________________________________________________________


#include "TKeyXML.h"

#include "TBufferXML.h"
#include "TXMLFile.h"
#include "TClass.h"
#include "TROOT.h"
#include "TBrowser.h"

ClassImp(TKeyXML);

//______________________________________________________________________________
TKeyXML::TKeyXML() :
   TKey(),
   fKeyNode(0),
   fKeyId(0),
   fSubdir(kFALSE)
{
   // default constructor
}

//______________________________________________________________________________
TKeyXML::TKeyXML(TDirectory* mother, Long64_t keyid, const TObject* obj, const char* name, const char* title) :
    TKey(mother),
    fKeyNode(0),
    fKeyId(keyid),
    fSubdir(kFALSE)
{
   // Creates TKeyXML and convert obj data to xml structures

   if (name)
      SetName(name);
   else
      if (obj!=0) {
         SetName(obj->GetName());
         fClassName=obj->ClassName();
      } else
         SetName("Noname");

   if (title) SetTitle(title);

   fCycle  = GetMotherDir()->AppendKey(this);

   TXMLEngine* xml = XMLEngine();
   if (xml!=0)
      fKeyNode = xml->NewChild(0, 0, xmlio::Xmlkey, 0);

   fDatime.Set();

   StoreObject((void*)obj, obj ? obj->IsA() : 0);
}

//______________________________________________________________________________
TKeyXML::TKeyXML(TDirectory* mother, Long64_t keyid, const void* obj, const TClass* cl, const char* name, const char* title) :
   TKey(mother),
   fKeyNode(0),
   fKeyId(keyid),
   fSubdir(kFALSE)
{
   // Creates TKeyXML and convert obj data to xml structures

   if (name && *name) SetName(name);
   else SetName(cl ? cl->GetName() : "Noname");

   if (title) SetTitle(title);

   fCycle  = GetMotherDir()->AppendKey(this);

   TXMLEngine* xml = XMLEngine();
   if (xml!=0)
      fKeyNode = xml->NewChild(0, 0, xmlio::Xmlkey, 0);

   fDatime.Set();

   StoreObject(obj, cl);
}

//______________________________________________________________________________
TKeyXML::TKeyXML(TDirectory* mother, Long64_t keyid, XMLNodePointer_t keynode) :
   TKey(mother),
   fKeyNode(keynode),
   fKeyId(keyid),
   fSubdir(kFALSE)
{
   // Creates TKeyXML and takes ownership over xml node, from which object can be restored

   TXMLEngine* xml = XMLEngine();

   SetName(xml->GetAttr(keynode, xmlio::Name));

   if (xml->HasAttr(keynode, xmlio::Title))
      SetTitle(xml->GetAttr(keynode, xmlio::Title));

   fCycle = xml->GetIntAttr(keynode, xmlio::Cycle);

   if (xml->HasAttr(keynode, xmlio::CreateTm)) {
      TDatime tm(xml->GetAttr(keynode, xmlio::CreateTm));
      fDatime = tm;
   }

   XMLNodePointer_t objnode = xml->GetChild(keynode);
   xml->SkipEmpty(objnode);

   fClassName = xml->GetAttr(objnode, xmlio::ObjClass);
}

//______________________________________________________________________________
TKeyXML::~TKeyXML()
{
   // TKeyXML destructor
   if (fKeyNode) {
      TXMLEngine* xml = XMLEngine();
      if (xml) {
         xml->FreeNode(fKeyNode);
      } else {
         TXMLEngine xml_;
         xml_.FreeNode(fKeyNode);
      }
   }
}

//______________________________________________________________________________
void TKeyXML::Delete(Option_t * /*option*/)
{
   // Delete key from current directory
   // Note: TKeyXML object is not deleted. You still have to call "delete key"

   TXMLEngine* xml = XMLEngine();
   if (fKeyNode && xml) {
      xml->FreeNode(fKeyNode);
      fKeyNode = 0;
   }

   fMotherDir->GetListOfKeys()->Remove(this);
}

//______________________________________________________________________________
void TKeyXML::StoreKeyAttributes()
{
   // Stores keys attributes in key node

   TXMLEngine* xml = XMLEngine();
   TXMLFile* f = (TXMLFile*) GetFile();
   if ((f==0) || (xml==0) || (fKeyNode==0)) return;

   xml->NewAttr(fKeyNode, 0, xmlio::Name, GetName());

   xml->NewIntAttr(fKeyNode, xmlio::Cycle, fCycle);

   if (f->GetIOVersion()>1) {
      if (strlen(GetTitle())>0)
         xml->NewAttr(fKeyNode, 0, xmlio::Title, GetTitle());
      xml->NewAttr(fKeyNode, 0, xmlio::CreateTm, fDatime.AsSQLString());
   }
}

//______________________________________________________________________________
void TKeyXML::StoreObject(const void* obj, const TClass* cl)
{
   //  convert object to xml structure and keep this structure in key

   TXMLFile* f = (TXMLFile*) GetFile();
   TXMLEngine* xml = XMLEngine();
   if ((f==0) || (xml==0) || (fKeyNode==0)) return;

   StoreKeyAttributes();

   TBufferXML buffer(TBuffer::kWrite, f);
   if (f->GetIOVersion()==1)
      buffer.SetBit(TBuffer::kCannotHandleMemberWiseStreaming, kFALSE);

   XMLNodePointer_t node = buffer.XmlWriteAny(obj, cl);

   if (node!=0)
      xml->AddChildFirst(fKeyNode, node);

   buffer.XmlWriteBlock(fKeyNode);

   if (cl) fClassName = cl->GetName();
}

//______________________________________________________________________________
void TKeyXML::UpdateAttributes()
{
   // update key attributes in key node

   TXMLEngine* xml = XMLEngine();
   if ((xml==0) || (fKeyNode==0)) return;

   xml->FreeAllAttr(fKeyNode);

   StoreKeyAttributes();
}

//______________________________________________________________________________
void TKeyXML::UpdateObject(TObject* obj)
{
   // updates object, stored in the node
   // Used for TDirectory data update

   TXMLFile* f = (TXMLFile*) GetFile();
   TXMLEngine* xml = XMLEngine();
   if ((f==0) || (xml==0) || (obj==0) || (fKeyNode==0)) return;

   XMLNodePointer_t objnode = xml->GetChild(fKeyNode);
   xml->SkipEmpty(objnode);

   if (objnode==0) return;

   xml->UnlinkNode(objnode);
   xml->FreeNode(objnode);

   xml->FreeAllAttr(fKeyNode);

   StoreObject(obj, obj->IsA());
}

//______________________________________________________________________________
Int_t TKeyXML::Read(TObject* tobj)
{
   // To read an object from the file.
   // The object associated to this key is read from the file into memory.
   // Before invoking this function, obj has been created via the
   // default constructor.

   if (tobj==0) return 0;

   void* res = XmlReadAny(tobj, 0);

   return res==0 ? 0 : 1;
}

//______________________________________________________________________________
TObject* TKeyXML::ReadObj()
{
   // read object derived from TObject class, from key
   // if it is not TObject or in case of error, return 0

   TObject* tobj = (TObject*) XmlReadAny(0, TObject::Class());

   if (tobj!=0) {
      if (gROOT->GetForceStyle()) tobj->UseCurrentStyle();
      if (tobj->IsA() == TDirectoryFile::Class()) {
         TDirectoryFile *dir = (TDirectoryFile*) tobj;
         dir->SetName(GetName());
         dir->SetTitle(GetTitle());
         dir->SetSeekDir(GetKeyId());
         // set mother before reading keys
         dir->SetMother(fMotherDir);
         dir->ReadKeys();
         fMotherDir->Append(dir);
         fSubdir = kTRUE;
      }
   }

   return tobj;
}

//______________________________________________________________________________
TObject* TKeyXML::ReadObjWithBuffer(char * /*bufferRead*/)
{
   // read object derived from TObject class, from key
   // if it is not TObject or in case of error, return 0

   TObject* tobj = (TObject*) XmlReadAny(0, TObject::Class());

   if (tobj!=0) {
      if (gROOT->GetForceStyle()) tobj->UseCurrentStyle();
      if (tobj->IsA() == TDirectoryFile::Class()) {
         TDirectoryFile *dir = (TDirectoryFile*) tobj;
         dir->SetName(GetName());
         dir->SetTitle(GetTitle());
         dir->SetSeekDir(GetKeyId());
         // set mother before reading keys
         dir->SetMother(fMotherDir);
         dir->ReadKeys();
         fMotherDir->Append(dir);
         fSubdir = kTRUE;
      }
   }

   return tobj;
}

//______________________________________________________________________________
void* TKeyXML::ReadObjectAny(const TClass *expectedClass)
{
   // read object of any type

   return XmlReadAny(0, expectedClass);
}

//______________________________________________________________________________
void* TKeyXML::XmlReadAny(void* obj, const TClass* expectedClass)
{
   // read object from key and cast to expected class

   if (fKeyNode==0) return obj;

   TXMLFile* f = (TXMLFile*) GetFile();
   TXMLEngine* xml = XMLEngine();
   if ((f==0) || (xml==0)) return obj;

   TBufferXML buffer(TBuffer::kRead, f);
   if (f->GetIOVersion()==1)
      buffer.SetBit(TBuffer::kCannotHandleMemberWiseStreaming, kFALSE);

   XMLNodePointer_t blocknode = xml->GetChild(fKeyNode);
   xml->SkipEmpty(blocknode);
   while (blocknode!=0) {
      if (strcmp(xml->GetNodeName(blocknode), xmlio::XmlBlock)==0) break;
      xml->ShiftToNext(blocknode);
   }
   buffer.XmlReadBlock(blocknode);

   XMLNodePointer_t objnode = xml->GetChild(fKeyNode);
   xml->SkipEmpty(objnode);

   TClass* cl = 0;
   void* res = buffer.XmlReadAny(objnode, obj, &cl);

   if ((cl==0) || (res==0)) return obj;

   Int_t delta = 0;

   if (expectedClass!=0) {
      delta = cl->GetBaseClassOffset(expectedClass);
      if (delta<0) {
         if (obj==0) cl->Destructor(res);
         return 0;
      }
      if (cl->GetState() > TClass::kEmulated && expectedClass->GetState() <= TClass::kEmulated) {
         //we cannot mix a compiled class with an emulated class in the inheritance
         Warning("XmlReadAny",
                 "Trying to read an emulated class (%s) to store in a compiled pointer (%s)",
                 cl->GetName(),expectedClass->GetName());
      }
   }

   return ((char*)res) + delta;
}

//______________________________________________________________________________
TXMLEngine* TKeyXML::XMLEngine()
{
   // return pointer on TXMLEngine object, used for xml conversion

   TXMLFile* f = (TXMLFile*) GetFile();
   return f==0 ? 0 : f->XML();
}
