// @(#)root/xml:$Name:  $:$Id: TKeyXML.cxx,v 1.6 2006/01/25 16:00:11 pcanal Exp $
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
   fKeyNode(0)
{
   // default constructor
}

//______________________________________________________________________________
TKeyXML::TKeyXML(TDirectory* mother, const TObject* obj, const char* name) :
    TKey(mother),
    fKeyNode(0)
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

   StoreObject((void*)obj, obj ? obj->IsA() : 0);
}

//______________________________________________________________________________
TKeyXML::TKeyXML(TDirectory* mother, const void* obj, const TClass* cl, const char* name) :
   TKey(mother),
   fKeyNode(0)
{
   // Creates TKeyXML and convert obj data to xml structures

   if (name && *name) SetName(name);
   else SetName(cl ? cl->GetName() : "Noname");

   StoreObject(obj, cl);
}

//______________________________________________________________________________
TKeyXML::TKeyXML(TDirectory* mother, XMLNodePointer_t keynode) :
   TKey(mother),
   fKeyNode(keynode)
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
   TXMLEngine* xml = XMLEngine();
   if (fKeyNode && xml)
      xml->FreeNode(fKeyNode);
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
void TKeyXML::StoreObject(const void* obj, const TClass* cl)
{
   //  convert object to xml structure and keep this structure in key
   
   TXMLFile* f = (TXMLFile*) GetFile();
   TXMLEngine* xml = XMLEngine();
   if ((f==0) || (xml==0)) return;

   fCycle  = GetMotherDir()->AppendKey(this);

   fKeyNode = xml->NewChild(0, 0, xmlio::Xmlkey, 0);
   xml->NewAttr(fKeyNode, 0, xmlio::Name, GetName());
   
   xml->NewIntAttr(fKeyNode, xmlio::Cycle, fCycle);
   
   if (f->GetIOVersion()>1) {
      if (strlen(GetTitle())>0)
         xml->NewAttr(fKeyNode, 0, xmlio::Title, GetTitle());
      fDatime.Set();
      xml->NewAttr(fKeyNode, 0, xmlio::CreateTm, fDatime.AsSQLString());
   }
   
   TBufferXML buffer(TBuffer::kWrite, f);
   if (f->GetIOVersion()==1)
      buffer.SetBit(TBuffer::kCannotHandleMemberWiseStreaming, kFALSE);
   
   XMLNodePointer_t node = buffer.XmlWriteAny(obj, cl);

   if (node!=0)
      xml->AddChild(fKeyNode, node);

   buffer.XmlWriteBlock(fKeyNode);

   if (cl) fClassName = cl->GetName();
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
      if (tobj->IsA() == TDirectory::Class()) {
         TDirectory *dir = (TDirectory*) tobj;
         dir->SetName(GetName());
         dir->SetTitle(GetTitle());
         dir->ReadKeys();
         dir->SetMother(fMotherDir);
         fMotherDir->Append(dir);
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
      if (cl->GetClassInfo() && !expectedClass->GetClassInfo()) {
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
