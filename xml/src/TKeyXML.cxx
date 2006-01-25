// @(#)root/xml:$Name:  $:$Id: TKeyXML.cxx,v 1.5 2006/01/20 01:12:13 pcanal Exp $
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
   fFile(0),
   fXML(0),
   fKeyNode(0)
{
   // default constructor
}

//______________________________________________________________________________
TKeyXML::TKeyXML(TXMLFile* file, const TObject* obj, const char* name) :
    TKey(),
    fFile(file),
    fXML(file->XML()),
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
TKeyXML::TKeyXML(TXMLFile* file, const void* obj, const TClass* cl, const char* name) :
   TKey(),
   fFile(file),
   fXML(file->XML()),
   fKeyNode(0)
{
   // Creates TKeyXML and convert obj data to xml structures

   if (name && *name) SetName(name);
   else SetName(cl ? cl->GetName() : "Noname");

   StoreObject(obj, cl);
}

//______________________________________________________________________________
TKeyXML::TKeyXML(TXMLFile* file, XMLNodePointer_t keynode) :
   TKey(),
   fFile(file),
   fXML(file->XML()),
   fKeyNode(keynode)
{
   // Creates TKeyXML and takes ownership over xml node, from which object can be restored

   SetName(fXML->GetAttr(keynode, xmlio::Name));
   
   if (fXML->HasAttr(keynode, xmlio::Title))
      SetTitle(fXML->GetAttr(keynode, xmlio::Title));

   fCycle = fXML->GetIntAttr(keynode, xmlio::Cycle);
      
   if (fXML->HasAttr(keynode, xmlio::CreateTm)) {
      TDatime tm(fXML->GetAttr(keynode, xmlio::CreateTm)); 
      fDatime = tm;
   }

   XMLNodePointer_t objnode = fXML->GetChild(keynode);
   fXML->SkipEmpty(objnode);

   fClassName = fXML->GetAttr(objnode, xmlio::ObjClass);
}

//______________________________________________________________________________
TKeyXML::~TKeyXML()
{
   // TKeyXML destructor
   if (fKeyNode)
      fXML->FreeNode(fKeyNode);
}

//______________________________________________________________________________
void TKeyXML::Browse(TBrowser *b)
{
   // Browse object corresponding to this key

   TObject *obj = gDirectory->GetList()->FindObject(GetName());
   if (obj && !obj->IsFolder()) {
      if (obj->InheritsFrom(TCollection::Class()))
         obj->Delete();   // delete also collection elements
      delete obj;
      obj = 0;
   }

   if (!obj)
      obj = ReadObj();

   if (b && obj) {
      obj->Browse(b);
      b->SetRefreshFlag(kTRUE);
   }
}

//______________________________________________________________________________
void TKeyXML::Delete(Option_t * /*option*/)
{
   // Delete key from current directory
   // Note: TKeyXML object is not deleted. You still have to call "delete key"

   gDirectory->GetListOfKeys()->Remove(this);
}

//______________________________________________________________________________
void TKeyXML::StoreObject(const void* obj, const TClass* cl)
{
   //  convert object to xml structure and keep this structure in key

   fCycle  = fFile->AppendKey(this);

   fKeyNode = fXML->NewChild(0, 0, xmlio::Xmlkey, 0);
   fXML->NewAttr(fKeyNode, 0, xmlio::Name, GetName());
   
   fXML->NewIntAttr(fKeyNode, xmlio::Cycle, fCycle);
   
   if (fFile->GetIOVersion()>1) {
      if (strlen(GetTitle())>0)
         fXML->NewAttr(fKeyNode, 0, xmlio::Title, GetTitle());
      fDatime.Set();
      fXML->NewAttr(fKeyNode, 0, xmlio::CreateTm, fDatime.AsSQLString());
   }
   
   TBufferXML buffer(TBuffer::kWrite, fFile);
   if (fFile->GetIOVersion()==1)
      buffer.SetBit(TBuffer::kCannotHandleMemberWiseStreaming, kFALSE);
   
   XMLNodePointer_t node = buffer.XmlWriteAny(obj, cl);

   if (node!=0)
      fXML->AddChild(fKeyNode, node);

   buffer.XmlWriteBlock(fKeyNode);

   if (cl) fClassName = cl->GetName();
}

//______________________________________________________________________________
XMLNodePointer_t TKeyXML::ObjNode()
{
   // return starting node, where object was stored

   if (fKeyNode==0) return 0;
   XMLNodePointer_t node = fXML->GetChild(fKeyNode);
   fXML->SkipEmpty(node);
   return node;
}

//______________________________________________________________________________
XMLNodePointer_t TKeyXML::BlockNode()
{
   // return node, where key binary data is stored
   if (fKeyNode==0) return 0;
   XMLNodePointer_t node = fXML->GetChild(fKeyNode);
   fXML->SkipEmpty(node);
   while (node!=0) {
      if (strcmp(fXML->GetNodeName(node), xmlio::XmlBlock)==0) return node;
      fXML->ShiftToNext(node);
   }
   return 0;
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
   
   if ((tobj!=0) && gROOT->GetForceStyle()) tobj->UseCurrentStyle();
       
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

   if (fKeyNode==0) return 0;
   
   TBufferXML buffer(TBuffer::kRead, fFile);
   if (fFile->GetIOVersion()==1)
      buffer.SetBit(TBuffer::kCannotHandleMemberWiseStreaming, kFALSE);
   buffer.XmlReadBlock(BlockNode());

   TClass* cl = 0;
   void* res = buffer.XmlReadAny(ObjNode(), obj, &cl);
   
   if ((cl==0) || (res==0)) return 0;
   
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
