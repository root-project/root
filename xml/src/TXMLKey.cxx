// @(#)root/xml:$Name:  $:$Id: TXMLKey.cxx,v 1.4 2004/05/14 14:30:46 brun Exp $
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
// TXMLKey is represents one block of data in TXMLFile
// Normally this block corresponds to data of single object like histogram, 
// TObjArray and so on.
//________________________________________________________________________


#include "TXMLKey.h"

#include "TXMLBuffer.h"
#include "TXMLFile.h"
#include "TClass.h"
#include "TBrowser.h"
#include "Riostream.h"

ClassImp(TXMLKey);

//______________________________________________________________________________
TXMLKey::TXMLKey() :
   TKey(), 
   fFile(0), 
   fXML(0),
   fKeyNode(0)
{
  // default constructor  
}

//______________________________________________________________________________
TXMLKey::TXMLKey(TXMLFile* file, const TObject* obj, const char* name) :
    TKey(), 
    fFile(file), 
    fXML(file->XML()), 
    fKeyNode(0)
{
// Creates TXMLKey and convert obj data to xml structures
    
    if (name) SetName(name); else
       if (obj!=0) {SetName(obj->GetName());  fClassName=obj->ClassName();}
        else SetName("Noname");

   StoreObject((void*)obj, obj ? obj->IsA() : 0);
}

//______________________________________________________________________________
TXMLKey::TXMLKey(TXMLFile* file, const void* obj, const TClass* cl, const char* name) :
    TKey(), 
    fFile(file), 
    fXML(file->XML()), 
    fKeyNode(0)
{
// Creates TXMLKey and convert obj data to xml structures
    
   if (name && *name) SetName(name);
                 else SetName(cl ? cl->GetName() : "Noname");

   StoreObject(obj, cl);
}

//______________________________________________________________________________
TXMLKey::TXMLKey(TXMLFile* file, xmlNodePointer keynode) :
    TKey(), 
    fFile(file), 
    fXML(file->XML()), 
    fKeyNode(keynode)
{
// Creates TXMLKey and takes ownership over xml node, from which object can be restored
    
    
  SetName(fXML->GetAttr(keynode, xmlNames_Name));
  fCycle = fXML->GetIntAttr(keynode, xmlNames_Cycle);

  xmlNodePointer objnode = fXML->GetChild(keynode);
  fXML->SkipEmpty(objnode);

  fClassName = fXML->GetAttr(objnode, xmlNames_ObjClass);
}

//______________________________________________________________________________
TXMLKey::~TXMLKey()
{
// TXMLKey destructor    
   if (fKeyNode)
      fXML->FreeNode(fKeyNode);
}

//______________________________________________________________________________
void TXMLKey::Browse(TBrowser *b)
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
void TXMLKey::Delete(Option_t * /*option*/)
{
// Delete key from current directory 
// Note: TXMLKey object is not deleted. You still have to call "delete key"
    
   gDirectory->GetListOfKeys()->Remove(this);  
}

//______________________________________________________________________________
void TXMLKey::StoreObject(const void* obj, const TClass* cl)
{
//  convert object to xml structure and keep this structure in key
    
   fCycle  = fFile->AppendKey(this);

   fKeyNode = fXML->NewChild(0, 0, xmlNames_Xmlkey, 0);
   fXML->NewAttr(fKeyNode, 0, xmlNames_Name, GetName());

   fXML->NewIntAttr(fKeyNode, xmlNames_Cycle, fCycle);

   TXMLBuffer buffer(TBuffer::kWrite, fFile);
   xmlNodePointer node = buffer.XmlWrite(obj, cl);

   if (node!=0)
      fXML->AddChild(fKeyNode, node);
      
   buffer.XmlWriteBlock(fKeyNode);   

   if (cl) fClassName = cl->GetName();
}

//______________________________________________________________________________
xmlNodePointer TXMLKey::ObjNode()
{
// return starting node, where object was stored
    
   if (fKeyNode==0) return 0;
   xmlNodePointer node = fXML->GetChild(fKeyNode);
   fXML->SkipEmpty(node);
   return node;
}

//______________________________________________________________________________
xmlNodePointer TXMLKey::BlockNode() 
{
// return node, where key binary data is stored    
   if (fKeyNode==0) return 0;    
   xmlNodePointer node = fXML->GetChild(fKeyNode);
   fXML->SkipEmpty(node);
   while (node!=0) {
     if (strcmp(fXML->GetNodeName(node), xmlNames_XmlBlock)==0) return node;
     fXML->ShiftToNext(node);   
   }
   return 0;
}

//______________________________________________________________________________
TObject* TXMLKey::ReadObj()
{
// read object derived from TObject class, from key 
// if it is not TObject or in case of error, return 0
    
   if (fKeyNode==0) return 0;
   TXMLBuffer buffer(TBuffer::kRead, fFile);
   buffer.XmlReadBlock(BlockNode());
   TObject* obj = buffer.XmlRead(ObjNode());
   return obj;
}

//______________________________________________________________________________
void* TXMLKey::ReadObjectAny(const TClass* /*cl*/)
{
// read object of any type    
    
   if (fKeyNode==0) return 0;
   TXMLBuffer buffer(TBuffer::kRead, fFile);
   buffer.XmlReadBlock(BlockNode());
   void* obj = buffer.XmlReadAny(ObjNode(), 0);
   return obj;
}

/*
//______________________________________________________________________________
void TXMLKey::ls(Option_t *) const
{
   // List Key contents.

   TROOT::IndentLevel();
   cout <<"KEY: "<<fClassName<<"\t"<<GetName()<<";"<<GetCycle()<<"\t"<<GetTitle()<<endl;
}
*/
