// @(#)root/xml:$Name:  $:$Id: TXMLKey.cxx,v 1.0 2004/04/21 15:06:45 brun Exp $
// Author: Sergey Linev, Rene Brun  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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
    fKeyNode(0),
	fObject(0) {
}

//______________________________________________________________________________
TXMLKey::TXMLKey(TXMLFile* file, const TObject* obj, const char* name) :
    TKey(),
    fFile(file),
    fKeyNode(0),
	fObject((void*)obj) {
    if (name) SetName(name); else
       if (obj!=0) {SetName(obj->GetName());  fClassName=obj->ClassName();}
              else SetName("Noname");
              
   StoreObject(file, (void*)obj, obj ? obj->IsA() : 0);
}

//______________________________________________________________________________
TXMLKey::TXMLKey(TXMLFile* file, const void* obj, const TClass* cl, const char* name) :
    TKey(),
    fFile(file),
    fKeyNode(0),
	fObject((void*)obj) {
   fClassName = cl->GetName();
   if (name) SetName(name);
        else SetName("Noname");
     
   StoreObject(file, obj, cl);
}

//______________________________________________________________________________
TXMLKey::TXMLKey(TXMLFile* file, xmlNodePointer keynode) :
    TKey(),
    fFile(file),
    fKeyNode(keynode),
	fObject(0) {
  SetName(gXML->GetProp(keynode, xmlNames_Name));
  fCycle = atoi(gXML->GetProp(keynode, xmlNames_Cycle));

  xmlNodePointer objnode = gXML->GetChild(keynode); 
  gXML->SkipEmpty(objnode);
  if (file->GetXmlLayout() == TXMLSetup::kGeneralized) 
     fClassName = gXML->GetProp(objnode, xmlNames_Class);
  else
     fClassName = gXML->GetNodeName(objnode);
}

//______________________________________________________________________________
TXMLKey::~TXMLKey() {
   if (fKeyNode)
      gXML->FreeNode(fKeyNode);
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

   if (b && obj) {
      obj->Browse(b);
      b->SetRefreshFlag(kTRUE);
   }
}

//______________________________________________________________________________
void TXMLKey::Delete(Option_t * /*option*/) {
}

//______________________________________________________________________________
void TXMLKey::StoreObject(TXMLFile* file, const void* obj, const TClass* cl) {
   fCycle  = file->AppendKey(this);
    
   TXMLBuffer buffer(TBuffer::kWrite, *file, file);
   buffer.SetParent(0);
   buffer.SetDtdGenerator(file->GetDtdGenerator());
   xmlNodePointer node = buffer.XmlWrite(obj, cl); 
   
   fKeyNode = gXML->NewChild(0, 0, xmlNames_Xmlkey, 0);
   gXML->NewProp(fKeyNode, 0, xmlNames_Name, GetName());

   char sbuf[100];
   sprintf(sbuf, "%d", fCycle);
   gXML->NewProp(fKeyNode, 0, xmlNames_Cycle, sbuf);
    
   if (node!=0) 
      gXML->AddChild(fKeyNode, node);

   if (cl) fClassName = cl->GetName();
}

//______________________________________________________________________________
xmlNodePointer TXMLKey::ObjNode() {
   if (fKeyNode==0) return 0;
   TXMLSetup setup;
   xmlNodePointer node = gXML->GetChild(fKeyNode);
   gXML->SkipEmpty(node);
   return node;
}

//______________________________________________________________________________
TObject* TXMLKey::GetObject() {
   if (fKeyNode==0) return 0;   
   if (fObject) return (TObject*)fObject;
   TXMLBuffer buffer(TBuffer::kRead, *fFile);
   fObject = buffer.XmlRead(ObjNode());
   return (TObject*)fObject;
}

//______________________________________________________________________________
void* TXMLKey::GetObjectAny() {
   if (fKeyNode==0) return 0; 
   if (fObject) return fObject;
   TXMLBuffer buffer(TBuffer::kRead, *fFile);
   fObject = buffer.XmlRead(ObjNode());
   return fObject;
}

//______________________________________________________________________________
void TXMLKey::ls(Option_t *) const
{
//*-*-*-*-*-*-*-*-*-*-*-*-*List Key contents-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =================
   TROOT::IndentLevel();
   cout <<"KEY: "<<fClassName<<"\t"<<GetName()<<";"<<GetCycle()<<"\t"<<GetTitle()<<endl;
}




