// @(#)root/xml:$Name:  $:$Id: TXMLEngine.cxx,v 1.7 2004/05/18 14:13:20 brun Exp $
// Author: Sergey Linev  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//________________________________________________________________________
//
//  The aim of TXMLEngine class is to provide interface to libxml2 library.
//  It defines only used in ROOT functions of libxml2
//  There is a problem to parse libxml2 include files through CINT, 
//  therefore several wrappers for libxml2 types are defined.
//  More information about libxml2 itself can be found in http://xmlsoft.org
//________________________________________________________________________


#include "TXMLEngine.h"

#include "Riostream.h"
#include "libxml/tree.h"

ClassImp(TXMLEngine);


#if (_MSC_VER == 1200) && (WINVER < 0x0500) 
extern "C" long _ftol( double ); //defined by VC6 C libs 
extern "C" long _ftol2( double dblSource ) { return _ftol( dblSource ); } 
#endif

//______________________________________________________________________________
TXMLEngine::TXMLEngine() 
{
   
}

//______________________________________________________________________________
TXMLEngine::~TXMLEngine() 
{
}

//______________________________________________________________________________
Bool_t TXMLEngine::HasAttr(xmlNodePointer node, const char* name) {

   if ((node==0) || (name==0)) return kFALSE;
   return xmlHasProp((xmlNodePtr) node, (const xmlChar*) name) != 0;
}

//______________________________________________________________________________
const char* TXMLEngine::GetAttr(xmlNodePointer node, const char* name) {
   if (node==0) return 0;
   xmlChar* prop = xmlGetProp((xmlNodePtr) node, (const xmlChar*) name);
   if (prop) {
      fStrBuf = (const char*) prop;
      xmlFree(prop);
      return fStrBuf.Data();
   }
   return 0;
}

//______________________________________________________________________________
Int_t TXMLEngine::GetIntAttr(xmlNodePointer node, const char* name) {
   if (node==0) return 0;
   
   Int_t res = 0;
   
   xmlChar* prop = xmlGetProp((xmlNodePtr) node, (const xmlChar*) name);
   if (prop) {
      sscanf((const char*) prop,"%d", &res);
      xmlFree(prop);
   }
   return res;
}

//______________________________________________________________________________
xmlAttrPointer TXMLEngine::NewAttr(xmlNodePointer node,
                                   xmlNsPointer ns,
                                   const char* name,
                                   const char* value) {
   if (ns==0)
      return (xmlAttrPointer) xmlNewProp((xmlNodePtr) node,
                                         (const xmlChar*) name,
                                         (const xmlChar*) value);
   else
      return (xmlAttrPointer) xmlNewNsProp((xmlNodePtr) node,
                                           (xmlNsPtr) ns,
                                           (const xmlChar*) name,
                                           (const xmlChar*) value);
}

//______________________________________________________________________________
xmlAttrPointer TXMLEngine::NewIntAttr(xmlNodePointer node,
                                      const char* name,
                                      Int_t value)
{
  char sbuf[30];
  sprintf(sbuf,"%d",value);
  return NewAttr(node, 0, name, sbuf);
}                                      

//______________________________________________________________________________
void TXMLEngine::FreeAttr(xmlNodePointer node, const char* name) 
{
   if ((node==0) || (name==0)) return;
   xmlAttrPtr attr = xmlHasProp((xmlNodePtr) node, (const xmlChar*) name);
   if (attr!=0) xmlRemoveProp(attr);
}


//______________________________________________________________________________
xmlNodePointer TXMLEngine::NewChild(xmlNodePointer parent,
                                    xmlNsPointer ns,
                                    const char* name,
                                    const char* content) {
   xmlNodePointer node = xmlNewNode((xmlNsPtr) ns, (const xmlChar*) name);
   if (parent!=0)
     xmlAddChild((xmlNodePtr) parent, (xmlNodePtr) node);
   if (content!=0)
     xmlNodeAddContent((xmlNodePtr) node, (const xmlChar*) content);
   return node;
}

//______________________________________________________________________________
xmlNsPointer TXMLEngine::NewNS(xmlNodePointer node,
                               const char* reference,
                               const char* name) {
   return (xmlNsPointer) xmlNewNs((xmlNodePtr) node, (const xmlChar*) reference, (const xmlChar*) name);
}

//______________________________________________________________________________
void TXMLEngine::AddChild(xmlNodePointer parent, xmlNodePointer child) {
   xmlAddChild((xmlNodePtr) parent, (xmlNodePtr) child);
}

//______________________________________________________________________________
void TXMLEngine::UnlinkNode(xmlNodePointer node) {
   if (node!=0)  
     xmlUnlinkNode((xmlNodePtr) node);
}

//______________________________________________________________________________
void TXMLEngine::FreeNode(xmlNodePointer node) {
   xmlFreeNode((xmlNodePtr) node);
}

//______________________________________________________________________________
void TXMLEngine::UnlinkFreeNode(xmlNodePointer node) 
{
   UnlinkNode(node); 
   FreeNode(node);
}

//______________________________________________________________________________
const char* TXMLEngine::GetNodeName(xmlNodePointer node) {
   if (node==0) return 0;
   return (const char*) ((xmlNodePtr) node)->name;
}

//______________________________________________________________________________
const char* TXMLEngine::GetNodeContent(xmlNodePointer node) {
   if (node==0) return 0;
   xmlChar* cont = xmlNodeGetContent((xmlNodePtr) node);
   if (cont) {
      fStrBuf = (const char*) cont;
      xmlFree(cont);
      return fStrBuf.Data();
   }
   return 0;
}

//______________________________________________________________________________
xmlNodePointer TXMLEngine::GetChild(xmlNodePointer node) {
   if (node==0) return 0;
   return ((xmlNodePtr) node)->xmlChildrenNode;
}

//______________________________________________________________________________
xmlNodePointer TXMLEngine::GetParent(xmlNodePointer node) {
   if (node==0) return 0;
   return ((xmlNodePtr) node)->parent;
}

//______________________________________________________________________________
xmlNodePointer TXMLEngine::GetNext(xmlNodePointer node) {
   if (node==0) return 0;
   return ((xmlNodePtr) node)->next;
}

//______________________________________________________________________________
Bool_t TXMLEngine::IsEmptyNode(xmlNodePointer node) 
{
// checks if provided node is empty
  if (node==0) return kTRUE;
  return xmlIsBlankNode((xmlNodePtr) node);
}

//______________________________________________________________________________
void TXMLEngine::CleanNode(xmlNodePointer node) {
   if (node==0) return;
   xmlNodePointer child = GetChild(node);
   while (child!=0) {
      xmlNodePointer next = GetNext(child);
      UnlinkNode(child);
      FreeNode(child);
      child = next;
   }
}

//______________________________________________________________________________
void TXMLEngine::ShiftToNext(xmlNodePointer &node, Bool_t skipempty) {
   if (node==0) return;
   node = ((xmlNodePtr) node)->next;

   if (skipempty) 
     SkipEmpty(node);
}

//______________________________________________________________________________
void TXMLEngine::SkipEmpty(xmlNodePointer &node) {
   while (node && IsEmptyNode(node))
     node = ((xmlNodePtr) node) ->next;
}

//______________________________________________________________________________
xmlDocPointer TXMLEngine::NewDoc(const char* version) {
   return xmlNewDoc((const xmlChar*)version);
}

//______________________________________________________________________________
void TXMLEngine::AssignDtd(xmlDocPointer doc, const char* dtdname, const char* rootname) {
   xmlCreateIntSubset((xmlDocPtr) doc,
                      (const xmlChar*) rootname,
                      (const xmlChar*) "-//CERN//ROOT//v 1.0//EN",
                      (const xmlChar*) dtdname);
}

//______________________________________________________________________________
void TXMLEngine::FreeDoc(xmlDocPointer doc) {
   xmlFreeDoc((xmlDocPtr) doc);
}

//______________________________________________________________________________
void TXMLEngine::SaveDoc(xmlDocPointer doc, const char* filename, Int_t layout) {
   xmlSaveFormatFile(filename, (xmlDocPtr) doc, layout);
}

//______________________________________________________________________________
void TXMLEngine::DocSetRootElement(xmlDocPointer doc, xmlNodePointer node) {
   xmlDocSetRootElement((xmlDocPtr) doc, (xmlNodePtr) node);
}

//______________________________________________________________________________
xmlNodePointer TXMLEngine::DocGetRootElement(xmlDocPointer doc) {
   return (xmlNodePointer) xmlDocGetRootElement((xmlDocPtr) doc);
}

//______________________________________________________________________________
xmlDocPointer TXMLEngine::ParseFile(const char* filename) {
   return (xmlDocPointer) xmlParseFile(filename);
}

//______________________________________________________________________________
Bool_t TXMLEngine::ValidateDocument(xmlDocPointer doc, Bool_t doout) {
	xmlValidCtxt cvp;
	cvp.userData = doout ? (void *) stderr : 0;
//	cvp.error    = doout ? (xmlValidityErrorFunc) fprintf : 0;
//	cvp.warning  = doout ? (xmlValidityWarningFunc) fprintf : 0;
// debug output excluded to solve problem for Solaris compilation
	cvp.error    = 0;
	cvp.warning  = 0;

    xmlDocPtr docptr = (xmlDocPtr) doc;

    int res = xmlValidateDocument(&cvp, docptr);

    if ((docptr->intSubset!=0) || (docptr->extSubset!=0)) {
       xmlDtdPtr i = docptr->intSubset;
       xmlDtdPtr e = docptr->extSubset;
       if (i==e) e = 0;
       if (e!=0) {
         xmlUnlinkNode((xmlNodePtr) docptr->extSubset);
         docptr->extSubset = 0;
         xmlFreeDtd(e);
       }
       if (i!=0) {
         xmlUnlinkNode((xmlNodePtr) docptr->intSubset);
         docptr->intSubset = 0;
         xmlFreeDtd(i);
       }
    }

    if (doout)
      if (res==1) cout << "Validation done" << endl;
             else cout << "Validation failed" << endl;

    return (res == 1);
}


