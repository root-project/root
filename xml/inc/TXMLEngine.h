// @(#)root/xml:$Name:  $:$Id: TXMLEngine.h,v 1.3 2004/05/14 14:30:46 brun Exp $
// Author: Sergey Linev  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXMLEngine
#define ROOT_TXMLEngine

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

typedef void* xmlNodePointer;
typedef void* xmlNsPointer;
typedef void* xmlAttrPointer;
typedef void* xmlDocPointer;


class TXMLEngine : public TObject {
   public:
      TXMLEngine();
      virtual ~TXMLEngine();

      Bool_t            HasAttr(xmlNodePointer node, const char* name);
      const char*       GetAttr(xmlNodePointer node, const char* name);
      Int_t             GetIntAttr(xmlNodePointer node, const char* name);
      xmlAttrPointer    NewAttr(xmlNodePointer node, xmlNsPointer ns,
                                const char* name, const char* value);
      xmlAttrPointer    NewIntAttr(xmlNodePointer node, const char* name, Int_t value);
      void              FreeAttr(xmlNodePointer node, const char* name);
      xmlNodePointer    NewChild(xmlNodePointer parent, xmlNsPointer ns,
                                 const char* name, const char* content = 0);
      xmlNsPointer      NewNS(xmlNodePointer node, const char* reference, const char* name = 0);
      void              AddChild(xmlNodePointer parent, xmlNodePointer child);
      void              UnlinkNode(xmlNodePointer node);
      void              FreeNode(xmlNodePointer node);
      void              UnlinkFreeNode(xmlNodePointer node);
      const char*       GetNodeName(xmlNodePointer node);
      const char*       GetNodeContent(xmlNodePointer node);
      xmlNodePointer    GetChild(xmlNodePointer node);
      xmlNodePointer    GetParent(xmlNodePointer node);
      xmlNodePointer    GetNext(xmlNodePointer node);
      void              ShiftToNext(xmlNodePointer &node, Bool_t skipempty = kTRUE);
      Bool_t            IsEmptyNode(xmlNodePointer node);
      void              SkipEmpty(xmlNodePointer &node);
      void              CleanNode(xmlNodePointer node);
      xmlDocPointer     NewDoc(const char* version = 0);
      void              AssignDtd(xmlDocPointer doc, const char* dtdname, const char* rootname);
      void              FreeDoc(xmlDocPointer doc);
      void              SaveDoc(xmlDocPointer doc, const char* filename, Int_t layout = 1);
      void              DocSetRootElement(xmlDocPointer doc, xmlNodePointer node);
      xmlNodePointer    DocGetRootElement(xmlDocPointer doc);
      xmlDocPointer     ParseFile(const char* filename);
      Bool_t            ValidateDocument(xmlDocPointer doc, Bool_t doout = kFALSE);

   protected:

      TString           fStrBuf;   //!

   ClassDef(TXMLEngine,1) // Interface to the libxml2 library
};

#endif
