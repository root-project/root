// @(#)root/xml:$Name:  $:$Id: TXMLEngine.h,v 1.11 2006/05/30 12:59:30 brun Exp $
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

typedef void* XMLNodePointer_t;
typedef void* XMLNsPointer_t;
typedef void* XMLAttrPointer_t;
typedef void* XMLDocPointer_t;

class TXMLInputStream;
class TXMLOutputStream;
class TString;

class TXMLEngine : public TObject {
   public:
      TXMLEngine();
      virtual ~TXMLEngine();
      
      Bool_t            HasAttr(XMLNodePointer_t xmlnode, const char* name);
      const char*       GetAttr(XMLNodePointer_t xmlnode, const char* name);
      Int_t             GetIntAttr(XMLNodePointer_t node, const char* name);
      XMLAttrPointer_t  NewAttr(XMLNodePointer_t xmlnode, XMLNsPointer_t,
                                const char* name, const char* value);
      XMLAttrPointer_t  NewIntAttr(XMLNodePointer_t xmlnode, const char* name, Int_t value);
      void              FreeAttr(XMLNodePointer_t xmlnode, const char* name);
      void              FreeAllAttr(XMLNodePointer_t xmlnode);
      XMLAttrPointer_t  GetFirstAttr(XMLNodePointer_t xmlnode);
      XMLAttrPointer_t  GetNextAttr(XMLAttrPointer_t xmlattr);
      const char*       GetAttrName(XMLAttrPointer_t xmlattr);
      const char*       GetAttrValue(XMLAttrPointer_t xmlattr);
      XMLNodePointer_t  NewChild(XMLNodePointer_t parent, XMLNsPointer_t ns,
                                 const char* name, const char* content = 0);
      XMLNsPointer_t    NewNS(XMLNodePointer_t xmlnode, const char* reference, const char* name = 0);
      XMLNsPointer_t    GetNS(XMLNodePointer_t xmlnode);
      const char*       GetNSName(XMLNsPointer_t ns);
      const char*       GetNSReference(XMLNsPointer_t ns);
      void              AddChild(XMLNodePointer_t parent, XMLNodePointer_t child);
      void              AddChildFirst(XMLNodePointer_t parent, XMLNodePointer_t child);
      void              UnlinkNode(XMLNodePointer_t node);
      void              FreeNode(XMLNodePointer_t xmlnode);
      void              UnlinkFreeNode(XMLNodePointer_t xmlnode);
      const char*       GetNodeName(XMLNodePointer_t xmlnode);
      const char*       GetNodeContent(XMLNodePointer_t xmlnode);
      XMLNodePointer_t  GetChild(XMLNodePointer_t xmlnode);
      XMLNodePointer_t  GetParent(XMLNodePointer_t xmlnode);
      XMLNodePointer_t  GetNext(XMLNodePointer_t xmlnode);
      void              ShiftToNext(XMLNodePointer_t &xmlnode);
      Bool_t            IsEmptyNode(XMLNodePointer_t) { return kFALSE; } // obsolete, should not be used
      void              SkipEmpty(XMLNodePointer_t &) {}                 // obsolete, should not be used
      void              CleanNode(XMLNodePointer_t xmlnode);
      XMLDocPointer_t   NewDoc(const char* version = 0);
      void              AssignDtd(XMLDocPointer_t xmldoc, const char* dtdname, const char* rootname);
      void              FreeDoc(XMLDocPointer_t xmldoc);
      void              SaveDoc(XMLDocPointer_t xmldoc, const char* filename, Int_t layout = 1);
      void              DocSetRootElement(XMLDocPointer_t xmldoc, XMLNodePointer_t xmlnode);
      XMLNodePointer_t  DocGetRootElement(XMLDocPointer_t xmldoc);
      XMLDocPointer_t   ParseFile(const char* filename);
      Bool_t            ValidateDocument(XMLDocPointer_t, Bool_t = kFALSE) { return kFALSE; } // obsolete
      void              SaveSingleNode(XMLNodePointer_t xmlnode, TString* res, Int_t layout = 1);
      XMLNodePointer_t  ReadSingleNode(const char* src);
   protected:
      char*             Makestr(const char* str);   
      char*             Makenstr(const char* start, int len);
      XMLNodePointer_t  AllocateNode(int namelen, XMLNodePointer_t parent);
      XMLAttrPointer_t  AllocateAttr(int namelen, int valuelen, XMLNodePointer_t xmlnode);
      XMLNsPointer_t    FindNs(XMLNodePointer_t xmlnode, const char* nsname);
      void              TruncateNsExtension(XMLNodePointer_t xmlnode);
      void              UnpackSpecialCharacters(char* target, const char* source, int srclen);
      void              OutputValue(Char_t* value, TXMLOutputStream* out);
      void              SaveNode(XMLNodePointer_t xmlnode, TXMLOutputStream* out, Int_t layout, Int_t level);
      XMLNodePointer_t  ReadNode(XMLNodePointer_t xmlparent, TXMLInputStream* inp, Int_t& resvalue);
      void              DisplayError(Int_t error, Int_t linenumber);
  
   ClassDef(TXMLEngine,1);   // ROOT XML I/O parser, user by TXMLFile to read/write xml files
};

#endif

