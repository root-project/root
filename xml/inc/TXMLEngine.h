// @(#)root/xml:$Name:  $:$Id: TXMLEngine.h,v 1.0 2004/01/28 22:31:11 brun Exp $
// Author: Sergey Linev  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TXMLEngine_h
#define TXMLEngine_h

#include "TObject.h"
#include "TString.h"

typedef void* xmlNodePointer;
typedef void* xmlNsPointer;
typedef void* xmlAttrPointer;
typedef void* xmlDocPointer;

class TXMLEngine : public TObject {
   protected:
      TXMLEngine();
      virtual ~TXMLEngine();
      static TXMLEngine* fgInstance;
      
   public:

      static TXMLEngine* GetInstance() { return fgInstance; }

      Bool_t HasProp(xmlNodePointer node, const char* name); 
        
      const char* GetProp(xmlNodePointer node, const char* name);

      xmlAttrPointer NewProp(xmlNodePointer node, 
                             xmlNsPointer ns,
                             const char* name, 
                             const char* value); 
      
      xmlNodePointer NewChild(xmlNodePointer parent, 
                              xmlNsPointer ns,
                              const char* name, 
                              const char* content = 0); 
                                  
      xmlNsPointer NewNS(xmlNodePointer node, 
                         const char* reference,
                         const char* name = 0);
                                  
      void AddChild(xmlNodePointer parent, xmlNodePointer child);
      
      void UnlinkChild(xmlNodePointer node);
      
      void FreeNode(xmlNodePointer node);
      
      const char* GetNodeName(xmlNodePointer node);
      
      const char* GetNodeContent(xmlNodePointer node);
      
      xmlNodePointer GetChild(xmlNodePointer node);
      
      xmlNodePointer GetParent(xmlNodePointer node);

      xmlNodePointer GetNext(xmlNodePointer node);
      
      void ShiftToNext(xmlNodePointer &node, Bool_t skipempty = kTRUE);
      
      void SkipEmpty(xmlNodePointer &node);
   
      xmlDocPointer NewDoc(const char* version = 0);   
      
      void AssignDtd(xmlDocPointer doc, const char* dtdname, const char* rootname);
      
      void FreeDoc(xmlDocPointer doc);
      
      void SaveDoc(xmlDocPointer doc, const char* filename, Int_t layout = 1);
      
      void DocSetRootElement(xmlDocPointer doc, xmlNodePointer node);
      
      xmlNodePointer DocGetRootElement(xmlDocPointer doc);
      
      xmlDocPointer ParseFile(const char* filename);
      
      Bool_t ValidateDocument(xmlDocPointer doc, Bool_t doout = kFALSE);
  
   protected: 
   
      TString  fStrBuf;
     
   ClassDef(TXMLEngine,1);  
};

extern TXMLEngine*  gXML;

#endif 
