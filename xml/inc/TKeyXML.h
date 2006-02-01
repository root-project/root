// @(#)root/xml:$Name:  $:$Id: TKeyXML.h,v 1.4 2006/01/25 16:00:11 pcanal Exp $
// Author: Sergey Linev  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TKeyXML
#define ROOT_TKeyXML

#ifndef ROOT_TXMLEngine
#include "TXMLEngine.h"
#endif
#ifndef ROOT_TKey
#include "TKey.h"
#endif

class TXMLFile;

class TKeyXML : public TKey {
   protected:
      TKeyXML();
    
   public:
      TKeyXML(TDirectory* mother, const TObject* obj, const char* name = 0);
      TKeyXML(TDirectory* mother, const void* obj, const TClass* cl, const char* name);
      TKeyXML(TDirectory* mother, XMLNodePointer_t keynode);
      virtual ~TKeyXML();

      // redefined TKey Methods
      virtual void      Delete(Option_t *option="");
      virtual void      DeleteBuffer() {}
      virtual void      FillBuffer(char *&) {}
      virtual char     *GetBuffer() const { return 0; }
      virtual Long64_t  GetSeekKey() const  { return fKeyNode ? 1024 : 0;}
      virtual Long64_t  GetSeekPdir() const { return fKeyNode ? 1024 : 0;}
      //virtual ULong_t   Hash() const { return 0; }
      virtual void      Keep() {}
      //virtual void      ls(Option_t* ="") const;
      //virtual void      Print(Option_t* ="") const {}

      virtual Int_t     Read(TObject* tobj);
      virtual TObject  *ReadObj();
      virtual void     *ReadObjectAny(const TClass *expectedClass);
      
      virtual void      ReadBuffer(char *&) {}
      virtual void      ReadFile() {}
      virtual void      SetBuffer() { fBuffer = 0; }
      virtual Int_t     WriteFile(Int_t =1, TFile* = 0) { return 0; }

      // TKeyXML specific methods

      XMLNodePointer_t  KeyNode() const { return fKeyNode; }

   protected:
      virtual Int_t     Read(const char *name) { return TKey::Read(name); }
      void              StoreObject(const void* obj, const TClass* cl);
      TXMLEngine*       XMLEngine();
      
      void*             XmlReadAny(void* obj, const TClass* expectedClass);
      
      XMLNodePointer_t  fKeyNode;  //!

   ClassDef(TKeyXML,1) // a special TKey for XML files      
};

#endif
