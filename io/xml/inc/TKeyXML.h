// @(#)root/xml:$Id$
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

private:
   TKeyXML(const TKeyXML&);            // TKeyXML objects are not copiable.
   TKeyXML& operator=(const TKeyXML&); // TKeyXML objects are not copiable.

protected:
   TKeyXML();

public:
   TKeyXML(TDirectory* mother, Long64_t keyid, const TObject* obj, const char* name = 0, const char* title = 0);
   TKeyXML(TDirectory* mother, Long64_t keyid, const void* obj, const TClass* cl, const char* name, const char* title = 0);
   TKeyXML(TDirectory* mother, Long64_t keyid, XMLNodePointer_t keynode);
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
   virtual TObject  *ReadObjWithBuffer(char *bufferRead);
   virtual void     *ReadObjectAny(const TClass *expectedClass);

   virtual void      ReadBuffer(char *&) {}
   virtual Bool_t    ReadFile() { return kTRUE; }
   virtual void      SetBuffer() { fBuffer = 0; }
   virtual Int_t     WriteFile(Int_t =1, TFile* = 0) { return 0; }

   // TKeyXML specific methods

   XMLNodePointer_t  KeyNode() const { return fKeyNode; }
   Long64_t          GetKeyId() const { return fKeyId; }
   Bool_t            IsSubdir() const { return fSubdir; }
   void              SetSubir() { fSubdir = kTRUE; }
   void              UpdateObject(TObject* obj);
   void              UpdateAttributes();

protected:
   virtual Int_t     Read(const char *name) { return TKey::Read(name); }
   void              StoreObject(const void* obj, const TClass* cl);
   void              StoreKeyAttributes();
   TXMLEngine*       XMLEngine();

   void*             XmlReadAny(void* obj, const TClass* expectedClass);

   XMLNodePointer_t  fKeyNode;  //! node with stored object
   Long64_t          fKeyId;    //! unique identifier of key for search methods
   Bool_t            fSubdir;   //! indicates that key contains subdirectory

   ClassDef(TKeyXML,1) // a special TKey for XML files
};

#endif
