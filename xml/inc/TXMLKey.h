// @(#)root/xml:$Name:  $:$Id: TXMLKey.h,v 1.3 2004/05/14 14:30:46 brun Exp $
// Author: Sergey Linev  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXMLKey
#define ROOT_TXMLKey

#ifndef ROOT_TXMLEngine
#include "TXMLEngine.h"
#endif
#ifndef ROOT_TKey
#include "TKey.h"
#endif


class TXMLFile;

class TXMLKey : public TKey {
   protected:
      TXMLKey();
    
   public:
      TXMLKey(TXMLFile* file, const TObject* obj, const char* name = 0);
      TXMLKey(TXMLFile* file, const void* obj, const TClass* cl, const char* name);
      TXMLKey(TXMLFile* file, xmlNodePointer keynode);
      virtual ~TXMLKey();

      // redefined TKey Methods
      virtual void      Browse(TBrowser *b);
      virtual void      Delete(Option_t *option="");
      virtual void      DeleteBuffer() {}
      virtual void      FillBuffer(char *&) {}
      virtual char     *GetBuffer() const { return 0; }
      virtual Long64_t  GetSeekKey() const  { return 1; }
      virtual Long64_t  GetSeekPdir() const { return 1;}
      //virtual ULong_t   Hash() const { return 0; }
      virtual void      Keep() {}
      //virtual void      ls(Option_t* ="") const;
      //virtual void      Print(Option_t* ="") const {}

      virtual Int_t     Read(TObject*) { return 0; }
      virtual TObject  *ReadObj();
      virtual void     *ReadObjectAny(const TClass *cl);
      
      virtual void      ReadBuffer(char *&) {}
      virtual void      ReadFile() {}
      virtual void      SetBuffer() { fBuffer = 0; }
      virtual void      SetParent(const TObject* ) { }
      virtual Int_t     Sizeof() const { return 0; }
      virtual Int_t     WriteFile(Int_t =1) { return 0; }

      // TXMLKey specific methods

      xmlNodePointer    KeyNode() const { return fKeyNode; }
      void              SetXML(TXMLEngine* xml) { fXML = xml; }


   protected:
      virtual Int_t     Read(const char *name) { return TKey::Read(name); }
      void              StoreObject(const void* obj, const TClass* cl);
      xmlNodePointer    ObjNode();
      xmlNodePointer    BlockNode();
      
      TXMLFile*         fFile;     //!
      TXMLEngine*       fXML;      //!
      xmlNodePointer    fKeyNode;  //!

   ClassDef(TXMLKey,1) // a special TKey for XML files      
};




#endif
