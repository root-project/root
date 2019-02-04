// @(#)root/treeplayer:$Id$
// Author: Philippe Canal 01/06/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// NOTE: This header is only used by selectors, to verify that the selector
//       source matches the ROOT interface. It should not end up in the
//       dictionary nor in the PCH, so it should not be added to the list
//       of headers of the TreePlayer library.

#ifndef ROOT_TBranchProxyTemplate
#define ROOT_TBranchProxyTemplate

#if R__BRANCHPROXY_GENERATOR_VERSION != 2
// Generated source and branch proxy interface are out of sync.
# error "Please regenerate this file using TTree::MakeProxy()!"
#endif

#include "TBranchProxy.h"

#define InjecTBranchProxyInterface()                     \
   ROOT::Detail::TBranchProxy *GetProxy() { return obj.GetProxy(); }   \
   void Reset() { obj.Reset(); }                         \
   bool Setup() { return obj.Setup(); }                  \
   bool IsInitialized() { return obj.IsInitialized(); }  \
   bool IsaPointer() const { return obj.IsaPointer(); }  \
   bool Read() { return obj.Read(); }

namespace ROOT {
namespace Internal {
   template <class T>
   class TObjProxy {
      Detail::TBranchProxy obj;
   public:
      InjecTBranchProxyInterface();

      TObjProxy() : obj() {};
      TObjProxy(TBranchProxyDirector *director, const char *name) : obj(director,name) {};
      TObjProxy(TBranchProxyDirector *director, const char *top, const char *name) :
         obj(director,top,name) {};
      TObjProxy(TBranchProxyDirector *director, const char *top, const char *name, const char *data) :
         obj(director,top,name,data) {};
      TObjProxy(TBranchProxyDirector *director, Detail::TBranchProxy *parent, const char *name, const char* top = 0, const char* mid = 0) :
         obj(director,parent, name, top, mid) {};
      ~TObjProxy() {};

      Int_t GetOffset() { return obj.GetOffset(); }

      void Print() {
         obj.Print();
         std::cout << "fWhere " << obj.GetWhere() << std::endl;
         if (obj.GetWhere()) std::cout << "address? " << (T*)obj.GetWhere() << std::endl;
      }

      T* GetPtr() {
         //static T default_val;
         if (!obj.Read()) return 0; // &default_val;
         T *temp = (T*)obj.GetStart();
         // if (temp==0) return &default_val;
         return temp;
      }

      T* operator->() { return GetPtr(); }
      operator T*() { return GetPtr(); }
      // operator T&() { return *GetPtr(); }

   };

   template <class T>
   class TClaObjProxy  {
      TClaProxy obj;
   public:
      InjecTBranchProxyInterface();

      void Print() {
         obj.Print();
         std::cout << "obj.GetWhere() " << obj.GetWhere() << std::endl;
         //if (obj.GetWhere()) std::cout << "value? " << *(T*)obj.GetWhere() << std::endl;
      }

      TClaObjProxy() : obj() {};
      TClaObjProxy(TBranchProxyDirector *director, const char *name) : obj(director,name) {};
      TClaObjProxy(TBranchProxyDirector *director,  const char *top, const char *name) :
         obj(director,top,name) {};
      TClaObjProxy(TBranchProxyDirector *director,  const char *top, const char *name, const char *data) :
         obj(director,top,name,data) {};
      TClaObjProxy(TBranchProxyDirector *director, Detail::TBranchProxy *parent, const char *name, const char* top = 0, const char* mid = 0) :
         obj(director,parent, name, top, mid) {};
      ~TClaObjProxy() {};

      const TClonesArray* GetPtr() { return obj.GetPtr(); }

      Int_t GetEntries() { return obj.GetEntries(); }

      const T* At(UInt_t i) {
         static T default_val;
         if (!obj.Read()) return &default_val;
         if (obj.GetWhere()==0) return &default_val;

         T* temp = (T*)obj.GetClaStart(i);
         if (temp) return temp;
         else return &default_val;
      }

      const T* operator [](Int_t i) { return At(i); }
      const T* operator [](UInt_t i) { return At(i); }

   };

   template <class T>
   class TStlObjProxy  {
      TStlProxy obj;
      typedef T value_t;
   public:
      InjecTBranchProxyInterface();

      void Print() {
         obj.Print();
         std::cout << "obj.GetWhere() " << obj.GetWhere() << std::endl;
         //if (obj.GetWhere()) std::cout << "value? " << *(T*)obj.GetWhere() << std::endl;
      }

      TStlObjProxy() : obj() {};
      TStlObjProxy(TBranchProxyDirector *director, const char *name) : obj(director,name) {};
      TStlObjProxy(TBranchProxyDirector *director,  const char *top, const char *name) :
         obj(director,top,name) {};
      TStlObjProxy(TBranchProxyDirector *director,  const char *top, const char *name, const char *data) :
         obj(director,top,name,data) {};
      TStlObjProxy(TBranchProxyDirector *director, Detail::TBranchProxy *parent, const char *name, const char* top = 0, const char* mid = 0) :
         obj(director,parent, name, top, mid) {};
      ~TStlObjProxy() {};

      TVirtualCollectionProxy* GetCollection() {
         return obj.GetPtr();
      }

      Int_t GetEntries() { return obj.GetEntries(); }

      const value_t& At(UInt_t i) {
         static const value_t default_val;
         if (!obj.Read()) return default_val;
         if (obj.GetWhere()==0) return default_val;

         value_t *temp = (value_t*)obj.GetStlStart(i);
         if (temp) return *temp;
         else return default_val;
      }

      const value_t& operator [](Int_t i) { return At(i); }
      const value_t& operator [](UInt_t i) { return At(i); }

   };


   template <class T>
   class TStlSimpleProxy : TObjProxy<T> {
      // Intended to compiled non-split collection

      TVirtualCollectionProxy *fCollection;
      typedef typename T::value_type value_t;
   public:

      TStlSimpleProxy() : TObjProxy<T>(),fCollection(0) {};
      TStlSimpleProxy(TBranchProxyDirector *director, const char *name) :  TObjProxy<T>(director,name),fCollection(0) {};
      TStlSimpleProxy(TBranchProxyDirector *director,  const char *top, const char *name) :
         TObjProxy<T>(director,top,name),fCollection(0) {};
      TStlSimpleProxy(TBranchProxyDirector *director,  const char *top, const char *name, const char *data) :
         TObjProxy<T>(director,top,name,data),fCollection(0) {};
      TStlSimpleProxy(TBranchProxyDirector *director, Detail::TBranchProxy *parent, const char *name, const char* top = 0, const char* mid = 0) :
          TObjProxy<T>(director,parent, name, top, mid),fCollection(0) {};
      ~TStlSimpleProxy() { delete fCollection; };

      TVirtualCollectionProxy* GetCollection() {
         if (fCollection==0) {
            TClass *cl = TClass::GetClass<T>();
            if (cl && cl->GetCollectionProxy()) {
               fCollection =  cl->GetCollectionProxy()->Generate();
            }
         }
         return fCollection;
      }

      Int_t GetEntries() {
         T *temp =   TObjProxy<T>::GetPtr();
         if (temp) {
            GetCollection();
            if (!fCollection) return 0;
            TVirtualCollectionProxy::TPushPop helper( fCollection, temp );
            return fCollection->Size();
         }
         return 0;
      }

      const value_t At(UInt_t i) {
         static value_t default_val;
         T *temp =  TObjProxy<T>::GetPtr();
         if (temp) {
            GetCollection();
            if (!fCollection) return 0;
            TVirtualCollectionProxy::TPushPop helper( fCollection, temp );
            return *(value_t*)(fCollection->At(i));
         }
         else return default_val;
      }

      const value_t operator [](Int_t i) { return At(i); }
      const value_t operator [](UInt_t i) { return At(i); }

      T* operator->() { return  TObjProxy<T>::GetPtr(); }
      operator T*() { return  TObjProxy<T>::GetPtr(); }
      // operator T&() { return *GetPtr(); }

   };

} // namespace Internal
} // namespace ROOT

#endif
