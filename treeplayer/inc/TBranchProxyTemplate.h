// @(#)root/treeplayer:$Name:  $:$Id: TBranchProxyTemplate.h,v 1.2 2004/06/25 22:45:41 rdm Exp $
// Author: Philippe Canal 01/06/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBranchProxyTemplate
#define ROOT_TBranchProxyTemplate

#if !defined(__CINT__) || defined(__MAKECINT__)
#include "TBranchProxy.h"
#endif

#define InjecTBranchProxyInterface()                     \
   TBranchProxy *proxy() { return obj.proxy(); }         \
   void Reset() { obj.Reset(); }                         \
   bool Setup() { return obj.Setup(); }                  \
   bool IsInitialized() { return obj.IsInitialized(); }  \
   bool IsaPointer() const { return obj.IsaPointer(); }  \
   bool Read() { return obj.Read(); }

namespace ROOT {
   template <class T>
   class TObjProxy {
      TBranchProxy obj;
   public:
      InjecTBranchProxyInterface();

      TObjProxy() : obj() {};
      TObjProxy(TBranchProxyDirector *director, const char *name) : obj(director,name) {};
      TObjProxy(TBranchProxyDirector *director, const char *top, const char *name) :
         obj(director,top,name) {};
      TObjProxy(TBranchProxyDirector *director, const char *top, const char *name, const char *data) :
         obj(director,top,name,data) {};
      TObjProxy(TBranchProxyDirector *director, TBranchProxy *parent, const char *name) : obj(director,parent, name) {};
      ~TObjProxy() {};

      void Print() {
         obj.Print();
         cout << "fWhere " << obj.fWhere << endl;
         if (obj.fWhere) cout << "address? " << (T*)obj.fWhere << endl;
      }

      T* ptr() {
         //static T default_val;
         if (!obj.Read()) return 0; // &default_val;
         T *temp = (T*)obj.GetStart();
         // if (temp==0) return &default_val;
         return temp;
      }

      T* operator->() { return ptr(); }
      operator T*() { return ptr(); }
      // operator T&() { return *ptr(); }

   };

#if !defined(_MSC_VER) || (_MSC_VER>1300)
  template <class T, int d2 >
   class TArray2Proxy {
   public:
      TBranchProxy obj;
      InjecTBranchProxyInterface();

      TArray2Proxy() : obj() {}
      TArray2Proxy(TBranchProxyDirector *director, const char *name) : obj(director,name) {};
      TArray2Proxy(TBranchProxyDirector *director, const char *top, const char *name) :
         obj(director,top,name) {};
      TArray2Proxy(TBranchProxyDirector *director, const char *top, const char *name, const char *data) :
         obj(director,top,name,data) {};
      TArray2Proxy(TBranchProxyDirector *director, TBranchProxy *parent, const char *name) : obj(director, parent, name) {};
      ~TArray2Proxy() {};

      typedef T array_t[d2];

      void Print() {
         TBranchProxy::Print();
         cout << "fWhere " << obj.fWhere << endl;
         if (obj.fWhere) cout << "value? " << *(T*)obj.fWhere << endl;
      }

      const array_t &at(int i) {
         static array_t default_val;
         if (!obj.Read()) return default_val;
         // should add out-of bound test
         array_t *arr = 0;
         arr = (array_t*)((T*)(obj.GetStart()));
         if (arr) return arr[i];
         else return default_val;
      }

      const array_t &operator [](int i) { return at(i); }

   };
#endif

   template <class T>
   class TClaObjProxy  {
      TClaProxy obj;
   public:
      InjecTBranchProxyInterface();

      void Print() {
         obj.Print();
         cout << "obj.fWhere " << obj.fWhere << endl;
         //if (obj.fWhere) cout << "value? " << *(T*)obj.fWhere << endl;
      }

      TClaObjProxy() : obj() {};
      TClaObjProxy(TBranchProxyDirector *director, const char *name) : obj(director,name) {};
      TClaObjProxy(TBranchProxyDirector *director,  const char *top, const char *name) :
         obj(director,top,name) {};
      TClaObjProxy(TBranchProxyDirector *director,  const char *top, const char *name, const char *data) :
         obj(director,top,name,data) {};
      TClaObjProxy(TBranchProxyDirector *director, TBranchProxy *parent, const char *name) : obj(director,parent, name) {};
      ~TClaObjProxy() {};

      const T* at(int i) {
         static T default_val;
         if (!obj.Read()) return &default_val;
         if (obj.fWhere==0) return &default_val;

         T* temp = (T*)obj.GetClaStart(i);
         if (temp) return temp;
         else return &default_val;
      }

      const T* operator [](int i) { return at(i); }

   };

}

#endif
