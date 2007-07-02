// @(#)root/treeplayer:$Name:  $:$Id: TBranchProxyTemplate.h,v 1.6 2007/06/04 17:07:17 pcanal Exp $
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
   TBranchProxy *GetProxy() { return obj.GetProxy(); }   \
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
         cout << "fWhere " << obj.GetWhere() << endl;
         if (obj.GetWhere()) cout << "address? " << (T*)obj.GetWhere() << endl;
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
         cout << "obj.GetWhere() " << obj.GetWhere() << endl;
         //if (obj.GetWhere()) cout << "value? " << *(T*)obj.GetWhere() << endl;
      }

      TClaObjProxy() : obj() {};
      TClaObjProxy(TBranchProxyDirector *director, const char *name) : obj(director,name) {};
      TClaObjProxy(TBranchProxyDirector *director,  const char *top, const char *name) :
         obj(director,top,name) {};
      TClaObjProxy(TBranchProxyDirector *director,  const char *top, const char *name, const char *data) :
         obj(director,top,name,data) {};
      TClaObjProxy(TBranchProxyDirector *director, TBranchProxy *parent, const char *name) : obj(director,parent, name) {};
      ~TClaObjProxy() {};

      const TClonesArray* GetPtr() { return obj.GetPtr(); }

      Int_t GetEntries() { return obj.GetEntries(); }

      const T* At(int i) {
         static T default_val;
         if (!obj.Read()) return &default_val;
         if (obj.GetWhere()==0) return &default_val;

         T* temp = (T*)obj.GetClaStart(i);
         if (temp) return temp;
         else return &default_val;
      }

      const T* operator [](int i) { return At(i); }

   };

}

#endif
