// @(#)root/treeplayer:$Name:  $:$Id: TBranchProxy.h,v 1.8 2005/09/03 02:21:32 pcanal Exp $
// Author: Philippe Canal 01/06/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBranchProxy
#define ROOT_TBranchProxy

#ifndef ROOT_TBranchProxyDirector
#include "TBranchProxyDirector.h"
#endif
#ifndef ROOT_TTree
#include "TTree.h"
#endif
#ifndef ROOT_TBranch
#include "TBranch.h"
#endif
#ifndef ROOT_TClonesArray
#include "TClonesArray.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_Riostream
#include "Riostream.h"
#endif
#ifndef ROOT_TError
#include "TError.h"
#endif

#include <list>
#include <algorithm>

class TBranch;
class TStreamerElement;

// Note we could protect the arrays more by introducing a class TArrayWrapper<class T> which somehow knows
// its internal dimensions and check for them ...
// template <class T> TArrayWrapper {
// public:
//    TArrayWrapper(void *where, int dim1);
//    const T operator[](int i) {
//       if (i>=dim1) return 0;
//       return where[i];
//    };
// };
// 2D array would actually be a wrapper of a wrapper i.e. has a method TArrayWrapper<T> operator[](int i);

namespace ROOT {

   class TBranchProxyHelper {
   public:
      TString fName;
      TBranchProxyHelper(const char *left,const char *right = 0) :
         fName() {
         if (left) {
            fName = left;
            if (strlen(left)&&right) fName += ".";
         }
         if (right) {
            fName += right;
         }
      }
      operator const char*() { return fName.Data(); };
   };


   class TBranchProxy {
   protected:
      TBranchProxyDirector *fDirector; // contain pointer to TTree and entry to be read

      Bool_t   fInitialized;

      const TString fBranchName;  // name of the branch to read
      TBranchProxy *fParent;      // Proxy to a parent object

      const TString fDataMember;  // name of the (eventual) data member being proxied

      const Bool_t  fIsMember;    // true if we proxy an unsplit data member
      Bool_t        fIsClone;     // true if we proxy the inside of a TClonesArray
      Bool_t        fIsaPointer;  // true if we proxy a data member of pointer type


      TString           fClassName;     // class name of the object pointed to by the branch
      TClass           *fClass;         // class name of the object pointed to by the branch
      TStreamerElement *fElement;
      Int_t             fMemberOffset;
      Int_t             fOffset;        // Offset inside the object

      TBranch *fBranch;       // branch to read
      TBranch *fBranchCount;  // eventual auxiliary branch (for example holding the size)

      TTree   *fLastTree; // TTree containing the last entry read
      Long64_t fRead;     // Last entry read

      void    *fWhere;    // memory location of the data

   public:
      virtual void Print();

      TBranchProxy();
      TBranchProxy(TBranchProxyDirector* boss, const char* top, const char* name = 0);
      TBranchProxy(TBranchProxyDirector* boss, const char *top, const char *name, const char *membername);
      TBranchProxy(TBranchProxyDirector* boss, TBranchProxy *parent, const char* membername);
      virtual ~TBranchProxy();

      TBranchProxy* GetProxy() { return this; }

      void Reset();

      Bool_t Setup();

      Bool_t IsInitialized() {
         return (fLastTree == fDirector->GetTree()) && (fLastTree);
      }

      Bool_t IsaPointer() const {
         return fIsaPointer;
      }

      Bool_t Read() {
         if (fDirector==0) return false;

         if (fDirector->GetReadEntry()!=fRead) {
            if (!IsInitialized()) {
               if (!Setup()) {
                  Error("Read",Form("Unable to initialize %s\n",fBranchName.Data()));
                  return false;
               }
            }
            if (fParent) fParent->Read();
            else {
               if (fBranchCount) {
                  fBranchCount->GetEntry(fDirector->GetReadEntry());
               }
               fBranch->GetEntry(fDirector->GetReadEntry());
            }
            fRead = fDirector->GetReadEntry();
         }
         return IsInitialized();
      }

      TClass *GetClass() {
         if (fDirector==0) return 0;
         if (fDirector->GetReadEntry()!=fRead) {
            if (!IsInitialized()) {
               if (!Setup()) {
                  return 0;
               }
            }
         }
         return fClass;
      }

      void* GetWhere() const { return fWhere; } // intentionally non-virtual

      // protected:
      virtual  void *GetStart(int /*i*/=0) {
         // return the address of the start of the object being proxied. Assumes
         // that Setup() has been called.

         if (fParent) {
            fWhere = ((unsigned char*)fParent->GetStart()) + fMemberOffset;
         }
         if (IsaPointer()) {
            if (fWhere) return *(void**)fWhere;
            else return 0;
         } else {
            return fWhere;
         }
      }

      virtual void *GetClaStart(int i=0) {
         // return the address of the start of the object being proxied. Assumes
         // that Setup() has been called.  Assumes the object containing this data
         // member is held in TClonesArray.

         void *tcaloc;
         char *location;

         if (fIsClone) {

            TClonesArray *tca;
            tca = (TClonesArray*)GetStart();

            if (tca->GetLast()<i) return 0;

            location = (char*)tca->At(i);

            return location;

         } else if (fParent) {

            tcaloc = ((unsigned char*)fParent->GetStart());
            location = (char*)fParent->GetClaStart(i);

         } else {

            tcaloc = fWhere;
            TClonesArray *tca;
            tca = (TClonesArray*)tcaloc;

            if (tca->GetLast()<i) return 0;

            location = (char*)tca->At(i);
         }

         if (location) location += fOffset;
         else return 0;

         if (IsaPointer()) {
            return *(void**)(location);
         } else {
            return location;
         }

      }
   };

   class TArrayCharProxy : public TBranchProxy {
   public:
      void Print() {
         TBranchProxy::Print();
         cout << "fWhere " << fWhere << endl;
         if (fWhere) cout << "value? " << *(unsigned char*)GetStart() << endl;
      }

      TArrayCharProxy() : TBranchProxy() {}
      TArrayCharProxy(TBranchProxyDirector *director, const char *name) : TBranchProxy(director,name) {};
      TArrayCharProxy(TBranchProxyDirector *director, const char *top, const char *name) :
         TBranchProxy(director,top,name) {};
      TArrayCharProxy(TBranchProxyDirector *director, const char *top, const char *name, const char *data) :
         TBranchProxy(director,top,name,data) {};
      TArrayCharProxy(TBranchProxyDirector *director, TBranchProxy *parent, const char *name) : TBranchProxy(director,parent, name) {};
      ~TArrayCharProxy() {};

      unsigned char At(int i) {
         static unsigned char default_val;
         if (!Read()) return default_val;
         // should add out-of bound test
         unsigned char* str = (unsigned char*)GetStart();
         return str[i];
      }

      unsigned char operator [](int i) {
         return At(i);
      }

      const char* c_str() {
         if (!Read()) return "";
         return (const char*)GetStart();
      }

      operator std::string() {
         if (!Read()) return "";
         return std::string((const char*)GetStart());
      }

   };

   class TClaProxy : public TBranchProxy {
   public:
      void Print() {
         TBranchProxy::Print();
         cout << "fWhere " << fWhere << endl;
         if (fWhere) {
            if (IsaPointer()) {
               cout << "location " << *(TClonesArray**)fWhere << endl;
            } else {
               cout << "location " << fWhere << endl;
            }
         }
      }

      TClaProxy() : TBranchProxy() {}
      TClaProxy(TBranchProxyDirector *director, const char *name) : TBranchProxy(director,name) {};
      TClaProxy(TBranchProxyDirector *director, const char *top, const char *name) :
         TBranchProxy(director,top,name) {};
      TClaProxy(TBranchProxyDirector *director, const char *top, const char *name, const char *data) :
         TBranchProxy(director,top,name,data) {};
      TClaProxy(TBranchProxyDirector *director, TBranchProxy *parent, const char *name) : TBranchProxy(director,parent, name) {};
      ~TClaProxy() {};

      const TClonesArray* GetPtr() {
         if (!Read()) return 0;
         return (TClonesArray*)GetStart();
      }

      const TClonesArray* operator->() { return GetPtr(); }

   };

   template <class T>
   class TImpProxy : public TBranchProxy {
   public:
      void Print() {
         TBranchProxy::Print();
         cout << "fWhere " << fWhere << endl;
         if (fWhere) cout << "value? " << *(T*)GetStart() << endl;
      }

      TImpProxy() : TBranchProxy() {};
      TImpProxy(TBranchProxyDirector *director, const char *name) : TBranchProxy(director,name) {};
      TImpProxy(TBranchProxyDirector *director, const char *top, const char *name) :
         TBranchProxy(director,top,name) {};
      TImpProxy(TBranchProxyDirector *director, const char *top, const char *name, const char *data) :
         TBranchProxy(director,top,name,data) {};
      TImpProxy(TBranchProxyDirector *director, TBranchProxy *parent, const char *name) : TBranchProxy(director,parent, name) {};
      ~TImpProxy() {};

      operator T() {
         if (!Read()) return 0;
         return *(T*)GetStart();
      }

      // Make sure that the copy methods are really private
#ifdef private
#undef private
#define private_was_replaced
#endif
      // For now explicitly disable copying into the value (i.e. the proxy is read-only).
   private:
      TImpProxy(T);
      TImpProxy &operator=(T);
#ifdef private_was_replaced
#define private public
#endif

   };

   template <class T>
   class TArrayProxy : public TBranchProxy {
   public:
      void Print() {
         TBranchProxy::Print();
         cout << "fWhere " << fWhere << endl;
         if (fWhere) cout << "value? " << *(T*)GetStart() << endl;
      }

      TArrayProxy() : TBranchProxy() {}
      TArrayProxy(TBranchProxyDirector *director, const char *name) : TBranchProxy(director,name) {};
      TArrayProxy(TBranchProxyDirector *director, const char *top, const char *name) :
         TBranchProxy(director,top,name) {};
      TArrayProxy(TBranchProxyDirector *director, const char *top, const char *name, const char *data) :
         TBranchProxy(director,top,name,data) {};
      TArrayProxy(TBranchProxyDirector *director, TBranchProxy *parent, const char *name) : TBranchProxy(director,parent, name) {};
      ~TArrayProxy() {};

      const T& At(int i) {
         static T default_val;
         if (!Read()) return default_val;
         // should add out-of bound test
         return ((T*)GetStart())[i];
      }

      const T& operator [](int i) {
         return At(i);
      }


   };

   template <class T, int d2, int d3 >
   class TArray3Proxy : public TBranchProxy {
   public:
      typedef T array_t[d2][d3];

      void Print() {
         TBranchProxy::Print();
         cout << "fWhere " << fWhere << endl;
         if (fWhere) cout << "value? " << *(T*)GetStart() << endl;
      }

      TArray3Proxy() : TBranchProxy() {}
      TArray3Proxy(TBranchProxyDirector *director, const char *name) : TBranchProxy(director,name) {};
      TArray3Proxy(TBranchProxyDirector *director, const char *top, const char *name) :
         TBranchProxy(director,top,name) {};
      TArray3Proxy(TBranchProxyDirector *director, const char *top, const char *name, const char *data) :
         TBranchProxy(director,top,name,data) {};
      TArray3Proxy(TBranchProxyDirector *director, TBranchProxy *parent, const char *name) : TBranchProxy(director,parent, name) {};
      ~TArray3Proxy() {};

      const array_t* At(int i) {
         static array_t default_val;
         if (!Read()) return &default_val;
         // should add out-of bound test
         return ((array_t**)GetStart())[i];
      }

      const array_t* operator [](int i) {
         return At(i);
      }

   };

   template <class T>
   class TClaImpProxy : public TBranchProxy {
   public:

      void Print() {
         TBranchProxy::Print();
         cout << "fWhere " << fWhere << endl;
         if (fWhere) cout << "value? " << *(T*)GetStart() << endl;
      }

      TClaImpProxy() : TBranchProxy() {};
      TClaImpProxy(TBranchProxyDirector *director, const char *name) : TBranchProxy(director,name) {};
      TClaImpProxy(TBranchProxyDirector *director,  const char *top, const char *name) :
         TBranchProxy(director,top,name) {};
      TClaImpProxy(TBranchProxyDirector *director,  const char *top, const char *name, const char *data) :
         TBranchProxy(director,top,name,data) {};
      TClaImpProxy(TBranchProxyDirector *director, TBranchProxy *parent, const char *name) : TBranchProxy(director,parent, name) {};
      ~TClaImpProxy() {};

      const T& At(int i) {
         static T default_val;
         if (!Read()) return default_val;
         if (fWhere==0) return default_val;

         T *temp = (T*)GetClaStart(i);

         if (temp) return *temp;
         else return default_val;

      }

      const T& operator [](int i) { return At(i); }

      // Make sure that the copy methods are really private
#ifdef private
#undef private
#define private_was_replaced
#endif
      // For now explicitly disable copying into the value (i.e. the proxy is read-only).
   private:
      TClaImpProxy(T);
      TClaImpProxy &operator=(T);
#ifdef private_was_replaced
#define private public
#endif

   };

   template <class T>
   class TClaArrayProxy : public TBranchProxy {
   public:
      typedef T* array_t;

      void Print() {
         TBranchProxy::Print();
         cout << "fWhere " << fWhere << endl;
         if (fWhere) cout << "value? " << *(T*)GetStart() << endl;
      }

      TClaArrayProxy() : TBranchProxy() {}
      TClaArrayProxy(TBranchProxyDirector *director, const char *name) : TBranchProxy(director,name) {};
      TClaArrayProxy(TBranchProxyDirector *director, const char *top, const char *name) :
         TBranchProxy(director,top,name) {};
      TClaArrayProxy(TBranchProxyDirector *director, const char *top, const char *name, const char *data) :
         TBranchProxy(director,top,name,data) {};
      TClaArrayProxy(TBranchProxyDirector *director, TBranchProxy *parent, const char *name) : TBranchProxy(director,parent, name) {};
      ~TClaArrayProxy() {};

      /* const */  array_t At(int i) {
         static T default_val;
         if (!Read()) return &default_val;
         if (fWhere==0) return &default_val;

         return (array_t)GetClaStart(i);
      }

      /* const */ array_t operator [](int i) { return At(i); }

   };

#if !defined(_MSC_VER) || (_MSC_VER>1300)
   template <class T, const int d2 >
   class TClaArray2Proxy : public TBranchProxy {
   public:
      typedef T array_t[d2];

      void Print() {
         TBranchProxy::Print();
         cout << "fWhere " << fWhere << endl;
         if (fWhere) cout << "value? " << *(T*)GetStart() << endl;
      }

      TClaArray2Proxy() : TBranchProxy() {}
      TClaArray2Proxy(TBranchProxyDirector *director, const char *name)
         : TBranchProxy(director,name) {};
      TClaArray2Proxy(TBranchProxyDirector *director, const char *top,
                      const char *name)
         : TBranchProxy(director,top,name) {};
      TClaArray2Proxy(TBranchProxyDirector *director, const char *top,
                      const char *name, const char *data)
         : TBranchProxy(director,top,name,data) {};
      TClaArray2Proxy(TBranchProxyDirector *director, TBranchProxy *parent,
                      const char *name)
         : TBranchProxy(director,parent, name) {};
      ~TClaArray2Proxy() {};

      const array_t &At(int i) {
         // might need a second param or something !?

         static array_t default_val;
         if (!Read()) return &default_val;
         if (fWhere==0) return &default_val;

         T *temp = (T*)GetClaStart(i);
         if (temp) return *temp;
         else return default_val;

         // T *temp = *(T**)location;
         // return ((array_t**)temp)[i];
      }

      const array_t &operator [](int i) { return At(i); }

   };

   template <class T, int d2, int d3 >
   class TClaArray3Proxy : public TBranchProxy {
   public:
      typedef T array_t[d2][d3];

      void Print() {
         TBranchProxy::Print();
         cout << "fWhere " << fWhere << endl;
         if (fWhere) cout << "value? " << *(T*)GetStart() << endl;
      }

      TClaArray3Proxy() : TBranchProxy() {}
      TClaArray3Proxy(TBranchProxyDirector *director, const char *name) : TBranchProxy(director,name) {};
      TClaArray3Proxy(TBranchProxyDirector *director, const char *top, const char *name) :
         TBranchProxy(director,top,name) {};
      TClaArray3Proxy(TBranchProxyDirector *director, const char *top, const char *name, const char *data) :
         TBranchProxy(director,top,name,data) {};
      TClaArray3Proxy(TBranchProxyDirector *director, TBranchProxy *parent, const char *name) : TBranchProxy(director,parent, name) {};
      ~TClaArray3Proxy() {};

      const array_t* At(int i) {
         static array_t default_val;
         if (!Read()) return &default_val;
         if (fWhere==0) return &default_val;

         T *temp = (T*)GetClaStart(i);
         if (temp) return (array_t*)temp;
         else return default_val;

         // T *temp = *(T**)location;
         // return ((array_t**)temp)[i];
      }

      const array_t* operator [](int i) { return At(i); }

   };
#endif

   //TImpProxy<TObject> d;
   typedef TImpProxy<Double_t>   TDoubleProxy;
   typedef TImpProxy<Double32_t> TDouble32Proxy;
   typedef TImpProxy<Float_t>    TFloatProxy;
   typedef TImpProxy<UInt_t>     TUIntProxy;
   typedef TImpProxy<ULong_t>    TULongProxy;
   typedef TImpProxy<ULong64_t>  TULong64Proxy;
   typedef TImpProxy<UShort_t>   TUShortProxy;
   typedef TImpProxy<UChar_t>    TUCharProxy;
   typedef TImpProxy<Int_t>      TIntProxy;
   typedef TImpProxy<Long_t>     TLongProxy;
   typedef TImpProxy<Long64_t>   TLong64Proxy;
   typedef TImpProxy<Short_t>    TShortProxy;
   typedef TImpProxy<Char_t>     TCharProxy;
   typedef TImpProxy<Bool_t>     TBoolProxy;

   typedef TArrayProxy<Double_t>   TArrayDoubleProxy;
   typedef TArrayProxy<Double32_t> TArrayDouble32Proxy;
   typedef TArrayProxy<Float_t>    TArrayFloatProxy;
   typedef TArrayProxy<UInt_t>     TArrayUIntProxy;
   typedef TArrayProxy<ULong_t>    TArrayULongProxy;
   typedef TArrayProxy<ULong64_t>  TArrayULong64Proxy;
   typedef TArrayProxy<UShort_t>   TArrayUShortProxy;
   typedef TArrayProxy<UChar_t>    TArrayUCharProxy;
   typedef TArrayProxy<Int_t>      TArrayIntProxy;
   typedef TArrayProxy<Long_t>     TArrayLongProxy;
   typedef TArrayProxy<Long64_t>   TArrayLong64Proxy;
   typedef TArrayProxy<UShort_t>   TArrayShortProxy;
   //specialized ! typedef TArrayProxy<Char_t>  TArrayCharProxy;
   typedef TArrayProxy<Bool_t>     TArrayBoolProxy;

   typedef TClaImpProxy<Double_t>   TClaDoubleProxy;
   typedef TClaImpProxy<Double32_t> TClaDouble32Proxy;
   typedef TClaImpProxy<Float_t>    TClaFloatProxy;
   typedef TClaImpProxy<UInt_t>     TClaUIntProxy;
   typedef TClaImpProxy<ULong_t>    TClaULongProxy;
   typedef TClaImpProxy<ULong64_t>  TClaULong64Proxy;
   typedef TClaImpProxy<UShort_t>   TClaUShortProxy;
   typedef TClaImpProxy<UChar_t>    TClaUCharProxy;
   typedef TClaImpProxy<Int_t>      TClaIntProxy;
   typedef TClaImpProxy<Long_t>     TClaLongProxy;
   typedef TClaImpProxy<Long64_t>   TClaLong64Proxy;
   typedef TClaImpProxy<Short_t>    TClaShortProxy;
   typedef TClaImpProxy<Char_t>     TClaCharProxy;
   typedef TClaImpProxy<Bool_t>     TClaBoolProxy;

   typedef TClaArrayProxy<Double_t>    TClaArrayDoubleProxy;
   typedef TClaArrayProxy<Double32_t>  TClaArrayDouble32Proxy;
   typedef TClaArrayProxy<Float_t>     TClaArrayFloatProxy;
   typedef TClaArrayProxy<UInt_t>      TClaArrayUIntProxy;
   typedef TClaArrayProxy<ULong_t>     TClaArrayULongProxy;
   typedef TClaArrayProxy<ULong64_t>   TClaArrayULong64Proxy;
   typedef TClaArrayProxy<UShort_t>    TClaArrayUShortProxy;
   typedef TClaArrayProxy<UChar_t>     TClaArrayUCharProxy;
   typedef TClaArrayProxy<Int_t>       TClaArrayIntProxy;
   typedef TClaArrayProxy<Long_t>      TClaArrayLongProxy;
   typedef TClaArrayProxy<Long64_t>    TClaArrayLong64Proxy;
   typedef TClaArrayProxy<UShort_t>    TClaArrayShortProxy;
   typedef TClaArrayProxy<Char_t>      TClaArrayCharProxy;
   typedef TClaArrayProxy<Bool_t>      TClaArrayBoolProxy;
   //specialized ! typedef TClaArrayProxy<Char_t>  TClaArrayCharProxy;

} // namespace ROOT

#endif

