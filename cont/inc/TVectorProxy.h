// @(#)root/cont:$Name:  $:$Id: TVectorProxy.h,v 1.7 2004/02/19 23:47:40 brun Exp $
// Author: Philippe Canal 20/08/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVectorProxy                                                         //
//                                                                      //
// Proxy around a stl vector                                            //
//                                                                      //
// In particular this is used to implement splitting,                   //
// and TTreeFormula access to STL vector                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef Root__TVectorProxy_h
#define Root__TVectorProxy_h

#include "TVirtualCollectionProxy.h"
#include "TClassEdit.h"
#include "TDataType.h"

namespace ROOT {

   template <class vec> class TVectorProxy : public TVirtualCollectionProxy {

      typedef typename vec::value_type nested;
      typedef std::vector<void*> ProxyList_t;
      
      vec*        fProxied;
      ProxyList_t fProxyList;
      
      TClass      *fValueClass;  //! TClass of object in collection
      EDataType    fType;        //! Type of the content (see TDataType).
      
   public:
      TVirtualCollectionProxy* Generate() const  { return new TVectorProxy<vec>(); }
      TVectorProxy() : fValueClass(0), fType(kNoType_t) {}
      
      void    SetProxy(void *objstart) { fProxied = (vec*)objstart; }
      void    PushProxy(void *objstart) { 
	 fProxyList.push_back(fProxied);
	 fProxied = (vec*)objstart;
      }
      void    PopProxy() { fProxied = (vec*)fProxyList.back(); fProxyList.pop_back(); } 

      UInt_t  Sizeof() const { return sizeof(vec); }

      virtual TClass *GetCollectionClass() { 
         // Return a pointer to the TClass representing the container
         if (fClass==0) { fClass=gROOT->GetClass(typeid(vec)); } 
         return fClass; 
      }
      virtual EDataType GetType() {
         // If the content is a simple numerical value, return its type (see TDataType)
         if (fType!=kNoType_t) return fType;

         if (GetValueClass()) fType = kOther_t;
         else {
            fType = (EDataType)TDataType::GetType(typeid(nested));
            if (fType==kOther_t || fType == kDouble_t) {
               // we could have a Double_t or a Double32_t!

               TClass *cl = GetCollectionClass();
               if (cl==0) return fType;

               std::string shortname = TClassEdit::ShortType(cl->GetName(),
                                                             TClassEdit::kDropAlloc);
               std::string inside = TClassEdit::ShortType(shortname.c_str(), 
                                                          TClassEdit::kInnerClass);
               TDataType *fundType = gROOT->GetType( inside.c_str() );
               if (fundType) fType = (EDataType)fundType->GetType();
            }
         }
         return fType;
      }

                    
      void   *At(UInt_t idx) {
         // Return the address of the value at index 'idx'
         
         if (!fProxied) return 0;
         return &( (*fProxied)[idx] );
      }
      
      void Clear(const char *opt){
         if (!Size()) return;

         Int_t force = 1;
         if (!opt || strstr(opt,"f")==0) force = 0;
         if (! force ) {
            fProxied->clear();
         } else {
            // We could try to write an optimization to prevent the destructor 
            // from begin called.
            fProxied->clear();
         }
         
         return;      
      }
      
      TClass *GetValueClass() { 
         if (fValueClass==0) {
            fValueClass = GetClass((nested*)0);
#ifdef R__NO_CLASS_TEMPLATE_SPECIALIZATION 
            // In case where the template mechanism is basically broken. 
            if (fValueClass==0) {
               TClass *cl = GetCollectionClass();
               if (cl==0) return 0;

               std::string shortname = TClassEdit::ShortType(cl->GetName(),
                                                             TClassEdit::kDropAlloc);
               std::string inside = TClassEdit::ShortType(shortname.c_str(), 
                                                          TClassEdit::kInnerClass);
               inside = TClassEdit::ShortType(inside.c_str(), TClassEdit::kDropTrailStar);
               fValueClass = TClass::GetClass( inside.c_str() );
            }
#endif
         }
         return fValueClass; 
      }
      
      Bool_t HasPointers() const {
#ifndef R__NO_CLASS_TEMPLATE_SPECIALIZATION
         return ROOT::IsPointer<nested>::val == 1;
#else
         // return ROOT::IsPointer( (nested *) 0x0 );         
         TVectorProxy *This = const_cast<TVectorProxy*>(this);
         TClass *cl = This->GetCollectionClass();
         if (cl==0) return 0;
         
         std::string shortname = TClassEdit::ShortType(cl->GetName(),
                                                       TClassEdit::kDropAlloc);
         std::string inside = TClassEdit::ShortType(shortname.c_str(), 
                                                    TClassEdit::kInnerClass);
         return (inside[inside.size()-1]=='*');
#endif
      }
      
      void Resize(UInt_t n, Bool_t forceDelete) {
         // Resize the container

         UInt_t nold = Size();
         
         //  We could implement a strategy to avoid calling desctructors
         //  if forceDelete is false
         //    if (n<nold) Destruct(n-1,nold,forceDelete);
         //    else        Clear(forceDelete?"f":"");
         
         fProxied->resize(n);
         
         if (n < nold)  return;
         
         void *obj,**abj;
         if (HasPointers()) {
            for (UInt_t idx=nold; idx<n; ++idx) {
               obj = At(idx);
               abj = &obj;
               *abj = 0;
               if (forceDelete) *abj = fValueClass->New();
            }
         }
      }
      
      UInt_t  Size() const { return fProxied ? (*fProxied).size() : 0; }

      void    Streamer(TBuffer &b) { GetCollectionClass()->Streamer( fProxied, b ); }
   };

   template <class vec> class TBoolVectorProxy : public TVirtualCollectionProxy {

      typedef typename vec::value_type nested;
      typedef std::vector<void*> ProxyList_t;
     
      vec*        fProxied;
      ProxyList_t fProxyList;
       
      struct boolholder {
         bool val;
      };

   public:
      TVirtualCollectionProxy* Generate() const  { return new TBoolVectorProxy<vec>(); }
      TBoolVectorProxy() {}
      
      void    SetProxy(void *objstart) { fProxied = (vec*)objstart; }
      void    PushProxy(void *objstart) { 
	 fProxyList.push_back(fProxied);
	 fProxied = (vec*)objstart;
      }
      void    PopProxy() { fProxied = (vec*)fProxyList.back(); fProxyList.pop_back(); } 
      UInt_t  Sizeof() const { return sizeof(vec); }
      virtual TClass *GetCollectionClass() { 
         // Return a pointer to the TClass representing the container
         if (fClass==0) { fClass=gROOT->GetClass(typeid(vec)); } 
         return fClass; 
      }
      virtual EDataType GetType() {
         return kBool_t;
      }

      void   *At(UInt_t /* idx */) {
         // Return the address of the value at index 'idx'
         
         Fatal("At","At is not useable for a proxy of vector<bool>");
         return 0x0;
      }
      
      void Clear(const char *){ fProxied->clear();  }
      
      TClass *GetValueClass() { return 0; }      
      Bool_t HasPointers() const { return 0; }      
      void Resize(UInt_t n, Bool_t ) { fProxied->resize(n); }
      
      UInt_t  Size() const { return fProxied ? (*fProxied).size() : 0; }
      void    Streamer(TBuffer &b) { GetCollectionClass()->Streamer( fProxied, b ); }
   };


}

#endif // Root_TVectorProxy_h
