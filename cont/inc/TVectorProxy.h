// @(#)root/cont:$Name:  $:$Id: TVectorProxy.h,v 1.5 2004/02/18 07:28:02 brun Exp $
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
      
      vec* fProxied;
      
      TClass      *fValueClass;  //! TClass of object in collection
      EDataType    fType;        //! Type of the content (see TDataType).
      UInt_t       fNarr;        //! Allocated size of fArr
      void       **fArr;         //! [fNarr] Implementing GetPtrArray
      
   public:
      TVirtualCollectionProxy* Generate() const  { return new TVectorProxy<vec>(); }
      TVectorProxy() : fValueClass(0), fType(kNoType_t), fNarr(0), fArr(0) {}
      
      void    SetProxy(void *objstart) { fProxied = (vec*)objstart; }

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

      void  **GetPtrArray() {
         // Return a contiguous array of pointer to the values in the container.

         if (gDebug>1) Info("TVectorProxy::GetPtrArray","called for %s at %p",GetCollectionClass()->GetName(),fProxied);
         if (HasPointers()) return (void**)At(0);
         
         unsigned int n = Size();
         if (n >= fNarr) {
            if (gDebug>3) Info("TVectorProxy::GetPtrArray","Resize cache-array  for %s at %p",GetCollectionClass()->GetName(),fProxied);
            delete [] fArr;
            // Note: heuristic for the increase in size.
            fNarr =  int(n*1.3) + 10;
            fArr  = new void*[fNarr+1];
            fArr[0] = 0;
            fArr[n-1] = 0;
            fArr[n] = 0;
         }
         
         if (n==0) { fArr[0]=0; return fArr; }

         if (fArr[0]==At(0) 
             && fArr[n-1]==At(n-1)) {
            if (gDebug>3) Info("TVectorProxy::GetPtrArray","Keeping old addresses for n==%d fArr[0]==%p fArr[n-1]==%p",n,fArr[0],fArr[n-1]);
            //fArr[n]=0;
            return fArr;
         }
         fArr[0] = At(0);
         Int_t valSize = sizeof(nested);
         for (unsigned int i=1;i<n;i++)   { fArr[i] = (char*)(fArr[i-1]) + valSize; }

         if (gDebug>3) Info("TVectorProxy::GetPtrArray","Init addresses for n==%d fArr[0]==%p fArr[n-1]==%p",n,fArr[0],fArr[n-1]);
         
         fArr[n]=0;
         return fArr;
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
               fprintf(stderr,"looking up %s\n",inside.c_str() );
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
         return ROOT::IsPointer( (nested *) 0x0 );
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
      
      vec* fProxied;
      
      struct boolholder {
         bool val;
      };

      UInt_t       fNarr;        //! Allocated size of fArr
      boolholder **fArr;         //! [fNarr] Implementing GetPtrArray
      
   public:
      TVirtualCollectionProxy* Generate() const  { return new TBoolVectorProxy<vec>(); }
      TBoolVectorProxy() : fNarr(0), fArr(0) {}
      
      void    SetProxy(void *objstart) { fProxied = (vec*)objstart; }
      UInt_t  Sizeof() const { return sizeof(vec); }
      virtual TClass *GetCollectionClass() { 
         // Return a pointer to the TClass representing the container
         if (fClass==0) { fClass=gROOT->GetClass(typeid(vec)); } 
         return fClass; 
      }
      virtual EDataType GetType() {
         return kBool_t;
      }

      void  **GetPtrArray() {
         // Return a contiguous array of pointer to the values in the container.

         if (gDebug>1) Info("TBoolVectorProxy::GetPtrArray","called for %s at %p",GetCollectionClass()->GetName(),fProxied);

         unsigned int n = Size();
         if (n >= fNarr) {
            for (unsigned int k=0;k<fNarr;++k) {
               delete fArr[k];
            }
            delete [] fArr;
            fNarr =  int(n*1.3) + 10;
            fArr  = new boolholder*[fNarr+1];
            for (unsigned int l=0;l<fNarr;++l) {
               fArr[l] = new boolholder;
            }
         }
         
         for (unsigned int i=0;i<n;i++) { 
            fArr[i]->val = (*fProxied)[i]; 
         }

         return (void**)fArr;
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
