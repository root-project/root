// @(#)root/cont:$Name:  $:$Id:  $
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

namespace ROOT {

   template <class vec> class TVectorProxy : public TVirtualCollectionProxy {

      typedef typename vec::value_type nested;
      
      vec* fProxied;
      
      TClass      *fValueClass;  //! TClass of object in collection
      UInt_t       fNarr;        //! Allocated size of fArr
      void       **fArr;         //! [fNarr] Implementing GetPtrArray
      
   public:
      TVirtualCollectionProxy* Generate() const  { return new TVectorProxy<vec>(); }
      TVectorProxy() : fValueClass(0),fNarr(0), fArr(0) {}
      
      void    SetProxy(void *objstart) { fProxied = (vec*)objstart; }

      UInt_t  Sizeof() const { return sizeof(vec); }

      virtual TClass *GetCollectionClass() { 
         // Return a pointer to the TClass representing the container
         if (fClass==0) { fClass=gROOT->GetClass(typeid(vec)); } 
         return fClass; 
      }

      void  **GetPtrArray() {
         // Return a contiguous array of pointer to the values in the container.

         if (gDebug>1) Info("TVectorProxy::GetPtrArray","called for %s at %p",GetCollectionClass()->GetName(),fProxied);
         if (HasPointers()) return (void**)At(0);
         
         unsigned int n = Size();
         if (n >= fNarr) {
            delete [] fArr;
            fNarr =  int(n*1.3) + 10;
            fArr  = new void*[fNarr+1];
         }
         
         fArr[0] = At(0);
         Int_t valSize = sizeof(nested);
         for (unsigned int i=1;i<n;i++)   { fArr[i] = (char*)(fArr[i-1]) + valSize;}
         
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


}

#endif // Root_TVectorProxy_h
