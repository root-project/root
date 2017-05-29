// @(#)root/meta:$Id$
// Author: Markus Frank 20/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TIsAProxy
#define ROOT_TIsAProxy

#include "TVirtualIsAProxy.h"
#include "RtypesCore.h"
#include <atomic>
#include <typeinfo>

class TClass;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TIsAProxy implementation class.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TIsAProxy  : public TVirtualIsAProxy {
private:
   template <typename T> using Atomic_t = std::atomic<T>;

   const std::type_info     *fType;        //Actual typeid of the proxy
   Atomic_t<TClass*>         fClass;       //Actual TClass
   Atomic_t<void*>           fLast;        //points into fSubTypes map for last used values
   Char_t                    fSubTypes[72];//map of known sub-types
   mutable Atomic_t<UInt_t>  fSubTypesReaders; //number of readers of fSubTypes
   Atomic_t<Bool_t>          fSubTypesWriteLockTaken; //True if there is a writer
   Bool_t                    fVirtual;     //Flag if class is virtual
   Atomic_t<Bool_t>          fInit;        //Initialization flag

   void* FindSubType(const std::type_info*) const;
   void* CacheSubType(const std::type_info*, TClass*);
protected:
   TIsAProxy(const TIsAProxy&) = delete;
   TIsAProxy& operator=(const TIsAProxy&) = delete;

public:
   // Standard initializing constructor
   TIsAProxy(const std::type_info &typ);
   // Standard destructor
   virtual ~TIsAProxy();
   // Callbacl to set the class
   virtual void SetClass(TClass *cl);
   // IsA callback
   virtual TClass* operator()(const void *obj);
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TInstrumentedIsAProxy implementation class.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
template <class T> class TInstrumentedIsAProxy : public TVirtualIsAProxy {

private:
   TClass *fClass;        //Actual TClass

protected:
   TInstrumentedIsAProxy(const TInstrumentedIsAProxy& iip) :
     TVirtualIsAProxy(iip), fClass(iip.fClass) { }
   TInstrumentedIsAProxy& operator=(const TInstrumentedIsAProxy& iip)
     {if(this!=&iip) {TVirtualIsAProxy::operator=(iip); fClass=iip.fClass;}
     return *this;}

public:
   // Standard initializing constructor
   TInstrumentedIsAProxy(TClass *cl) : fClass(cl)      {}
   // Standard destructor
   virtual ~TInstrumentedIsAProxy()                    {}
   // Callbacl to set the class
   virtual void SetClass(TClass *cl)                   { fClass = cl; }
   // IsA callback
   virtual TClass* operator()(const void *obj) {
      return obj==0 ? fClass : ((const T*)obj)->IsA();
   }
};

#endif // ROOT_TIsAProxy
