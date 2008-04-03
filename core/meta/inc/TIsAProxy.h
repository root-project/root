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

#ifndef ROOT_TVirtualIsAProxy
#include "TVirtualIsAProxy.h"
#endif
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif


class TClass;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TIsAProxy implementation class.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TIsAProxy  : public TVirtualIsAProxy {

private:
   const type_info   *fType;         //Actual typeid of the proxy
   const type_info   *fLastType;     //Last used subtype
   TClass            *fClass;        //Actual TClass
   TClass            *fLastClass;    //Last used TClass
   Char_t             fSubTypes[72]; //map of known sub-types
   Bool_t             fVirtual;      //Flag if class is virtual
   void              *fContext;      //Optional user contex
   Bool_t             fInit;         //Initialization flag

protected:
   TIsAProxy(const TIsAProxy&);
   TIsAProxy& operator=(const TIsAProxy&);

public:
   // Standard initializing constructor
   TIsAProxy(const type_info &typ, void *ctxt = 0);
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
      return obj==0 ? fClass : ((T*)obj)->IsA();
   }
};

#endif // ROOT_TIsAProxy
