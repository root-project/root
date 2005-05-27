// @(#)root/meta:$Name:  $:$Id: TClass.h,v 1.49 2005/03/20 21:25:12 brun Exp $
// Author: Markus Frank 20/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TIsaProxy
#define ROOT_TIsaProxy

#ifndef ROOT_TVirtualIsaProxy
#include "TVirtualIsaProxy.h"
#endif

class TClass;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TIsaProxy implementation class.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TIsaProxy  : public TVirtualIsaProxy {
private:
   const type_info   *fType;         //Actual typeid of the proxy
   const type_info   *fLastType;     //Last used subtype
   TClass            *fClass;        //Actual TClass
   TClass            *fLastClass;    //Last used TClass
   Char_t             fSubTypes[64]; //map of known sub-types
   Bool_t             fVirtual;      //Flag if class is virtual
   void*              fContext;      //Optional user contex
   Bool_t             fInit;         //Initialization flag
public:
   /// Standard initializing constructor
   TIsaProxy(const type_info &typ, void *ctxt = 0);
   /// Standard destructor
   virtual ~TIsaProxy();
   /// Callbacl to set the class
   virtual void SetClass(TClass *cl);
   /// IsA callback
   virtual TClass* operator()(const void *obj);
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TInstrumentedIsaProxy implementation class.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
template <class T> class TInstrumentedIsaProxy : public TVirtualIsaProxy {
private:
   TClass *fClass;        //Actual TClass

public:
   /// Standard initializing constructor
   TInstrumentedIsaProxy(TClass *cl) : fClass(cl)      {}
   /// Standard destructor
   virtual ~TInstrumentedIsaProxy()                    {}
   /// Callbacl to set the class
   virtual void SetClass(TClass *cl)                   { fClass = cl; }
   /// IsA callback
   virtual TClass* operator()(const void *obj) {
      return obj==0 ? fClass : ((T*)obj)->IsA();
   }
};

#endif // ROOT_TIsaProxy
