// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_TCONSTRUCTORHOLDER_H
#define PYROOT_TCONSTRUCTORHOLDER_H

// Bindings
#include "TMethodHolder.h"


namespace PyROOT {

   class TConstructorHolder : public TMethodHolder {
   public:
      using TMethodHolder::TMethodHolder;

   public:
      PyObject* GetDocString() override;
      PyCallable* Clone() override { return new TConstructorHolder( *this ); }

   public:
      PyObject* Call(
         ObjectProxy*& self, PyObject* args, PyObject* kwds, TCallContext* ctxt = 0 ) override;

   protected:
      Bool_t InitExecutor_( TExecutor*&, TCallContext* ctxt = 0 ) override;
   };

} // namespace PyROOT

#endif // !PYROOT_TCONSTRUCTORHOLDER_H
