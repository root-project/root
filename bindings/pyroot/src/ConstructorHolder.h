// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_TCONSTRUCTORHOLDER_H
#define PYROOT_TCONSTRUCTORHOLDER_H

// Bindings
#include "MethodHolder.h"


namespace PyROOT {

/** Python side holder for ROOT constructor
      @author  WLAV
      @date    15/03/2013
      @version 4.0
 */

   class TExecutor;
   class TMemberAdapter;
   class TScopeAdapter;

   class TConstructorHolder : public TMethodHolder {
   public:
      TConstructorHolder( const TScopeAdapter& klass );    // pseudo-ctors
      TConstructorHolder( const TScopeAdapter& klass, const TMemberAdapter& method );

   public:
      virtual PyObject* GetDocString();
      virtual PyCallable* Clone() { return new TConstructorHolder( *this ); }

   public:
      virtual PyObject* operator()( ObjectProxy* self, PyObject* args, PyObject* kwds,
                                    Long_t user = 0, Bool_t release_gil = kFALSE );

   protected:
      virtual Bool_t InitExecutor_( TExecutor*& );
   };

} // namespace PyROOT

#endif // !PYROOT_TCONSTRUCTORHOLDER_H
