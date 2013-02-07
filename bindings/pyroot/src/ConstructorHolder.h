// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_TCONSTRUCTORHOLDER_H
#define PYROOT_TCONSTRUCTORHOLDER_H

// ROOT
class TClass;
class TMethod;

// Bindings
#include "MethodHolder.h"


namespace PyROOT {

/** Python side holder for ROOT constructor
      @author  WLAV
      @date    09/30/2003
      @version 3.0
 */

   class TExecutor;

   template< class T, class M >
   class TConstructorHolder : public TMethodHolder< T, M > {
   public:
      TConstructorHolder( const T& klass, const M& method );
      TConstructorHolder( const T& klass );

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
