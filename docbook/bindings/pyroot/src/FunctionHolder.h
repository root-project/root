// Author: Wim Lavrijsen, Apr 2005

#ifndef PYROOT_TFUNCTIONHOLDER_H
#define PYROOT_TFUNCTIONHOLDER_H

// ROOT
class TFunction;

// Bindings
#include "MethodHolder.h"


namespace PyROOT {

/** Python side ROOT global function
      @author  WLAV
      @date    08/03/2004
      @version 4.0
 */

   template< class T, class M >
   class TFunctionHolder : public TMethodHolder< T, M > {
   public:
      TFunctionHolder( const M& function );
      TFunctionHolder( const T& scope, const M& function );

      virtual PyCallable* Clone() { return new TFunctionHolder( *this ); }

      virtual PyObject* FilterArgs( ObjectProxy*& self, PyObject* args, PyObject* kwds );
      virtual PyObject* operator()( ObjectProxy*, PyObject* args, PyObject* kwds, Long_t = 0 );
   };

} // namespace PyROOT

#endif // !PYROOT_TFUNCTIONHOLDER_H
