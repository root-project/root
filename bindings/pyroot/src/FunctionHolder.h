// Author: Wim Lavrijsen, Apr 2005

#ifndef PYROOT_TFUNCTIONHOLDER_H
#define PYROOT_TFUNCTIONHOLDER_H

// Bindings
#include "MethodHolder.h"


namespace PyROOT {

/** Python side ROOT global function
      @author  WLAV
      @date    15/03/2013
      @version 5.0
 */

   class TMemberAdapter;
   class TScopeAdapter;

   class TFunctionHolder : public TMethodHolder {
   public:
      TFunctionHolder( const TMemberAdapter& function );
      TFunctionHolder( const TScopeAdapter& scope, const TMemberAdapter& function );

      virtual PyCallable* Clone() { return new TFunctionHolder( *this ); }

      virtual PyObject* FilterArgs( ObjectProxy*& self, PyObject* args, PyObject* kwds );
      virtual PyObject* operator()( ObjectProxy*, PyObject* args, PyObject* kwds,
                                    Long_t user = 0, Bool_t release_gil = kFALSE );
   };

} // namespace PyROOT

#endif // !PYROOT_TFUNCTIONHOLDER_H
