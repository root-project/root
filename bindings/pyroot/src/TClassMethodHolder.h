// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Aug 2004

#ifndef PYROOT_TCLASSMETHODHOLDER_H
#define PYROOT_TCLASSMETHODHOLDER_H

// Bindings
#include "MethodHolder.h"

namespace PyROOT {

   class TMemberAdapter;
   class TScopeAdapter;

/** Python side ROOT static function
      @author  WLAV
      @date    15/03/2013
      @version 4.0
 */

   class TClassMethodHolder : public TMethodHolder {
   public:
      TClassMethodHolder( const TScopeAdapter& klass, const TMemberAdapter& method );

      virtual PyCallable* Clone() { return new TClassMethodHolder( *this ); }

      virtual PyObject* operator()( ObjectProxy*, PyObject* args, PyObject* kwds,
                                    Long_t user = 0, Bool_t release_gil = kFALSE );
   };

} // namespace PyROOT

#endif // !PYROOT_TCLASSMETHODHOLDER_H
