// @(#)root/pyroot:$Name:  $:$Id: ClassMethodHolder.h,v 1.3 2005/04/13 05:04:49 brun Exp $
// Author: Wim Lavrijsen, Aug 2004

#ifndef PYROOT_TCLASSMETHODHOLDER_H
#define PYROOT_TCLASSMETHODHOLDER_H

// ROOT
class TClass;
class TMethod;

// Bindings
#include "MethodHolder.h"


namespace PyROOT {

/** Python side ROOT static function
      @author  WLAV
      @date    08/03/2004
      @version 2.0
 */

   class TClassMethodHolder : public TMethodHolder {
   public:
      TClassMethodHolder( TClass* klass, TMethod* method );

      virtual PyObject* operator()( ObjectProxy*, PyObject* args, PyObject* kwds );
    };

} // namespace PyROOT

#endif // !PYROOT_TCLASSMETHODHOLDER_H
