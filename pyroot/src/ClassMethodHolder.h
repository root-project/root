// @(#)root/pyroot:$Name:  $:$Id: ClassMethodHolder.h,v 1.68 2005/01/28 05:45:41 brun Exp $
// Author: Wim Lavrijsen, Aug 2004

#ifndef PYROOT_CLASSMETHODHOLDER_H
#define PYROOT_CLASSMETHODHOLDER_H

// ROOT
class TClass;
class TMethod;

// Bindings
#include "MethodHolder.h"


namespace PyROOT {

/** Python side ROOT global/static function
      @author  WLAV
      @date    08/03/2004
      @version 2.0
 */

   class ClassMethodHolder : public MethodHolder {
   public:
      ClassMethodHolder( TClass* klass, TMethod* method );

      virtual PyObject* operator()( ObjectProxy*, PyObject* args, PyObject* kwds );
    };

} // namespace PyROOT

#endif // !PYROOT_CLASSMETHODHOLDER_H
