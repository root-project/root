// @(#)root/pyroot:$Name:  $:$Id: ClassMethodHolder.h,v 1.2 2005/03/04 07:44:11 brun Exp $
// Author: Wim Lavrijsen, Aug 2004

#ifndef PYROOT_CLASSMETHODHOLDER_H
#define PYROOT_CLASSMETHODHOLDER_H

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

   class ClassMethodHolder : public MethodHolder {
   public:
      ClassMethodHolder( TClass* klass, TMethod* method );

      virtual PyObject* operator()( ObjectProxy*, PyObject* args, PyObject* kwds );
    };

} // namespace PyROOT

#endif // !PYROOT_CLASSMETHODHOLDER_H
