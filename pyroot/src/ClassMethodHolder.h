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
      @version 1.1
 */

   class ClassMethodHolder : public MethodHolder {
   public:
      ClassMethodHolder( TClass*, TMethod* );

      virtual PyObject* operator()( PyObject* aTuple, PyObject* aDict );
    };

} // namespace PyROOT

#endif // !PYROOT_CLASSMETHODHOLDER_H
