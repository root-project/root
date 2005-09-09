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
      @version 2.0
 */

   class TFunctionHolder : public TMethodHolder {
   public:
      TFunctionHolder( TFunction* function );

      virtual PyObject* operator()( ObjectProxy*, PyObject* args, PyObject* kwds );
   };

} // namespace PyROOT

#endif // !PYROOT_TFUNCTIONHOLDER_H
