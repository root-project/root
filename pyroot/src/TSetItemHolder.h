// Author: Wim Lavrijsen, Oct 2005

#ifndef PYROOT_TSETITEMHOLDER_H
#define PYROOT_TSETITEMHOLDER_H

// ROOT
class TClass;
class TMethod;

// Bindings
#include "MethodHolder.h"


namespace PyROOT {

/** Special holder to allow byref return as extra argument
      @author  WLAV
      @date    10/05/2003
      @version 1.0
 */

   class TExecutor;

   class TSetItemHolder : public TMethodHolder {
   public:
      TSetItemHolder( TClass* klass, TMethod* method );

   public:
      virtual PyObject* FilterArgs( ObjectProxy*& self, PyObject* args, PyObject* kwds );

   protected:
      virtual Bool_t InitExecutor_( TExecutor*& );
   };

} // namespace PyROOT

#endif // !PYROOT_TSETITEMHOLDER_H
