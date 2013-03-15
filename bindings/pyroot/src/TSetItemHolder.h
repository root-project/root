// Author: Wim Lavrijsen, Oct 2005

#ifndef PYROOT_TSETITEMHOLDER_H
#define PYROOT_TSETITEMHOLDER_H

// Bindings
#include "MethodHolder.h"


namespace PyROOT {

/** Special holder to allow byref return as extra argument
      @author  WLAV
      @date    15/03/2013
      @version 3.0
 */

   class TExecutor;
   class TMemberAdapter;
   class TScopeAdapter;

   class TSetItemHolder : public TMethodHolder {
   public:
      TSetItemHolder( const TScopeAdapter& klass, const TMemberAdapter& method );

   public:
      virtual PyCallable* Clone() { return new TSetItemHolder( *this ); }

      virtual PyObject* FilterArgs( ObjectProxy*& self, PyObject* args, PyObject* kwds );

   protected:
      virtual Bool_t InitExecutor_( TExecutor*& );
   };

} // namespace PyROOT

#endif // !PYROOT_TSETITEMHOLDER_H
