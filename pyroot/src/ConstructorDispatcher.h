// @(#)root/pyroot:$Name:  $:$Id: ConstructorDispatcher.h,v 1.1 2004/04/27 06:28:48 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_CONSTRUCTORDISPATCHER_H
#define PYROOT_CONSTRUCTORDISPATCHER_H

// ROOT
class TClass;
class TMethod;

// Bindings
#include "MethodHolder.h"


namespace PyROOT {

/** Python side ROOT constructor
      @author  WLAV
      @date    09/30/2003
      @version 1.1
 */

   class ConstructorDispatcher : public MethodHolder {
   public:
      ConstructorDispatcher( TClass*, TMethod* );

      virtual PyObject* operator()( PyObject* aTuple, PyObject* aDict );
    };

} // namespace PyROOT

#endif // !PYROOT_CONSTRUCTORDISPATCHER_H
