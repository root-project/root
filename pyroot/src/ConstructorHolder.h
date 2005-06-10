// @(#)root/pyroot:$Name:  $:$Id: ConstructorHolder.h,v 1.1 2005/03/04 07:44:11 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_CONSTRUCTORHOLDER_H
#define PYROOT_CONSTRUCTORHOLDER_H

// ROOT
class TClass;
class TMethod;

// Bindings
#include "MethodHolder.h"


namespace PyROOT {

/** Python side holder for ROOT constructor
      @author  WLAV
      @date    09/30/2003
      @version 2.0
 */

   class Executor;

   class ConstructorHolder : public MethodHolder {
   public:
      ConstructorHolder( TClass* klass, TMethod* method );

   public:
      virtual PyObject* GetDocString();

   public:
      virtual PyObject* operator()( ObjectProxy* self, PyObject* args, PyObject* kwds );

   protected:
      virtual bool InitExecutor_( Executor*& );
   };

} // namespace PyROOT

#endif // !PYROOT_CONSTRUCTORHOLDER_H
