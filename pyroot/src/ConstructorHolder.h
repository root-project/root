// @(#)root/pyroot:$Name:  $:$Id: ConstructorHolder.h,v 1.2 2005/06/10 14:30:22 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_TCONSTRUCTORHOLDER_H
#define PYROOT_TCONSTRUCTORHOLDER_H

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

   class TExecutor;

   class TConstructorHolder : public TMethodHolder {
   public:
      TConstructorHolder( TClass* klass, TMethod* method );

   public:
      virtual PyObject* GetDocString();

   public:
      virtual PyObject* operator()( ObjectProxy* self, PyObject* args, PyObject* kwds );

   protected:
      virtual Bool_t InitExecutor_( TExecutor*& );
   };

} // namespace PyROOT

#endif // !PYROOT_TCONSTRUCTORHOLDER_H
