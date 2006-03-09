// @(#)root/pyroot:$Name:  $:$Id: PyCallable.h,v 1.2 2005/03/04 07:44:11 brun Exp $
// Author: Wim Lavrijsen, Aug 2004

#ifndef PYROOT_PYCALLABLE_H
#define PYROOT_PYCALLABLE_H


namespace PyROOT {

/** Python callable object interface
      @author  WLAV
      @date    08/10/2004
      @version 2.0
 */

   class ObjectProxy;

   class PyCallable {
   public:
      virtual ~PyCallable() {}

   public:
      virtual PyObject* GetDocString() = 0;
      virtual Int_t GetPriority() { return 0; }

   public:
      virtual PyObject* operator()( ObjectProxy* self, PyObject* args, PyObject* kwds ) = 0;
   };

} // namespace PyROOT

#endif // !PYROOT_PYCALLABLE_H
