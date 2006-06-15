// @(#)root/pyroot:$Name:  $:$Id: PyCallable.h,v 1.4 2006/03/24 06:04:09 brun Exp $
// Author: Wim Lavrijsen, Aug 2004

#ifndef PYROOT_PYCALLABLE_H
#define PYROOT_PYCALLABLE_H


namespace PyROOT {

/** Python callable object interface
      @author  WLAV
      @date    08/10/2004
      @version 3.0
 */

   class ObjectProxy;

   class PyCallable {
   public:
      virtual ~PyCallable() {}

   public:
      virtual PyObject* GetSignature() = 0;
      virtual PyObject* GetPrototype() = 0;
      virtual PyObject* GetDocString() { return GetPrototype(); }
      virtual Int_t GetPriority() { return 0; }

   public:
      virtual PyObject* operator()( ObjectProxy* self, PyObject* args, PyObject* kwds ) = 0;
   };

} // namespace PyROOT

#endif // !PYROOT_PYCALLABLE_H
