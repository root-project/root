// Author: Wim Lavrijsen, Aug 2004

#ifndef PYROOT_PYCALLABLE_H
#define PYROOT_PYCALLABLE_H


namespace PyROOT {

/** Python callable object interface
      @author  WLAV
      @date    08/10/2004
      @version 1.0
 */

   class PyCallable {
   public:
      virtual ~PyCallable() {}

   public:
      virtual PyObject* operator()( PyObject* aTuple, PyObject* aDict ) = 0;
   };

} // namespace PyROOT

#endif // !PYROOT_PYCALLABLE_H
