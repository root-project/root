// @(#)root/pyroot:$Name:  $:$Id:  $
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_UTILITY_H
#define PYROOT_UTILITY_H

namespace PyROOT {

// Bindings
   class ObjectHolder;

   namespace Utility {

   // convenience functions
      ObjectHolder* getObjectHolder( PyObject* self );

   // data
      extern char* theObject_;
      extern PyObject* theObjectString_;

   } // namespace Utility

} // namespace PyROOT

#endif // !PYROOT_UTILITY_H
