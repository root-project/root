// @(#)root/pyroot:$Name:  $:$Id: Utility.h,v 1.3 2004/06/12 05:35:10 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_UTILITY_H
#define PYROOT_UTILITY_H

// Standard
#include <string>


namespace PyROOT {

// Bindings
   class ObjectHolder;

   namespace Utility {

   // convenience functions
      void addToClass( const char* label, PyCFunction cfunc, PyObject* cls,
                       int flags = METH_VARARGS );

      ObjectHolder* getObjectHolder( PyObject* self );
      void* getObjectFromHolderFromArgs( PyObject* argsTuple );

   // data/return types
      enum EDataType { kBool, kChar, kShort, kInt, kLong, kFloat, kDouble,
         kString, kVoid, kOther,
         kDoublePtr, kFloatPtr, kLongPtr, kIntPtr, kVoidPtr };

      EDataType effectiveType( const std::string& typeName );
      bool isPointer( const std::string& typeName );

   // data
      extern char* theObject_;
      extern PyObject* theObjectString_;

   } // namespace Utility

} // namespace PyROOT

#endif // !PYROOT_UTILITY_H
