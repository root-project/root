// @(#)root/pyroot:$Name:  $:$Id: Utility.h,v 1.2 2004/05/07 20:47:20 brun Exp $
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
