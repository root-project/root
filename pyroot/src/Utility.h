// @(#)root/pyroot:$Name:  $:$Id: Utility.h,v 1.5 2004/07/29 04:41:38 brun Exp $
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
      enum EDataType {
         kBool        = 0x0001,
         kChar        = 0x0002,
         kShort       = 0x0004,
         kInt         = 0x0008,
         kLong        = 0x0010,
         kFloat       = 0x0020,
         kDouble      = 0x0040,
         kVoid        = 0x0080,
         kOther       = 0x0100,
         kLongLong    = 0x0200,
         kString      = 0x1000 | kChar,
         kDoublePtr   = 0x1000 | kDouble,
         kFloatPtr    = 0x1000 | kFloat,
         kLongPtr     = 0x1000 | kLong,
         kIntPtr      = 0x1000 | kInt,
         kVoidPtr     = 0x1000 | kVoid
      };

      EDataType effectiveType( const std::string& typeName );
      int isPointer( const std::string& typeName ); // 0 = no, 1 = yes, 2 = ref

   // data
      extern char* theObject_;
      extern PyObject* theObjectString_;

   } // namespace Utility

} // namespace PyROOT

#endif // !PYROOT_UTILITY_H
