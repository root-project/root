// @(#)root/pyroot:$Name:  $:$Id: Utility.h,v 1.9 2004/11/02 10:13:06 rdm Exp $
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
      const int kPtrMask = 0x10000000;

      enum EDataType {
         kBool        = 0x00000001,
         kChar        = 0x00000002,
         kShort       = 0x00000004,
         kInt         = 0x00000008,
         kUInt        = 0x00000010,
         kLong        = 0x00000020,
         kULong       = 0x00000040,
         kFloat       = 0x00000080,
         kDouble      = 0x00000100,
         kVoid        = 0x00000200,
         kOther       = 0x00000400,
         kLongLong    = 0x00000800,
         kEnum        = 0x00001000,
         kSTLString   = 0x00002000,
         kString      = kPtrMask | kChar,
         kDoublePtr   = kPtrMask | kDouble,
         kFloatPtr    = kPtrMask | kFloat,
         kLongPtr     = kPtrMask | kLong,
         kIntPtr      = kPtrMask | kInt,
         kVoidPtr     = kPtrMask | kVoid
      };

      EDataType effectiveType( const std::string& typeName );
      int isPointer( const std::string& typeName ); // 0 = no, 1 = yes, 2 = ref

   // data
      extern char* theObject_;
      extern PyObject* theObjectString_;

   } // namespace Utility

} // namespace PyROOT

#endif // !PYROOT_UTILITY_H
