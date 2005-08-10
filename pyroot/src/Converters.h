// @(#)root/pyroot:$Name:  $:$Id: Converters.h,v 1.9 2005/06/24 07:19:03 brun Exp $
// Author: Wim Lavrijsen, Jan 2005
#ifndef PYROOT_CONVERTERS_H
#define PYROOT_CONVERTERS_H

// ROOT
#include "Rtypes.h"
#include "DllImport.h"
#include "TString.h"
#include "TClassRef.h"

// CINT
class G__CallFunc;
class G__TypeInfo;

// Standard
#include <string>
#include <map>


namespace PyROOT {

/** Python to CINT call converters
      @author  WLAV
      @date    01/26/2005
      @version 1.0
*/

   class Converter {
   public:
      virtual ~Converter() {}

   public:
      virtual bool SetArg( PyObject*, G__CallFunc* ) = 0;
      virtual PyObject* FromMemory( void* address );
      virtual bool ToMemory( PyObject* value, void* address );
   };

#define PYROOT_DECLARE_BASIC_CONVERTER( name )                                \
   class name##Converter : public Converter {                                 \
   public:                                                                    \
      virtual bool SetArg( PyObject*, G__CallFunc* );                         \
      virtual PyObject* FromMemory( void* );                                  \
      virtual bool ToMemory( PyObject*, void* );                              \
   }

#define PYROOT_DECLARE_BASIC_CONVERTER2( name, base )                         \
   class name##Converter : public base##Converter {                           \
   public:                                                                    \
      virtual PyObject* FromMemory( void* );                                  \
      virtual bool ToMemory( PyObject*, void* );                              \
   }

#define PYROOT_DECLARE_ARRAY_CONVERTER( name )                                \
   class name##Converter : public Converter {                                 \
   public:                                                                    \
      name##Converter( int size = -1 ) { fSize = size; }                      \
      virtual bool SetArg( PyObject*, G__CallFunc* );                         \
      virtual PyObject* FromMemory( void* );                                  \
      virtual bool ToMemory( PyObject*, void* );                              \
   private:                                                                   \
     int fSize;                                                               \
   }

// converters for built-ins
   PYROOT_DECLARE_BASIC_CONVERTER( Long );
   PYROOT_DECLARE_BASIC_CONVERTER( LongRef );
   PYROOT_DECLARE_BASIC_CONVERTER( Bool );
   PYROOT_DECLARE_BASIC_CONVERTER( Char );
   PYROOT_DECLARE_BASIC_CONVERTER( UChar );
   PYROOT_DECLARE_BASIC_CONVERTER2( Short, Long );
   PYROOT_DECLARE_BASIC_CONVERTER2( UShort, Long );
   PYROOT_DECLARE_BASIC_CONVERTER2( Int, Long );
   PYROOT_DECLARE_BASIC_CONVERTER2( UInt, Long );
   PYROOT_DECLARE_BASIC_CONVERTER2( ULong, Long );
   PYROOT_DECLARE_BASIC_CONVERTER( LongLong );
   PYROOT_DECLARE_BASIC_CONVERTER( Double );
   PYROOT_DECLARE_BASIC_CONVERTER2( Float, Double );
   PYROOT_DECLARE_BASIC_CONVERTER( DoubleRef );

   class VoidConverter : public Converter {
   public:
      virtual bool SetArg( PyObject*, G__CallFunc* );
   };

   class CStringConverter : public Converter {
   public:
      virtual bool SetArg( PyObject*, G__CallFunc* );
      virtual PyObject* FromMemory( void* address );
      virtual bool ToMemory( PyObject* value, void* address );

   private:
      std::string fBuffer;
   };

// pointer/array conversions
   class VoidArrayConverter : public Converter {
   public:
      VoidArrayConverter( bool keepControl = true ) { fKeepControl = keepControl; }
      virtual bool SetArg( PyObject*, G__CallFunc* );
      virtual PyObject* FromMemory( void* address );
      virtual bool ToMemory( PyObject* value, void* address );

   protected:
      bool KeepControl() { return fKeepControl; }

   private:
      bool fKeepControl;
   };

   PYROOT_DECLARE_ARRAY_CONVERTER( ShortArray );
   PYROOT_DECLARE_ARRAY_CONVERTER( UShortArray );
   PYROOT_DECLARE_ARRAY_CONVERTER( IntArray );
   PYROOT_DECLARE_ARRAY_CONVERTER( UIntArray );
   PYROOT_DECLARE_ARRAY_CONVERTER( LongArray );
   PYROOT_DECLARE_ARRAY_CONVERTER( ULongArray );
   PYROOT_DECLARE_ARRAY_CONVERTER( FloatArray );
   PYROOT_DECLARE_ARRAY_CONVERTER( DoubleArray );

   class LongLongArrayConverter : public VoidArrayConverter {
   public:
      virtual bool SetArg( PyObject*, G__CallFunc* );
   };

// converters for special cases
#define PYROOT_DECLARE_STRING_CONVERTER( name, strtype )                      \
   class name##Converter : public Converter {                                 \
   public:                                                                    \
      virtual bool SetArg( PyObject*, G__CallFunc* );                         \
      virtual PyObject* FromMemory( void* address );                          \
      virtual bool ToMemory( PyObject* value, void* address );                \
   private:                                                                   \
      strtype fBuffer;                                                        \
   }

   PYROOT_DECLARE_STRING_CONVERTER( TString,   TString );
   PYROOT_DECLARE_STRING_CONVERTER( STLString, std::string );

   class RootObjectConverter: public VoidArrayConverter {
   public:
      RootObjectConverter( const TClassRef& klass, bool keepControl = false ) :
         VoidArrayConverter( keepControl ), fClass( klass ) {}
      RootObjectConverter( TClass* klass, bool keepControl = false ) :
         VoidArrayConverter( keepControl ), fClass( klass ) {}
      virtual bool SetArg( PyObject*, G__CallFunc* );
      virtual PyObject* FromMemory( void* address );
      virtual bool ToMemory( PyObject* value, void* address );

   protected:
      TClassRef fClass;
   };

   class RootObjectPtrConverter : public RootObjectConverter {
   public:
      RootObjectPtrConverter( const TClassRef& klass, bool keepControl = false ) :
         RootObjectConverter( klass, keepControl ) {}
      RootObjectPtrConverter( TClass* klass, bool keepControl = false ) :
         RootObjectConverter( klass, keepControl ) {}
      virtual bool SetArg( PyObject*, G__CallFunc* );
      virtual PyObject* FromMemory( void* address );
      virtual bool ToMemory( PyObject* value, void* address );
   };

   class VoidPtrRefConverter : public Converter {
   public:
      virtual bool SetArg( PyObject*, G__CallFunc* );
   };

   PYROOT_DECLARE_BASIC_CONVERTER( PyObject );

// factories
   typedef Converter* (*ConverterFactory_t) ( long user );
   typedef std::map< std::string, ConverterFactory_t > ConvFactories_t;
   R__EXTERN ConvFactories_t gConvFactories;

// create converter from fully qualified type
   Converter* CreateConverter( const std::string& fullType, long user = -1 );

} // namespace PyROOT

#endif // !PYROOT_CONVERTERS_H
