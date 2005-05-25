// @(#)root/pyroot:$Name:  $:$Id: Converters.h,v 1.6 2005/05/25 06:23:36 brun Exp $
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

#define PYROOT_BASIC_CONVERTER( name )                                        \
   class name##Converter : public Converter {                                 \
   public:                                                                    \
      virtual bool SetArg( PyObject*, G__CallFunc* );                         \
      virtual PyObject* FromMemory( void* );                                  \
      virtual bool ToMemory( PyObject*, void* );                              \
   }

#define PYROOT_ARRAY_CONVERTER( name )                                        \
   class name##Converter : public Converter {                                 \
   public:                                                                    \
      name##Converter( int size = -1 ) { fSize = size; }                      \
      virtual bool SetArg( PyObject*, G__CallFunc* );                         \
      virtual PyObject* FromMemory( void* );                                  \
      virtual bool ToMemory( PyObject*, void* );                              \
   private:                                                                   \
     int fSize;                                                               \
   }

#define PYROOT_BASIC_CONVERTER2( name, base )                                 \
   class name##Converter : public base##Converter {                           \
   public:                                                                    \
      virtual PyObject* FromMemory( void* );                                  \
      virtual bool ToMemory( PyObject*, void* );                              \
   }

// converters for built-ins
   PYROOT_BASIC_CONVERTER( Long );
   PYROOT_BASIC_CONVERTER2( Bool, Long );
   PYROOT_BASIC_CONVERTER( Char );
   PYROOT_BASIC_CONVERTER( UChar );
   PYROOT_BASIC_CONVERTER2( Short, Long );
   PYROOT_BASIC_CONVERTER2( UShort, Long );
   PYROOT_BASIC_CONVERTER2( Int, Long );
   PYROOT_BASIC_CONVERTER2( UInt, Long );
   PYROOT_BASIC_CONVERTER2( ULong, Long );
   PYROOT_BASIC_CONVERTER( Double );
   PYROOT_BASIC_CONVERTER2( Float, Double );
   PYROOT_BASIC_CONVERTER( Void );
   PYROOT_BASIC_CONVERTER( LongLong );

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
      VoidArrayConverter( bool isConst = false ) { fIsConst = isConst; }
      virtual bool SetArg( PyObject*, G__CallFunc* );

   protected:
      bool IsConst() { return fIsConst; }

   private:
      bool fIsConst;
   };

   PYROOT_ARRAY_CONVERTER( ShortArray );
   PYROOT_ARRAY_CONVERTER( UShortArray );
   PYROOT_ARRAY_CONVERTER( IntArray );
   PYROOT_ARRAY_CONVERTER( UIntArray );
   PYROOT_ARRAY_CONVERTER( LongArray );
   PYROOT_ARRAY_CONVERTER( ULongArray );
   PYROOT_ARRAY_CONVERTER( FloatArray );
   PYROOT_ARRAY_CONVERTER( DoubleArray );

// converters for special cases
   class TStringConverter : public Converter {
   public:
      virtual bool SetArg( PyObject*, G__CallFunc* );

   private:
      TString fBuffer;
   };

   class KnownClassConverter: public VoidArrayConverter {
   public:
      KnownClassConverter( const TClassRef& klass, bool isConst = false ) :
         VoidArrayConverter( isConst ), fClass( klass ) {}
      KnownClassConverter( TClass* klass, bool isConst = false ) :
         VoidArrayConverter( isConst ), fClass( klass ) {}
      virtual bool SetArg( PyObject*, G__CallFunc* );

   private:
      TClassRef fClass;
   };

   class LongLongArrayConverter : public VoidArrayConverter {
   public:
      virtual bool SetArg( PyObject*, G__CallFunc* );
   };

   PYROOT_BASIC_CONVERTER( PyObject );

// factories
   typedef Converter* (*ConverterFactory_t) ( long user );
   typedef std::map< std::string, ConverterFactory_t > ConvFactories_t;
   R__EXTERN ConvFactories_t gConvFactories;

// create converter from fully qualified type
   Converter* CreateConverter( const std::string& fullType, long user = -1 );

} // namespace PyROOT

#endif // !PYROOT_CONVERTERS_H
