// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Jan 2005
#ifndef PYROOT_CONVERTERS_H
#define PYROOT_CONVERTERS_H

// ROOT
#include "DllImport.h"
#include "TString.h"
#include "TClassRef.h"

// CINT
namespace Cint {
   class G__CallFunc;
   class G__TypeInfo;
}
using namespace Cint;

// Standard
#include <limits.h>
#include <string>
#include <map>


namespace PyROOT {

/** Python to CINT call converters
      @author  WLAV
      @date    01/26/2005
      @version 1.0
*/

   union TParameter_t;

   class TConverter {
   public:
      virtual ~TConverter() {}

   public:
      virtual Bool_t SetArg( PyObject*, TParameter_t&, G__CallFunc* = 0, Long_t = 0 ) = 0;
      virtual PyObject* FromMemory( void* address );
      virtual Bool_t ToMemory( PyObject* value, void* address );
   };

#define PYROOT_DECLARE_BASIC_CONVERTER( name )                                \
   class T##name##Converter : public TConverter {                             \
   public:                                                                    \
      virtual Bool_t SetArg( PyObject*, TParameter_t&, G__CallFunc* = 0, Long_t = 0 );\
      virtual PyObject* FromMemory( void* );                                  \
      virtual Bool_t ToMemory( PyObject*, void* );                            \
   }

#define PYROOT_DECLARE_BASIC_CONVERTER2( name, base )                         \
   class T##name##Converter : public T##base##Converter {                     \
   public:                                                                    \
      virtual PyObject* FromMemory( void* );                                  \
      virtual Bool_t ToMemory( PyObject*, void* );                            \
   }

#define PYROOT_DECLARE_ARRAY_CONVERTER( name )                                \
   class T##name##Converter : public TConverter {                             \
   public:                                                                    \
      T##name##Converter( Py_ssize_t size = -1 ) { fSize = size; }            \
      virtual Bool_t SetArg( PyObject*, TParameter_t&, G__CallFunc* = 0, Long_t = 0 );\
      virtual PyObject* FromMemory( void* );                                  \
      virtual Bool_t ToMemory( PyObject*, void* );                            \
   private:                                                                   \
      Py_ssize_t fSize;                                                       \
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
   PYROOT_DECLARE_BASIC_CONVERTER( IntRef );
   PYROOT_DECLARE_BASIC_CONVERTER( ULong );
   PYROOT_DECLARE_BASIC_CONVERTER2( UInt, ULong );
   PYROOT_DECLARE_BASIC_CONVERTER( LongLong );
   PYROOT_DECLARE_BASIC_CONVERTER( ULongLong );
   PYROOT_DECLARE_BASIC_CONVERTER( Double );
   PYROOT_DECLARE_BASIC_CONVERTER2( Float, Double );
   PYROOT_DECLARE_BASIC_CONVERTER( DoubleRef );

   class TConstLongRefConverter : public TLongConverter {
   public:
      virtual Bool_t SetArg( PyObject*, TParameter_t&, G__CallFunc* = 0, Long_t = 0 );

   private:
      Long_t fBuffer;
   };

   class TConstDoubleRefConverter : public TDoubleConverter { // required for Cintex only
   public:
      virtual Bool_t SetArg( PyObject*, TParameter_t&, G__CallFunc* = 0, Long_t = 0 );

   private:
      Double_t fBuffer;
   };

   class TVoidConverter : public TConverter {
   public:
      virtual Bool_t SetArg( PyObject*, TParameter_t&, G__CallFunc* = 0, Long_t = 0 );
   };

   class TMacroConverter : public TConverter {
   public:
      virtual Bool_t SetArg( PyObject*, TParameter_t&, G__CallFunc* = 0, Long_t = 0 );
      virtual PyObject* FromMemory( void* address );
   };

   class TCStringConverter : public TConverter {
   public:
      TCStringConverter( UInt_t maxSize = UINT_MAX ) : fMaxSize( maxSize ) {}

   public:
      virtual Bool_t SetArg( PyObject*, TParameter_t&, G__CallFunc* = 0, Long_t = 0 );
      virtual PyObject* FromMemory( void* address );
      virtual Bool_t ToMemory( PyObject* value, void* address );

   protected:
      std::string fBuffer;
      UInt_t fMaxSize;
   };

   class TNonConstCStringConverter : public TCStringConverter {
   public:
      TNonConstCStringConverter( UInt_t maxSize = UINT_MAX ) : TCStringConverter( maxSize ) {}

   public:
      virtual Bool_t SetArg( PyObject*, TParameter_t&, G__CallFunc* = 0, Long_t = 0 );
      virtual PyObject* FromMemory( void* address );
   };

   class TNonConstUCStringConverter : public TNonConstCStringConverter {
   public:
      TNonConstUCStringConverter( UInt_t maxSize = UINT_MAX ) : TNonConstCStringConverter( maxSize ) {}

   public:
      virtual Bool_t SetArg( PyObject*, TParameter_t&, G__CallFunc* = 0, Long_t = 0 );
   };

// pointer/array conversions
   class TVoidArrayConverter : public TConverter {
   public:
      TVoidArrayConverter( Bool_t keepControl = kTRUE ) { fKeepControl = keepControl; }
      virtual Bool_t SetArg( PyObject*, TParameter_t&, G__CallFunc* = 0, Long_t = 0 );
      virtual PyObject* FromMemory( void* address );
      virtual Bool_t ToMemory( PyObject* value, void* address );

   protected:
      virtual Bool_t GetAddressSpecialCase( PyObject* pyobject, void*& address );

   protected:
      Bool_t KeepControl() { return fKeepControl; }

   private:
      Bool_t fKeepControl;
   };

   PYROOT_DECLARE_ARRAY_CONVERTER( BoolArray );
   PYROOT_DECLARE_ARRAY_CONVERTER( ShortArray );
   PYROOT_DECLARE_ARRAY_CONVERTER( UShortArray );
   PYROOT_DECLARE_ARRAY_CONVERTER( IntArray );
   PYROOT_DECLARE_ARRAY_CONVERTER( UIntArray );
   PYROOT_DECLARE_ARRAY_CONVERTER( LongArray );
   PYROOT_DECLARE_ARRAY_CONVERTER( ULongArray );
   PYROOT_DECLARE_ARRAY_CONVERTER( FloatArray );
   PYROOT_DECLARE_ARRAY_CONVERTER( DoubleArray );

   class TLongLongArrayConverter : public TVoidArrayConverter {
   public:
      virtual Bool_t SetArg( PyObject*, TParameter_t&, G__CallFunc* = 0, Long_t = 0 );
   };

// converters for special cases
   class TRootObjectConverter : public TVoidArrayConverter {
   public:
      TRootObjectConverter( const TClassRef& klass, Bool_t keepControl = kFALSE ) :
         TVoidArrayConverter( keepControl ), fClass( klass ) {}

   public:
      virtual Bool_t SetArg( PyObject*, TParameter_t&, G__CallFunc* = 0, Long_t = 0 );
      virtual PyObject* FromMemory( void* address );
      virtual Bool_t ToMemory( PyObject* value, void* address );

   protected:
      TClassRef fClass;
   };

   class TStrictRootObjectConverter : public TRootObjectConverter {
   public:
      TStrictRootObjectConverter( const TClassRef& klass, Bool_t keepControl = kFALSE ) :
         TRootObjectConverter( klass, keepControl ) {}

   protected:
      virtual Bool_t GetAddressSpecialCase( PyObject*, void*& ) { return kFALSE; }
   };

   class TRootObjectPtrConverter : public TRootObjectConverter {
   public:
      TRootObjectPtrConverter( const TClassRef& klass, Bool_t keepControl = kFALSE ) :
         TRootObjectConverter( klass, keepControl ) {}

   public:
      virtual Bool_t SetArg( PyObject*, TParameter_t&, G__CallFunc* = 0, Long_t = 0 );
      virtual PyObject* FromMemory( void* address );
      virtual Bool_t ToMemory( PyObject* value, void* address );
   };

   class TVoidPtrRefConverter : public TConverter {
   public:
      virtual Bool_t SetArg( PyObject*, TParameter_t&, G__CallFunc* = 0, Long_t = 0 );
   };

   class TVoidPtrPtrConverter : public TConverter {
   public:
      virtual Bool_t SetArg( PyObject*, TParameter_t&, G__CallFunc* = 0, Long_t = 0 );
      virtual PyObject* FromMemory( void* address );
   };

   PYROOT_DECLARE_BASIC_CONVERTER( PyObject );

#define PYROOT_DECLARE_STRING_CONVERTER( name, strtype )                      \
   class T##name##Converter : public TRootObjectConverter {                   \
   public:                                                                    \
      T##name##Converter( Bool_t keepControl = kTRUE );                       \
   public:                                                                    \
   virtual Bool_t SetArg( PyObject*, TParameter_t&, G__CallFunc* = 0, Long_t = 0 );\
      virtual PyObject* FromMemory( void* address );                          \
      virtual Bool_t ToMemory( PyObject* value, void* address );              \
   private:                                                                   \
      strtype fBuffer;                                                        \
   }

   PYROOT_DECLARE_STRING_CONVERTER( TString,   TString );
   PYROOT_DECLARE_STRING_CONVERTER( STLString, std::string );

// factories
   typedef TConverter* (*ConverterFactory_t) ( Long_t user );
   typedef std::map< std::string, ConverterFactory_t > ConvFactories_t;
   R__EXTERN ConvFactories_t gConvFactories;

// create converter from fully qualified type
   TConverter* CreateConverter( const std::string& fullType, Long_t user = -1 );

} // namespace PyROOT

#endif // !PYROOT_CONVERTERS_H
