// @(#)root/pyroot:$Name:  $:$Id: Converters.h,v 1.68 2005/01/28 05:45:41 brun Exp $
// Author: Wim Lavrijsen, Jan 2005
#ifndef PYROOT_CONVERTERS_H
#define PYROOT_CONVERTERS_H

// ROOT
#include "Rtypes.h"
#include "DllImport.h"
#include "TString.h"

// CINT
class G__CallFunc;

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
   };

#define PYROOT_BASIC_CONVERTER( name )                 \
   class name : public Converter {                     \
   public:                                             \
      virtual bool SetArg( PyObject*, G__CallFunc* );  \
   }

// converters for built-ins
   PYROOT_BASIC_CONVERTER( LongConverter );
   PYROOT_BASIC_CONVERTER( DoubleConverter );
   PYROOT_BASIC_CONVERTER( VoidConverter );

   class LongLongConverter : public Converter {
   public:
      virtual bool SetArg( PyObject*, G__CallFunc* );

   private:
      Long64_t fBuffer;
   };

   class CStringConverter : public Converter {
   public:
      virtual bool SetArg( PyObject*, G__CallFunc* );

   private:
      std::string fBuffer;
   };

// pointer/array conversions
   class VoidArrayConverter : public Converter {
   public:
      VoidArrayConverter( bool isConst = false ) { fIsConst = isConst; }
      virtual bool SetArg( PyObject*, G__CallFunc* );

   private:
      bool fIsConst;
   };

   PYROOT_BASIC_CONVERTER( IntArrayConverter );
   PYROOT_BASIC_CONVERTER( LongArrayConverter );
   PYROOT_BASIC_CONVERTER( FloatArrayConverter );
   PYROOT_BASIC_CONVERTER( DoubleArrayConverter );

// converters for special cases
   class TStringConverter : public Converter {
   public:
      virtual bool SetArg( PyObject*, G__CallFunc* );

   private:
      TString fBuffer;
   };

   PYROOT_BASIC_CONVERTER( PyObjectConverter );

// factories
   typedef Converter* (*ConverterFactory_t) ();
   typedef std::map< std::string, ConverterFactory_t > ConvFactories_t;
   R__EXTERN ConvFactories_t gConvFactories;

} // namespace PyROOT

#endif // !PYROOT_CONVERTERS_H
