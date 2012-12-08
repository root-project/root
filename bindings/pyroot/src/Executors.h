// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Jan 2005
#ifndef PYROOT_EXECUTORS_H
#define PYROOT_EXECUTORS_H

// ROOT
#include "DllImport.h"
#include "TClassRef.h"

// CINT
namespace Cint {
class G__CallFunc;
}
using namespace Cint;

// Standard
#include <string>
#include <map>


namespace PyROOT {

/** Executors of CINT calls and conversions back to python
      @author  WLAV
      @date    01/27/2005
      @version 1.0
*/

   class TExecutor {
   public:
      virtual ~TExecutor() {}
      virtual PyObject* Execute( G__CallFunc*, void*, Bool_t release_gil ) = 0;
   };

#define PYROOT_DECLARE_BASIC_EXECUTOR( name )                                 \
   class T##name##Executor : public TExecutor {                               \
   public:                                                                    \
      virtual PyObject* Execute( G__CallFunc*, void*, Bool_t release_gil );   \
   }

// executors for built-ins
   PYROOT_DECLARE_BASIC_EXECUTOR( Bool );
   PYROOT_DECLARE_BASIC_EXECUTOR( Long );
   PYROOT_DECLARE_BASIC_EXECUTOR( Char );
   PYROOT_DECLARE_BASIC_EXECUTOR( Int );
   PYROOT_DECLARE_BASIC_EXECUTOR( ULong );
   PYROOT_DECLARE_BASIC_EXECUTOR( LongLong );
   PYROOT_DECLARE_BASIC_EXECUTOR( ULongLong );
   PYROOT_DECLARE_BASIC_EXECUTOR( Double );
   PYROOT_DECLARE_BASIC_EXECUTOR( Void );
   PYROOT_DECLARE_BASIC_EXECUTOR( CString );

// pointer/array executors
   PYROOT_DECLARE_BASIC_EXECUTOR( VoidArray );
   PYROOT_DECLARE_BASIC_EXECUTOR( BoolArray );
   PYROOT_DECLARE_BASIC_EXECUTOR( ShortArray );
   PYROOT_DECLARE_BASIC_EXECUTOR( UShortArray );
   PYROOT_DECLARE_BASIC_EXECUTOR( IntArray );
   PYROOT_DECLARE_BASIC_EXECUTOR( UIntArray );
   PYROOT_DECLARE_BASIC_EXECUTOR( LongArray );
   PYROOT_DECLARE_BASIC_EXECUTOR( ULongArray );
   PYROOT_DECLARE_BASIC_EXECUTOR( FloatArray );
   PYROOT_DECLARE_BASIC_EXECUTOR( DoubleArray );

// special cases
   PYROOT_DECLARE_BASIC_EXECUTOR( STLString );
   PYROOT_DECLARE_BASIC_EXECUTOR( TGlobal );

   class TRootObjectExecutor : public TExecutor {
   public:
      TRootObjectExecutor( const TClassRef& klass ) : fClass( klass ) {}
      virtual PyObject* Execute( G__CallFunc*, void*, Bool_t release_gil );

   protected:
      TClassRef fClass;
   };

   class TRootObjectByValueExecutor : public TRootObjectExecutor {
   public:
      TRootObjectByValueExecutor( const TClassRef& klass ) : TRootObjectExecutor ( klass ) {}
      virtual PyObject* Execute( G__CallFunc*, void*, Bool_t release_gil );
   };

   PYROOT_DECLARE_BASIC_EXECUTOR( Constructor );
   PYROOT_DECLARE_BASIC_EXECUTOR( PyObject );

   class TRefExecutor : public TExecutor {
   public:
      TRefExecutor() : fAssignable( 0 ) {}

   public:
      virtual Bool_t SetAssignable( PyObject* );

   protected:
      PyObject* fAssignable;
   };

#define PYROOT_DECLARE_BASIC_REFEXECUTOR( name )                              \
   class T##name##RefExecutor : public TRefExecutor {                         \
   public:                                                                    \
   virtual PyObject* Execute( G__CallFunc*, void*, Bool_t release_gil );      \
   }

   PYROOT_DECLARE_BASIC_REFEXECUTOR( Short );
   PYROOT_DECLARE_BASIC_REFEXECUTOR( UShort );
   PYROOT_DECLARE_BASIC_REFEXECUTOR( Int );
   PYROOT_DECLARE_BASIC_REFEXECUTOR( UInt );
   PYROOT_DECLARE_BASIC_REFEXECUTOR( Long );
   PYROOT_DECLARE_BASIC_REFEXECUTOR( ULong );
   PYROOT_DECLARE_BASIC_REFEXECUTOR( Float );
   PYROOT_DECLARE_BASIC_REFEXECUTOR( Double );
   PYROOT_DECLARE_BASIC_REFEXECUTOR( STLString );

// special cases
   class TRootObjectRefExecutor : public TRefExecutor {
   public:
      TRootObjectRefExecutor( const TClassRef& klass ) : fClass( klass ) {}
      virtual PyObject* Execute( G__CallFunc*, void*, Bool_t release_gil );

   protected:
      TClassRef fClass;
   };

// factories
   typedef TExecutor* (*ExecutorFactory_t) ();
   typedef std::map< std::string, ExecutorFactory_t > ExecFactories_t;
   R__EXTERN ExecFactories_t gExecFactories;

// create executor from fully qualified type
   TExecutor* CreateExecutor( const std::string& fullType );

} // namespace PyROOT

#endif // !PYROOT_EXECUTORS_H
