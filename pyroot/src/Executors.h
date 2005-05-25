// @(#)root/pyroot:$Name:  $:$Id: Executors.h,v 1.2 2005/03/30 05:16:19 brun Exp $
// Author: Wim Lavrijsen, Jan 2005
#ifndef PYROOT_EXECUTORS_H
#define PYROOT_EXECUTORS_H

// ROOT
#include "DllImport.h"
#include "TClassRef.h"

// CINT
class G__CallFunc;

// Standard
#include <string>
#include <map>


namespace PyROOT {

/** Executors of CINT calls and conversions back to python
      @author  WLAV
      @date    01/27/2005
      @version 1.0
*/

   class Executor {
   public:
      virtual ~Executor() {}

   public:
      virtual PyObject* Execute( G__CallFunc*, void* ) = 0;
   };

#define PYROOT_BASIC_EXECUTOR( name )                  \
   class name : public Executor {                      \
   public:                                             \
      virtual PyObject* Execute( G__CallFunc*, void* );\
   }

// executors for built-ins
   PYROOT_BASIC_EXECUTOR( CharExecutor );
   PYROOT_BASIC_EXECUTOR( IntExecutor );
   PYROOT_BASIC_EXECUTOR( LongExecutor );
   PYROOT_BASIC_EXECUTOR( ULongExecutor );
   PYROOT_BASIC_EXECUTOR( DoubleExecutor );
   PYROOT_BASIC_EXECUTOR( VoidExecutor );
   PYROOT_BASIC_EXECUTOR( LongLongExecutor );
   PYROOT_BASIC_EXECUTOR( CStringExecutor );

// pointer/array executors
   PYROOT_BASIC_EXECUTOR( VoidArrayExecutor );
   PYROOT_BASIC_EXECUTOR( ShortArrayExecutor );
   PYROOT_BASIC_EXECUTOR( UShortArrayExecutor );
   PYROOT_BASIC_EXECUTOR( IntArrayExecutor );
   PYROOT_BASIC_EXECUTOR( UIntArrayExecutor );
   PYROOT_BASIC_EXECUTOR( LongArrayExecutor );
   PYROOT_BASIC_EXECUTOR( ULongArrayExecutor );
   PYROOT_BASIC_EXECUTOR( FloatArrayExecutor );
   PYROOT_BASIC_EXECUTOR( DoubleArrayExecutor );

// special cases
   PYROOT_BASIC_EXECUTOR( STLStringExecutor );
   PYROOT_BASIC_EXECUTOR( TGlobalExecutor );

   class RootObjectExecutor : public Executor {
   public:
      RootObjectExecutor( const TClassRef& klass ) : fClass( klass ) {}

   public:
      virtual PyObject* Execute( G__CallFunc*, void* );

   private:
      TClassRef fClass;
   };

   PYROOT_BASIC_EXECUTOR( ConstructorExecutor );

// factories
   typedef Executor* (*ExecutorFactory_t) ();
   typedef std::map< std::string, ExecutorFactory_t > ExecFactories_t;
   R__EXTERN ExecFactories_t gExecFactories;

} // namespace PyROOT

#endif // !PYROOT_EXECUTORS_H
