// @(#)root/pyroot:$Name:  $:$Id: MethodHolder.h,v 1.11 2005/03/04 07:44:11 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_METHODHOLDER_H
#define PYROOT_METHODHOLDER_H

// Bindings
#include "Utility.h"
#include "PyCallable.h"

// ROOT
#include "TClassRef.h"
class TMethod;

// CINT
class G__CallFunc;

// Standard
#include <string>
#include <vector>


namespace PyROOT {

/** Python side ROOT method
      @author  WLAV
      @date    05/06/2004
      @version 2.0
 */

   class Executor;
   class Converter;

   class MethodHolder : public PyCallable {
   public:
      MethodHolder( TClass* klass, TMethod* method );
      MethodHolder( const MethodHolder& );
      MethodHolder& operator=( const MethodHolder& );
      virtual ~MethodHolder();

   public:
      virtual PyObject* GetDocString();

   public:
      virtual PyObject* operator()( ObjectProxy* self, PyObject* args, PyObject* kwds );

      virtual bool Initialize();
      virtual bool FilterArgs( ObjectProxy*& self, PyObject*& args, PyObject*& kwds );
      virtual bool SetMethodArgs( PyObject* args );
      virtual PyObject* Execute( void* self );

   protected:
      TClass* GetClass() { return fClass.GetClass(); }
      TMethod* GetMethod() { return fMethod; }

      virtual bool InitExecutor_( Executor*& );

   private:
      void Copy_( const MethodHolder& );
      void Destroy_() const;

      bool InitCallFunc_( std::string& );
      void CalcOffset_( void* self, TClass* klass );

   private:
   // representation
      TClassRef    fClass;
      TMethod*     fMethod;
      G__CallFunc* fMethodCall;
      Executor*    fExecutor;

   // call dispatch buffers
      std::vector< Converter* > fConverters;

   // cached values
      int          fArgsRequired;
      long         fOffset;
      long         fTagnum;

   // admin
      bool fIsInitialized;
   };

} // namespace PyROOT

#endif // !PYROOT_METHODHOLDER_H
