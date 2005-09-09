// @(#)root/pyroot:$Name:  $:$Id: MethodHolder.h,v 1.15 2005/06/06 15:08:40 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_TMETHODHOLDER_H
#define PYROOT_TMETHODHOLDER_H

// Bindings
#include "Utility.h"
#include "PyCallable.h"

// ROOT
#include "TClassRef.h"
class TMethod;

// CINT
class G__CallFunc;
class G__ClassInfo;

// Standard
#include <string>
#include <vector>


namespace PyROOT {

/** Python side ROOT method
      @author  WLAV
      @date    05/06/2004
      @version 2.0
 */

   class TExecutor;
   class TConverter;

   class TMethodHolder : public PyCallable {
   public:
      TMethodHolder( TClass* klass, TMethod* method );
      TMethodHolder( TFunction* function );
      TMethodHolder( const TMethodHolder& );
      TMethodHolder& operator=( const TMethodHolder& );
      virtual ~TMethodHolder();

   public:
      virtual PyObject* GetDocString();

   public:
      virtual PyObject* operator()( ObjectProxy* self, PyObject* args, PyObject* kwds );

      virtual Bool_t Initialize();
      virtual PyObject* FilterArgs( ObjectProxy*& self, PyObject* args, PyObject* kwds );
      virtual Bool_t SetMethodArgs( PyObject* args );
      virtual PyObject* Execute( void* self );

   protected:
      TClass* GetClass() { return fClass.GetClass(); }
      TFunction* GetMethod() { return fMethod; }

      virtual Bool_t InitExecutor_( TExecutor*& );

   private:
      void Copy_( const TMethodHolder& );
      void Destroy_() const;

      Bool_t InitCallFunc_( std::string& );
      void CalcOffset_( void* self, TClass* klass );

      void SetPyError_( PyObject* msg );

   private:
   // representation
      TClassRef    fClass;
      TFunction*   fMethod;
      G__CallFunc* fMethodCall;
      TExecutor*   fExecutor;

   // call dispatch buffers
      std::vector< TConverter* > fConverters;

   // cached values
      Int_t        fArgsRequired;
      Long_t       fOffset;
      Long_t       fTagnum;

   // admin
      Bool_t fIsInitialized;
   };

} // namespace PyROOT

#endif // !PYROOT_METHODHOLDER_H
