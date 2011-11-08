// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_TMETHODHOLDER_H
#define PYROOT_TMETHODHOLDER_H

// Bindings
#include "Utility.h"
#include "PyCallable.h"

// ROOT
#include "TClassRef.h"
class TMethod;

// Reflex
#ifdef PYROOT_USE_REFLEX
#include "Reflex/Scope.h"
#include "Reflex/Member.h"
#endif

// CINT
namespace Cint {
   class G__CallFunc;
   class G__ClassInfo;
}
using namespace Cint;

// Standard
#include <string>
#include <vector>


namespace PyROOT {

/** Python side ROOT method
      @author  WLAV
      @date    05/06/2004
      @version 3.0
 */

   class TExecutor;
   class TConverter;

   template< class T, class M >
   class TMethodHolder : public PyCallable {
   public:
      TMethodHolder( const T& klass, const M& method );
      TMethodHolder( const TMethodHolder& );
      TMethodHolder& operator=( const TMethodHolder& );
      virtual ~TMethodHolder();

   public:
      virtual PyObject* GetSignature();
      virtual PyObject* GetPrototype();
      virtual Int_t GetPriority();

      virtual Int_t GetMaxArgs();
      virtual PyObject* GetArgSpec( Int_t iarg );
      virtual PyObject* GetArgDefault( Int_t iarg );
      virtual PyObject* GetScope();

      virtual PyCallable* Clone() { return new TMethodHolder( *this ); }

   public:
      virtual PyObject* operator()( ObjectProxy* self, PyObject* args, PyObject* kwds, Long_t = 0 );

      virtual Bool_t Initialize();
      virtual PyObject* FilterArgs( ObjectProxy*& self, PyObject* args, PyObject* kwds );
      virtual Bool_t SetMethodArgs( PyObject* args, Long_t user );
      virtual PyObject* Execute( void* self );

   protected:
      const M& GetMethod() { return fMethod; }
      const T& GetClass() { return fClass; }
      TExecutor* GetExecutor() { return fExecutor; }
      const std::string& GetSignatureString();

      virtual Bool_t InitExecutor_( TExecutor*& );

   private:
      void Copy_( const TMethodHolder& );
      void Destroy_() const;

      PyObject* CallFast( void* );
      PyObject* CallSafe( void* );

      Bool_t InitCallFunc_();

      void CreateSignature_();
      void SetPyError_( PyObject* msg );

   private:
   // representation
      M fMethod;
      T fClass;
      G__CallFunc* fMethodCall;
      TExecutor*   fExecutor;

      std::string fSignature;

   // call dispatch buffers
      std::vector< TConverter* > fConverters;

      std::vector< TParameter_t > fParameters;
      std::vector< void* >      fParamPtrs;

   // cached values
      Int_t        fArgsRequired;
      Long_t       fOffset;

   // admin
      Bool_t fIsInitialized;
   };

} // namespace PyROOT

#endif // !PYROOT_METHODHOLDER_H
