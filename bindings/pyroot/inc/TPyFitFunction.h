// Author: Wim Lavrijsen   November 2010

#ifndef ROOT_TPyFitFunction
#define ROOT_TPyFitFunction

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// TPyFitFunction                                                           //
//                                                                          //
// Python base class to work with Math::IMultiGenFunction                   //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////


//- ROOT
#ifndef ROOT_Math_IFunction
#include "Math/IFunction.h"
#endif

// Python
struct _object;
typedef _object PyObject;


class TPyMultiGenFunction : public ROOT::Math::IMultiGenFunction {
public:
// ctor/dtor, and assignment
   TPyMultiGenFunction( PyObject* self = 0 );
   virtual ~TPyMultiGenFunction();

// Math::IMultiGenFunction implementation
   virtual TPyMultiGenFunction* Clone() const;
   virtual unsigned int NDim() const;
   virtual double DoEval( const double* ) const;

   ClassDef( TPyMultiGenFunction, 1 );   //Python base class for Math::IMultiGenFunction

private:
// private helpers for forwarding to python
   PyObject* CallSelf( const char* method, PyObject* pyobject = 0 ) const;

private:
// to prevent confusion when handing 'self' from python
   TPyMultiGenFunction( const TPyMultiGenFunction& ) {}
   TPyMultiGenFunction& operator=( const TPyMultiGenFunction& ) { return *this; }

private:
   PyObject* fPySelf;              //! actual python object
};

#endif
