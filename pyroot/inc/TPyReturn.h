// @(#)root/pyroot:$Name:  $:$Id: TPyReturn.h,v 1.3 2005/03/04 07:44:11 brun Exp $
// Author: Wim Lavrijsen   May 2004

#ifndef ROOT_TPyReturn
#define ROOT_TPyReturn

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// TPyReturn                                                                //
//                                                                          //
// Morphing return type from evaluating python expressions.                 //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////


// ROOT
#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TClassRef
#include "TClassRef.h"
#endif
class TClass;


// Python
struct _object;
typedef _object PyObject;


class TPyReturn : public TObject {
public:
   TPyReturn();
   TPyReturn( PyObject* pyobject, TClass* klass );
   virtual ~TPyReturn();

   virtual TClass* IsA() const;

// conversions to standard types, may fail if unconvertible
   operator const char*() const;
   operator long() const;
   operator int() const;
   operator double() const;
   operator float() const;

   operator TObject*() const;

private:
   TPyReturn( const TPyReturn& );
   TPyReturn& operator=( const TPyReturn& );

   void AutoDestruct_() const;

private:
   PyObject* fPyObject;            // python side object
   TClassRef fClass;               // TClass of held object if ROOT object

};
#endif
