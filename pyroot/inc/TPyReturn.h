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
class TClass;

// Python
struct _object;
typedef _object PyObject;


class TPyReturn : public TObject {
public:
   TPyReturn();
   TPyReturn( PyObject* obj, TClass* cls );
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

   void autoDestruct() const;

private:
   PyObject* m_object;             // python side object
   TClass* m_class;                // TClass of held object if ROOT object
};

#endif
