#include <TObject.h>

class t02 {
public:
   TObject * const * obj; //!
   t02() : obj(new TObject*(new TObject)) {
      // obj = new TObject*;
      //*obj = new TObject;
   }
   TObject * const * getVal() { return obj; }

};

