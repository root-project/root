#ifndef DICT2_CLASSH_H
#define DICT2_CLASSH_H

#include "ClassG.h"
#include <typeinfo>

class ClassH: public ClassG {
public:
   ClassH(): fH('h') {}

   virtual ~ClassH() {}

   int
   h() { return fH; }

   void
   setH(int v) { fH = v; }

   template <class T>
   static bool
   testLookup(const std::type_info& ti) {
      const std::type_info& ti2 = typeid(T);
      return ti == ti2;
   }


private:
   int fH;
};


#endif // DICT2_CLASSH_H
