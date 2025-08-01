#ifndef myclass_h
#define myclass_h

#include "TObject.h"

class wrapper;

class myclass {
public:
   myclass *myself;
   wrapper *indirect;
   int fValue;

   myclass() : myself(this),indirect(0),fValue(0) {}
   virtual ~myclass() {}

   void set();

   void verify();

   ClassDef(myclass,1);
};

class wrapper {
public:
   myclass *fParent;
   int fIndex;

   wrapper(myclass *p = 0) : fParent(p),fIndex(0) {}
   virtual ~wrapper() {}

   ClassDef(wrapper,1);
};

inline void myclass::set() {
   indirect = new wrapper(this);
}

inline void myclass::verify() {
   if (myself != this) {
      fprintf(stdout,"The myself data member is incorrect\n");
   }
   if (indirect == 0) {
      fprintf(stdout,"The indirect pointer is still null\n");
   } else {
      if (this != indirect->fParent) {
         if (indirect->fParent==0) {
            fprintf(stdout,"The indirect fParent member is still null\n");
         } else {
            fprintf(stdout,"The indirect fParent member is incorrect\n");
         }
      }
   }
}

#endif
