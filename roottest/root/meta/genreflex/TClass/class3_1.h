#ifndef CLASS3_1_H
#define CLASS3_1_H

#include "class3_2.h"

class class3_1 {

public:
   class3_1(int i);
   class3_1();
   ~class3_1();
   int getI();
   void setI(int i);
private:
   int m_i;
   class3_2 m_class3_2_obj;
};
#endif

class3_1::class3_1(int i):
  m_i(i){
   };

class3_1::class3_1():
   m_i(0){
   };

class3_1::~class3_1(){};

int class3_1::getI(){return m_i;};

void class3_1::setI(int i){m_i=i;};

