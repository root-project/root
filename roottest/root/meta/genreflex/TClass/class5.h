#ifndef CLASS5_H
#define CLASS5_H

#include "class3.h"

class class5 {

public:
   class5(int i, int j);
   class5();
   ~class5();
   int getI();
   void setI(int i);
   int getJ();
   void setJ(int i);
private:
   int m_i;
   int m_j;
   class3* m_class3_ptr;
};
#endif

class5::class5(int i, int j):
  m_i(i),
  m_j(j),
  m_class3_ptr(new class3()){};

class5::class5():
  m_i(0),
  m_j(0),
  m_class3_ptr(new class3()){};

class5::~class5(){};

int class5::getI(){return m_i;};

void class5::setI(int i){m_i=i;};

int class5::getJ(){return m_j;};

void class5::setJ(int j){m_j=j;};
