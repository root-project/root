#ifndef CLASS3_H
#define CLASS3_H

#include "class3_1.h"

class class3 {

public:
class3(int i, int j):
  m_i(i),
  m_j(j){};
class3():
   m_i(0),
   m_j(0){};
~class3(){};
int getI(){return m_i;};
void setI(int i){m_i=i;};
int getJ(){return m_j;};
void setJ(int j){m_j=j;};

private:
   int m_i;
   int m_j;
   class3_1 m_class3_1_obj;
};
#endif
