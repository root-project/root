#include "class1.h"

//#ifndef __MAKECINT__
class1::class1(int i, int j):
  m_i(i),
  m_j(j){};
class1::class1():
   m_i(0),
   m_j(0){};
class1::~class1(){};
int class1::getI(){return m_i;};
void class1::setI(int i){m_i=i;};
int class1::getJ(){return m_j;};
void class1::setJ(int j){m_j=j;};
//#endif