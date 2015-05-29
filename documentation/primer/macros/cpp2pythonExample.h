#include "stdio.h"

class A{
public:
 A(int i):m_i(i){}
 int getI() const {return m_i;}
private:
 int m_i=0;
};

void printA(const A& a ){
  printf ("The value of A instance is %i.\n",a.getI());
}
