#ifndef CLASS3_2_H
#define CLASS3_2_H


class class3_2 {

public:
   class3_2(int i);
   class3_2();
   ~class3_2();
   int getI();
   void setI(int i);
private:
   int m_i;
};
#endif

class3_2::class3_2(int i):
  m_i(i){};

class3_2::class3_2():
   m_i(0){};

class3_2::~class3_2(){};

int class3_2::getI(){return m_i;};

void class3_2::setI(int i){m_i=i;};

