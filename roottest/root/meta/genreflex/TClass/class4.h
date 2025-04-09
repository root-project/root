#ifndef CLASS4_H
#define CLASS4_H


class class4 {

public:
   class4(int i, int j);
   class4();
   ~class4();
   int getI();
   void setI(int i);
   int getJ();
   void setJ(int i);
   double* getArr();
private:
   int m_i;
   int m_j;
   double* m_dArray; //[m_i]
};
#endif


class4::class4(int i, int j):
  m_i(i),
  m_j(j){
};

class4::class4():
   m_i(10),
   m_j(10){
   };

class4::~class4(){};

int class4::getI(){return m_i;};

void class4::setI(int i){m_i=i;};

int class4::getJ(){return m_j;};

void class4::setJ(int j){m_j=j;};

double* class4::getArr(){return m_dArray;}
