#ifndef CLASS1_H
#define CLASS1_H
class class1 {

public:
   class1():
        m_i(0), m_j(0){};
   class1(int i, int j):
  	m_i(i),
	m_j(j){};
   ~class1(){};
  int getI(){return m_i;};
  void setI(int i){m_i=i;};
  int getJ(){return m_j;};
  void setJ(int j){m_j=j;};

private:
   int m_i;
   int m_j;
};
#endif

