#ifndef CLASS1_H
#define CLASS1_H
class class1 {
public:
   class1();
   class1(int i, int j);
   ~class1();
   int getI();
   void setI(int i);
   int getJ();
   void setJ(int i);
private:
   int m_i;
   int m_j;
};
#endif
