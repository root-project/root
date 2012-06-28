/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef VBASE_H
#define VBASE_H

#include <stddef.h>
#include <iostream>
using namespace std;

double d = 1.25;
int    i = 100;

class T {
 public:
  int dummy;
};

class A {
 public:
  double ad;
  int    ai;
  A() { 
    ad = d ; 
    ai = i++;
    d += 0.25;
    disp();
  }	
  // virtual ~A() {disp();}
  ~A() {disp();}
  void disp() {
    cout << " ad=" << ad << " ai=" << ai ;
  }
};

class B : virtual public A {
 public:
  double bd;
  int    bi;
  B(double din,int iin) { 
    bd = din ; 
    bi = iin ;
    disp();
  }	
  ~B() {disp();}
  void disp() {
    A::disp();
    cout << " bd=" << bd << " bi=" << bi ;
  }
};


class C : virtual public A {
 public:
  double cd;
  int    ci;
  C(double din,int iin) { 
    cd = din ; 
    ci = iin ;
    disp();
  }	
  ~C() {disp();}
  void disp() {
    A::disp();
    cout << " cd=" << cd << " ci=" << ci ;
  }
};


class D : public B, public C {
 public:
  double dd;
  int    di;
  D(double din,int iin) : B(din+10.0,iin+100) , C(din+100.0,iin+1000) { 
    dd = din ; 
    di = iin ;
    disp();
  }	
  ~D() {disp();}
  void disp() {
    A::disp();
    B::disp();
    C::disp();
    cout << " dd=" << dd << " di=" << di ;
  }
};

class E : public B, public C , virtual public A {
 public:
  double ed;
  int    ei;
  E(double din,int iin) : B(din+10.0,iin+100) , C(din+100.0,iin+1000) { 
    ed = din ; 
    ei = iin ;
    disp();
  }	
  ~E() {disp();}
  void disp() {
    A::disp();
    B::disp();
    C::disp();
    cout << " ed=" << ed << " ei=" << ei ;
  }
};

class F : virtual public A , public B, public C {
 public:
  double fd;
  int    fi;
  F(double din,int iin) : B(din+10.0,iin+100) , C(din+100.0,iin+1000) { 
    fd = din ; 
    fi = iin ;
    disp();
  }	
  ~F() {disp();}
  void disp() {
    A::disp();
    B::disp();
    C::disp();
    cout << " fd=" << fd << " fi=" << fi ;
  }
};

class G : public T , public B, public C {
 public:
  double gd;
  int    gi;
  G(double din,int iin) : B(din+10.0,iin+100) , C(din+100.0,iin+1000) { 
    gd = din ; 
    gi = iin ;
    disp();
  }	
  ~G() {disp();}
  void disp() {
    A::disp(); 
    // B::A::disp();  // this only works for cint, compiler fails
    // C::A::disp();  // this only works for cint, compiler fails
    B::disp();
    C::disp();
    cout << " gd=" << gd << " gi=" << gi ;
  }
};

void btest() {
 cout << "BTEST==========================" << endl;
 B b1(0,0); cout << endl;
 B b2(10000.0,100000); cout<<endl;
 b1.disp(); 
 cout << endl;
 b2.disp();
 cout << endl;
}

void ctest() {
 cout << endl;
 cout << "CTEST==========================" << endl;
 C c1(0,0); cout<<endl;
 C c2(10000.0,100000); cout<<endl;
 c1.disp(); 
 cout << endl;
 c2.disp();
 cout << endl;
}

void dtest() {
 cout << endl;
 cout << "DTEST==========================" << endl;
 D d1(0,0); cout<<endl;
 D d2(10000.0,100000); cout<<endl;
 d1.disp(); 
 cout << endl;
 d2.disp();
 cout << endl;
}

void etest() {
 cout << endl;
 cout << "ETEST==========================" << endl;
 E e1(0,0); cout<<endl;
 E e2(10000.0,100000); cout<<endl;
 e1.disp(); 
 cout << endl;
 e2.disp();
 cout << endl;
}

void ftest() {
 cout << endl;
 cout << "ETEST==========================" << endl;
 F f1(0,0); cout<<endl;
 F f2(10000.0,100000); cout<<endl;
 f1.disp(); 
 cout << endl;
 f2.disp();
 cout << endl;
}

void gtest() {
 cout << endl;
 cout << "ETEST==========================" << endl;
 G g1(0,0); cout<<endl;
 G g2(10000.0,100000); cout<<endl;
 g1.disp(); 
 cout << endl;
 g2.disp();
 cout << endl;
}

#endif
