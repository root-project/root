void j();


#ifndef __CINT__
#define INST  template <class Q> friend class analyzer;
// template <class Q> friend class analyzer;
#define private public: template <class Q> friend class analyzer; private
#endif

class top {
  //public: template <class T> class analyzer {};
  //friend class analyzer<top>;
  //  INST
private:
  int priv;
protected:
  int prot;
public:
  int pub;
  //private:
  int priv2;
};

template <> class analyzer<top> {
public:

  /*
  template <class T, class Q, Q T::* dd > void * GetAddress(T * obj) {
    return &( obj->member );
  }
  */
  void * GetAddress(top * obj) {
    return &( obj->priv );
  }

};

class top2 {
private:
  int priv;
protected:
  int prot;
  //private:
  int priv2;
public:
  int pub;
  friend void j();
  //  friend void j();
  virtual void get() {};
};

class bottom : public top {
  friend void j();

};

#include <iostream>
using namespace std;

void j() {
  top *t = new top;
  cerr << "object : " << t;

  void * addr = &(t->pub);
  cerr << " pub: " << addr;

  addr = &(((bottom*)t)->pub);
  cerr << " pub: " << addr;

  addr = &(((bottom*)t)->prot);
  cerr << " prot: " << addr;

  analyzer<top> z;
  addr = z.GetAddress( t );
    //&(((bottom*)t)->priv);
  cerr << " priv: " << addr;

  cerr << endl;

  cerr << "Virtual: " << sizeof(top2)
       << " non virtual: " << sizeof(top)
       << endl;
  
}
