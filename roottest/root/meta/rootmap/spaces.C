template <class T> class A {
  public:
  A(const T& i) : a(i) {}
  T a;
};

#ifdef __CINT__
#pragma link C++ class A<int>;
#pragma link C++ class A<unsigned int>;
#endif


