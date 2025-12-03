//
//  This is a dumb template to prove a point

template<class T>
class Test {
public:
  Test(const T& t) : mData(t) {}
  T Get(void) {return mData;}
  void Set(const T& t) {mData=t;}
  bool equal(const T& t) {return t==mData;}
private:
  T mData;
};

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class Test<bool>;
//#pragma link C++ class Test<double>;
//#pragma link C++ class Test<int>;

#endif
