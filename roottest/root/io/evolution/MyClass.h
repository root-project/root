
#include <TNamed.h>
#include <cstdio>

class MyClass : public TObject {
 public:
  MyClass();
  MyClass(int siz);
  ~MyClass() override;

  void check();

  int n;
#if (VERSION==2)
  float *arr; //[n];
#else
  float arr[20];
#endif

#if (VERSION==1)
  ClassDefOverride(MyClass, 1);
#endif
#if (VERSION==2)
  ClassDefOverride(MyClass, 2);
#endif

};

class Cont : public TObject {
public:
  MyClass data;
  Cont() {}
  Cont(int siz) : data(siz) {}
  ~Cont() override {}

  ClassDefOverride(Cont,1);
};


MyClass::MyClass() : n(0) {
#if (VERSION==2)
    arr = 0;
#else
    for(int i=0;i<20;i++) arr[i] = 0;
#endif
  }

MyClass::MyClass(int siz) : n(siz) {

#if (VERSION==2)
    if (n) arr = new float[n];
    else arr = 0;
#endif

    for(int i=0; i<n; i++) arr[i] = 10+i;

  }

MyClass::~MyClass() {
#if (VERSION==2)
    delete arr;
#endif
  }


void MyClass::check() {
    fprintf(stderr,"n is %d\n",n);

#if (VERSION==2)
    if (arr==0) fprintf(stderr,"no array\n");
    else {
#endif
      for(int i=0;i<n;i++) {
        if (arr[i]!=10+i) fprintf(stderr,"arr no properly created arr[%d]==%f\n",
                               i,arr[i]);
      }
#if (VERSION==2)
    }
#endif
  }
