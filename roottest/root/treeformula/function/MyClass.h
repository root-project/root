#ifndef MYCLASS_H
#define MYCLASS_H

#include <TObject.h>
#include <TF1.h>


//_____________________________________________________________________________
class MyClass : public TObject {
 public:
  MyClass();

  ~MyClass() override {
    delete fgWSb;
  }

  void Init();
  void Integral(Double_t a, Double_t b);

  static Double_t WSb(Double_t *xx, Double_t *par);

  
 private:

  static TF1*    fgWSb;            // Wood-Saxon Function (b)
    
  ClassDefOverride(MyClass,1)  // Test
};

#endif


#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
 
#pragma link C++ class  MyClass+;

#endif
