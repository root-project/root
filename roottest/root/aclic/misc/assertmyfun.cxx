#include "TObject.h"
#include "Fit/Chi2FCN.h"

class MyFCN: public ROOT::Fit::Chi2Function{
private:
    MyFCN(); // Not implemented
public:
    MyFCN(const ROOT::Fit::BinData&  data, const ROOT::Fit::Chi2Function::IModelFunction&  func): ROOT::Fit::Chi2Function(data,func) {}
    ~MyFCN(){}
};

#ifdef __MAKECINT__
#pragma link C++ class MyFCN-;
#endif

int assertmyfun() {
  return 0;
}
