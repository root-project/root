#include "TObject.h"
template <class AParamType>
class Parameter : public TObject {

private:
   AParamType  fVal;
   ClassDef(Parameter,1);
};

#ifdef __MAKECINT__
#pragma link C++ class Parameter<Long64_t>+;
#pragma link C++ class Parameter<ULong64_t>+;
#pragma link C++ class Parameter<long long>+;
#pragma link C++ class Parameter<unsigned long long>+;
//#pragma link C++ class TParameter<const int>+;
#endif
