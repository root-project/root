#include "TObject.h"
template <class AParamType>
class TParameter : public TObject {

private:
   AParamType  fVal;
   ClassDef(TParameter,1);
};

#ifdef __MAKECINT__
#pragma link C++ class TParameter<Long64_t>+;
#pragma link C++ class TParameter<ULong64_t>+;
#pragma link C++ class TParameter<long long>+;
#pragma link C++ class TParameter<unsigned long long>+;
//#pragma link C++ class TParameter<const int>+;
#endif
