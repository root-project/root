#include <vector>

namespace edm {
struct refhelper {
template <typename what>
struct ValueTrait {
  typedef typename what::value_type value;
};
};

template <typename what,
	  typename trait = typename refhelper::ValueTrait<what>::value >
class Ref {
public:
   trait fValue;
};
}

#ifdef __ROOTCLING__
#pragma link C++ class edm::Ref<vector<Double32_t> >+;
#endif

#include "TClass.h"
int execdefaultTemplateParam()
{
   TClass *c = TClass::GetClass("edm::Ref<vector<Double32_t> >");
   if (c) return 0;
   else return 1;
}

