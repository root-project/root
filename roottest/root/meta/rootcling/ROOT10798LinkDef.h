#include <map>
namespace SH {
  class SamplePtr;
}
struct HasUnknown {
  std::map<string,SH::SamplePtr> fUnknownValueType;
};
#pragma link C++ class HasUnknown+;
