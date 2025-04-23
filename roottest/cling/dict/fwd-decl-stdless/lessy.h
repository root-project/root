#include <functional>

namespace edm {
  struct AJet {
    float m_E;
  };
} // edm

namespace std {
  template <>
  struct less<edm::AJet> {
    bool operator()(edm::AJet one, edm::AJet two) const {
      return one.m_E < two.m_E;
    }
  };
}
