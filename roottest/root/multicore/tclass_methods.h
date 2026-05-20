#ifndef DataFormats_TestObjects_ToyProducts_h
#define DataFormats_TestObjects_ToyProducts_h

// A MOCK FROM CMSSW: CMSSW_7_5_X/DataFormats/TestObjects/interface/ToyProducts.h
using cms_int32_t=int;

/*----------------------------------------------------------------------

Toy EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <vector>

namespace edmtest {

  // Toy products

  // DP: many lines removed

  struct Simple {
    Simple() : key(0), value(0.0) {}
    virtual ~Simple(){};
    typedef cms_int32_t key_type;
    key_type    key;
    double      value;
    key_type id() const { return key; }
    virtual Simple* clone() const  { return new Simple(*this); };
  };

  inline
  bool
  operator==(Simple const& a, Simple const& b) {
    return(a.key == b.key && a.value == b.value);
  }

  inline
  bool operator<(Simple const& a, Simple const& b) {
    return a.key < b.key;
  }

  // DP: many lines removed

}
#endif
