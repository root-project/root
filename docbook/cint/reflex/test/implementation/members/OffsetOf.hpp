#include <iostream>

// Indirectly test TypeBuilder's OffsetOf macro.
// See https://savannah.cern.ch/bugs/?33071

struct UnNamedMember {
   char fPadding[16];
   struct {} fUnNamed;
};

struct HasAddrOfOp {
   HasAddrOfOp* operator&() const {
      std::cout << "BAD: Address-of operator got called!" << std::endl;
      return 0;
   }
};

class HasAddrOfOp_Base: public HasAddrOfOp {};

class Test_OffsetOf_With_AddrOfOp {
 public:
   HasAddrOfOp fHasAddrOfOp;
   HasAddrOfOp_Base fHasAddrOfOp_Base;
};
