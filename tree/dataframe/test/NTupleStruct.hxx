#ifndef ROOT7_RDataFrame_Test_NTupleStruct
#define ROOT7_RDataFrame_Test_NTupleStruct

#include <set>
#include <vector>

/**
 * Used to test serialization and deserialization of classes in RNTuple with TClass
 */
struct Electron {
   float pt;

   friend bool operator<(const Electron &left, const Electron &right) { return left.pt < right.pt; }
};

struct Jet {
   std::vector<Electron> electrons;
};

#endif
