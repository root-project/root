#ifndef ROOT7_RDataFrame_Test_NTupleStruct
#define ROOT7_RDataFrame_Test_NTupleStruct

#include <vector>

/**
 * Used to test serialization and deserialization of classes in RNTuple with TClass
 */
struct Electron {
   float pt;
};

struct Jet {
   std::vector<Electron> electrons;
};

#endif
