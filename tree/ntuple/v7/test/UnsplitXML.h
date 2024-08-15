#ifndef ROOT7_RNTuple_Test_Unsplit
#define ROOT7_RNTuple_Test_Unsplit

#include <Rtypes.h>

#include <vector>

// TClass reports this class to be unsplittable due to its custom streamer; we enforce it to be stored in
// RNTuple nevertheless.
struct ForceSplitXML {
   float a;
   ClassDefNV(ForceSplitXML, 1);
};

// A begning and supported class, which we force to be stored in unsplit mode in RNTuple
struct ForceUnsplitXML {
   float a;
};

#endif // ROOT7_RNTuple_Test_Unsplit
