#ifndef ROOT7_RNTuple_Test_StreamerFieldXML
#define ROOT7_RNTuple_Test_StreamerFieldXML

#include <Rtypes.h>

#include <vector>

// TClass reports this class to be unsplittable due to its custom streamer; we enforce it to be stored in
// RNTuple nevertheless.
struct ForceNativeXML {
   float a;
   ClassDefNV(ForceNativeXML, 1);
};

// A begning and supported class, which we force to be stored in streamer mode in RNTuple
struct ForceStreamedXML {
   float a;
};

#endif // ROOT7_RNTuple_Test_StreamerFieldXML
