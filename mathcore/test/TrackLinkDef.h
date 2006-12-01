#ifdef __CINT__


#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;



#pragma link C++ class TrackD+;
#pragma link C++ class TrackD32+;

#pragma extra_include "vector";
#include <vector>

#pragma link C++ class vector<ROOT::Math::XYZPoint >+;
#pragma link C++ class vector<TrackD >+;

#pragma link C++ class VecTrackD+;
#pragma link C++ class ClusterD+;

#endif
