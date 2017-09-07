#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class TrackD + ;
#pragma link C++ class TrackD32 + ;
#pragma link C++ class TrackErrD + ;
#pragma link C++ class TrackErrD32 + ;

#pragma extra_include "vector";
#include <vector>

#pragma link C++ class VecTrack < TrackD > +;
#pragma link C++ class VecTrack < TrackErrD > +;

#pragma link C++ class std::vector < TrackD > +;
#pragma link C++ class std::vector < TrackErrD > +;

// typedefs must also be defined in the dictionaries
#pragma link C++ typedef Vector4D_t;
#pragma link C++ typedef Vector4D32_t;

#pragma link C++ typedef Point3D_t;
#pragma link C++ typedef Point3D32_t;

#pragma link C++ typedef Matrix4D_t;
#pragma link C++ typedef Matrix4D32_t;

#pragma link C++ typedef SymMatrix6D_t;
#pragma link C++ typedef SymMatrix6D32_t;
