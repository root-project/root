#include "TestHelpers.h"
#include "TestOutput.h"

#ifdef __MAKECINT__
#pragma link C++ function DebugTest;
#pragma link C++ class pair<float,int>+;
#pragma link C++ class pair<std::string,double>+;
#pragma link C++ class GHelper<float>+;
#pragma link C++ class GHelper<GHelper<float> >+;
#pragma link C++ class GHelper<GHelper<GHelper<float> > >+;
#endif
