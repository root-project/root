#include "LinkDefDirectory/Wrapper.h"

#ifdef __ROOTCLING__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ class RootclingDirectory::DirectPayload + ;
#endif
