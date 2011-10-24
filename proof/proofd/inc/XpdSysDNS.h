
//
// These are needed to handle backward compatibility
//
#include "XrdProofdXrdVers.h"

#if ROOTXRDVERS < ROOT_XrdUtils
#define XrdSysDNS           XrdNetDNS
#include "XrdNet/XrdNetDNS.hh"
#else
#include "XrdSys/XrdSysDNS.hh"
#endif


