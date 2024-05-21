
//
// These are needed to handle backward compatibility
//
#include "XrdProofdXrdVers.h"

#if defined(ROOT_XrdNoUtils)
#define XrdSysDNS           XrdNetDNS
#include "XrdNet/XrdNetDNS.hh"
#else
#include "XrdSys/XrdSysDNS.hh"
#endif


