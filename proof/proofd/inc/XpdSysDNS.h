
//
// These are needed to handle backward compatibility
//

#ifndef HAVEXRDSYSDNS
#define XrdSysDNS           XrdNetDNS
#include "XrdNet/XrdNetDNS.hh"
#else
#include "XrdSys/XrdSysDNS.hh"
#endif


