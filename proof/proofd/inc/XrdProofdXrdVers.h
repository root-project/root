//
// File defining Xrootd version numbers corresponding to API changes
// relevant for ROOT modules. The versions are expressed following
// the internal conventions defined by configure

#ifndef ROOT_XrdProofdXrdVers
#define ROOT_XrdProofdXrdVers

// This is when the PhyConn arguments constructor changed
#define ROOT_OldPhyConn      20070628    // Not used anylonger

// This is when some headers moved from XrdOuc to XrdSys
#define ROOT_OldXrdOuc       20070723    // Not used anylonger

// This is when the XrdClientAdmin::Locate interface changed
#define ROOT_OldXrdLocate    20071004    // Not used anylonger

// This is when libXrdNetUtil appeared
#define ROOT_XrdNetUtil      20100729

// This is when the PhyConn interface changed (v3.0.2)
#define ROOT_PhyConnNoReuse  20110126

// This is when the header and library packaging changed (v3.1.0)
#define ROOT_XrdUtils        20111021

#endif
