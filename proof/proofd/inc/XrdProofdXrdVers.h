//
// File defining Xrootd version numbers corresponding to API changes
// relevant for ROOT modules. The versions are expressed following
// the internal conventions defined by configure

#ifndef ROOT_XrdProofdXrdVers
#define ROOT_XrdProofdXrdVers

// This is when the PhyConn arguments constructor changed
#define ROOT_OldPhyConn      20070628

// This is when some headers moved from XrdOuc to XrdSys
#define ROOT_OldXrdOuc       20070723

// This is when the XrdClientAdmin::Locate interface changed
#define ROOT_OldXrdLocate    20071004

// This is when libXrdNetUtil appeared
#define ROOT_XrdNetUtil      20100729

// This is when the PhyConn interface changed
#define ROOT_PhyConnNoReuse  300000002

// This is when the header and library packaging changed
#define ROOT_XrdUtils        300010000

#endif
