/*****************************************************************************/
/*                                                                           */
/*                              XrdMonUtils.hh                               */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONUTILS_HH
#define XRDMONUTILS_HH

#include "XProtocol/XPtypes.hh"
#include <string>
#include <utility>
using std::pair;
using std::string;

enum { GMT = 1 };


extern string generateTimestamp();
extern string timestamp2string(kXR_int32 t, bool gmt=false);
extern void   timestamp2string(kXR_int32 t, char s[24], bool gmt);
extern pair<string, string> breakHostPort(const string& hp);
extern void mkdirIfNecessary(const char* dir);


#endif /* XRDMONUTILS_HH */
