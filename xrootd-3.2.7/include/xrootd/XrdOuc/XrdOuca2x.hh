#ifndef __OOUC_A2X__
#define __OOUC_A2X__
/******************************************************************************/
/*                                                                            */
/*                          X r d O u c a 2 x . h h                           */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

#include "XrdSys/XrdSysError.hh"

// This class is a holding area for various conversion utility routines
//

class XrdOuca2x
{
public:
static int a2i( XrdSysError &, const char *emsg, const char *item, int *val, int minv=-1, int maxv=-1);
static int a2ll(XrdSysError &, const char *emsg, const char *item, long long *val, long long minv=-1, long long maxv=-1);
static int a2fm(XrdSysError &, const char *emsg, const char *item, int *val, int minv);
static int a2fm(XrdSysError &, const char *emsg, const char *item, int *val, int minv, int maxv);
static int a2sp(XrdSysError &, const char *emsg, const char *item, long long *val, long long minv=-1, long long maxv=-1);
static int a2sz(XrdSysError &, const char *emsg, const char *item, long long *val, long long minv=-1, long long maxv=-1);
static int a2tm(XrdSysError &, const char *emsg, const char *item, int *val, int minv=-1, int maxv=-1);
static int a2vp(XrdSysError &, const char *emsg, const char *item, int *val, int minv=-1, int maxv=-1);

private:
static int Emsg(XrdSysError &Eroute, const char *etxt1, const char *item,
                                     const char *etxt2, int       val);
static int Emsg(XrdSysError &Eroute, const char *etxt1, const char *item,
                                     const char *etxt2, long long val);
};

#endif
