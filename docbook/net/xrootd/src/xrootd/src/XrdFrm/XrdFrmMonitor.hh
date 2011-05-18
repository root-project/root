#ifndef __XRDFRMMONITOR__
#define __XRDFRMMONITOR__
/******************************************************************************/
/*                                                                            */
/*                      X r d F r m M o n i t o r . h h                       */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

#include <inttypes.h>
#include <time.h>
#include <netinet/in.h>
#include <sys/types.h>

#include "XrdNet/XrdNetPeer.hh"
#include "XrdXrootd/XrdXrootdMonData.hh"
#include "XProtocol/XPtypes.hh"
  
/******************************************************************************/
/*                            X r d M o n i t o r                             */
/******************************************************************************/

#define XROOTD_MON_INFO     1
#define XROOTD_MON_STAGE    2

class XrdFrmMonitor
{
public:

static void              Defaults(char *dest1, int m1, char *dest2, int m2);

static int               Init();

static kXR_unt32         Map(const char code,const char *uname,const char *path);

static char              monSTAGE;

                         XrdFrmMonitor();
                        ~XrdFrmMonitor(); 

private:
static void              fillHeader(XrdXrootdMonHeader *hdr,
                                    const char id, int size);
static int               Send(int mmode, void *buff, int size);

static char              *Dest1;
static int                monMode1;
static int                monFD1;
static struct sockaddr    InetAddr1;
static char              *Dest2;
static int                monFD2;
static int                monMode2;
static struct sockaddr    InetAddr2;
static kXR_int32          startTime;
static int                isEnabled;
};
#endif
