#ifndef __XRDOUCREQID__
#define __XRDOUCREQID__
/******************************************************************************/
/*                                                                            */
/*                        X r d O u c R e q I D . h h                         */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <stdlib.h>
#ifndef WIN32
#include <strings.h>
#endif

#include "XrdSys/XrdSysPthread.hh"
  
class XrdOucReqID
{
public:

static char *ID(char *buff, int blen); // blen >= 48

static int   isMine(char *reqid)
             {return !strncmp((const char *)reqPFX,(const char *)reqid,reqPFXlen);}

static int   isMine(char *reqid, int &hport, char *hname, int hlen);

static char *PFX() {return reqPFX;}

static int   Index(int KeyMax, const char *KeyVal, int KeyLen=0);

             XrdOucReqID(int instance, const char *myhost, unsigned int myaddr);
            ~XrdOucReqID() {} // Statics go away at exit

private:

static XrdSysMutex  myMutex;
static int          reqPFXlen;
static char        *reqPFX;
static char        *reqFMT;
static int          reqNum;
};
#endif
