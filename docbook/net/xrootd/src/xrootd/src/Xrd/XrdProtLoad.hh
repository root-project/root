#ifndef __XrdProtLoad_H__
#define __XrdProtLoad_H__
/******************************************************************************/
/*                                                                            */
/*                        X r d P r o t L o a d . h h                         */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include "Xrd/XrdProtocol.hh"

class XrdSysPlugin;
  
// This class load and allows the selection of the appropriate link protocol. 
//
class XrdProtLoad : public XrdProtocol
{
public:

void          DoIt() {}

static int    Load(const char *lname, const char *pname, char *parms,
                   XrdProtocol_Config *pi);

static int    Port(const char *lname, const char *pname, char *parms,
                   XrdProtocol_Config *pi);

XrdProtocol  *Match(XrdLink *) {return 0;}

int           Process(XrdLink *lp);

void          Recycle(XrdLink *lp, int ctime, const char *txt);

int           Stats(char *buff, int blen, int do_sync=0);

              XrdProtLoad(int port=-1);
             ~XrdProtLoad();

static const int ProtoMax = 8;

private:

static XrdProtocol *getProtocol    (const char *lname, const char *pname,
                                    char *parms, XrdProtocol_Config *pi);
static int          getProtocolPort(const char *lname, const char *pname,
                                    char *parms, XrdProtocol_Config *pi);

static char          *ProtName[ProtoMax];   // ->Supported protocol names
static XrdProtocol   *Protocol[ProtoMax];   // ->Supported protocol objects
static int            ProtPort[ProtoMax];   // ->Supported protocol ports
static XrdProtocol   *ProtoWAN[ProtoMax];   // ->Supported protocol objects WAN
static int            ProtoCnt;             // Number in table (at least 1)
static int            ProtWCnt;             // Number in table (WAN may be 0)

static char          *liblist[ProtoMax];    // -> Path used for shared library
static XrdSysPlugin  *libhndl[ProtoMax];    // -> Plugin object
static int            libcnt;               // Number in table

       int            myPort;
};
#endif
