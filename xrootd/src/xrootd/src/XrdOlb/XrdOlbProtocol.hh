#ifndef __OLB_PROTOCOL_H__
#define __OLB_PROTOCOL_H__
/******************************************************************************/
/*                                                                            */
/*                     X r d O l b P r o t o c o l . h h                      */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

#include "Xrd/XrdProtocol.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdOlbProtocol : public XrdProtocol
{
public:

static XrdOlbProtocol *Alloc();

       void            DoIt() {}             // Protocol never rescheduled

       XrdProtocol    *Match(XrdLink *lp);   // Upon    accept

       int             Process(XrdLink *lp); // Initial entry

       void            Recycle(XrdLink *lp, int consec, const char *reason);

static void            setNet(XrdInet *net, int rwt)
                             {myNet = net; readWait = rwt;}

       int             Stats(char *buff, int blen, int do_sync=0) {return 0;}

              XrdOlbProtocol() : XrdProtocol("olb protocol handler") 
                               {ProtLink = 0;}
             ~XrdOlbProtocol() {}

private:
static XrdInet        *myNet;
static int             readWait;
static XrdSysMutex     ProtMutex;
static XrdOlbProtocol *ProtStack;
       XrdOlbProtocol *ProtLink;
};
#endif
