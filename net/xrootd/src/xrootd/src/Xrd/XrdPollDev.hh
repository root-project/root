#ifndef __XRD_POLLDEV_H__
#define __XRD_POLLDEV_H__
/******************************************************************************/
/*                                                                            */
/*                         X r d P o l l D e v . h h                          */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//        $Id$

#include <poll.h>

#include "Xrd/XrdPoll.hh"
  
class XrdPollDev : public XrdPoll
{
public:

       void Disable(XrdLink *lp, const char *etxt=0);

       int   Enable(XrdLink *lp);

       void Start(XrdSysSemaphore *syncp, int &rc);

            XrdPollDev(struct pollfd *ptab, int numfd, int pfd)
                       {PollTab = ptab; PollMax = numfd; PollDfd = pfd;}
           ~XrdPollDev();

protected:
       void Exclude(XrdLink *lp);
       int  Include(XrdLink *lp) {return 1;}

private:

void doRequests(int maxreq);
void LogEvent(struct pollfd *pp);
int  sendCmd(char *cmdbuff, int cmdblen);

struct pollfd *PollTab;
       int     PollDfd;
       int     PollMax;
};
#endif
