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

#include <sys/epoll.h>

#include "Xrd/XrdPoll.hh"
  
class XrdPollE : public XrdPoll
{
public:

       void Disable(XrdLink *lp, const char *etxt=0);

       int   Enable(XrdLink *lp);

       void Start(XrdSysSemaphore *syncp, int &rc);

            XrdPollE(struct epoll_event *ptab, int numfd, int pfd)
                       {PollTab = ptab; PollMax = numfd; PollDfd = pfd;}
           ~XrdPollE();

protected:
       void  Exclude(XrdLink *lp);
       int   Include(XrdLink *lp);
       char *x2Text(unsigned int evf);

private:
void remFD(XrdLink *lp, unsigned int events);

#ifdef EPOLLONESHOT
   static const int ePollOneShot = EPOLLONESHOT;
#else
   static const int ePollOneShot = 0;
#endif
   static const int ePollEvents = EPOLLIN  | EPOLLHUP | EPOLLPRI | EPOLLERR |
                                  ePollOneShot;

struct epoll_event *PollTab;
       int          PollDfd;
       int          PollMax;
};
#endif
