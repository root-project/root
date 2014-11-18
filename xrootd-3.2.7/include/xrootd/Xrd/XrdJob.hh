#ifndef ___XRD_JOB_H___
#define ___XRD_JOB_H___
/******************************************************************************/
/*                                                                            */
/*                             X r d J o b . h h                              */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$ 

#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
  
// The XrdJob class is a super-class that is inherited by any class that needs
// to schedule work on behalf of itself. The XrdJob class is optimized for
// queue processing since that's where it spends a lot of time. This class
// should not be depedent on any other class.

class XrdJob
{
friend class XrdScheduler;
public:
XrdJob    *NextJob;   // -> Next job in the queue (zero if last)
const char *Comment;   // -> Description of work for debugging (static!)

virtual void  DoIt() = 0;

              XrdJob(const char *desc="")
                    {Comment = desc; NextJob = 0; SchedTime = 0;}
virtual      ~XrdJob() {}

private:
time_t      SchedTime; // -> Time job is to be scheduled
};
#endif
