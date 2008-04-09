#ifndef __XRDXROOTDJOB_HH_
#define __XRDXROOTDJOB_HH_
/******************************************************************************/
/*                                                                            */
/*                       X r d X r o o t d J o b . h h                        */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <sys/types.h>
  
#include "Xrd/XrdJob.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdOuc/XrdOucTable.hh"

class XrdOucProg;
class XrdLink;
class XrdScheduler;
class XrdXrootdJob2Do;
class XrdXrootdResponse;

// Definition of options that can be passed to Schedule()
//
#define JOB_Sync   0x0001
#define JOB_Unique 0x0002

class XrdXrootdJob : public XrdJob
{
friend class XrdXrootdJob2Do;
public:

int      Cancel(const char *jkey=0, XrdXrootdResponse *resp=0);

void     DoIt();

// List() returns a list of all jobs in xml format
//
XrdOucTList *List(void);

// args[0]   if not null if prefixes the response
// args[1-n] are passed to the prgram
// The return value is whatever resp->Send() returns
//
int      Schedule(const char         *jkey,   // Job Identifier
                  const char        **args,   // Zero terminated arglist
                  XrdXrootdResponse  *resp,   // Response object
                  int                 Opts=0);// Options (see above)

         XrdXrootdJob(XrdScheduler *schp,       // -> Scheduler
                      XrdOucProg   *pgm,        // -> Program Object
                      const char   *jname,      // -> Job name
                      int           maxjobs=4); // Maximum simultaneous jobs
        ~XrdXrootdJob();

private:
void CleanUp(XrdXrootdJob2Do *jp);
int  sendResult(XrdXrootdResponse *resp,
                const char        *rpfx,
                XrdXrootdJob2Do   *job);

static const int              reScan = 15*60;

XrdSysMutex                   myMutex;
XrdScheduler                 *Sched;
XrdOucTable<XrdXrootdJob2Do>  JobTable;
XrdOucProg                   *theProg;
char                         *JobName;
int                           maxJobs;
int                           numJobs;
};
#endif
