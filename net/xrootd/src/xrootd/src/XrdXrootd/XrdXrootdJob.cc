/******************************************************************************/
/*                                                                            */
/*                       X r d X r o o t d J o b . c c                        */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdXrootdJobCVSID = "$Id$";

#include <stdio.h>
#include <string.h>
#include <netinet/in.h>
#include <sys/uio.h>

#include "Xrd/XrdLink.hh"
#include "Xrd/XrdScheduler.hh"
#include "XrdOuc/XrdOucProg.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdXrootd/XrdXrootdJob.hh"
#include "XrdXrootd/XrdXrootdResponse.hh"
#include "XrdXrootd/XrdXrootdTrace.hh"
#include "XProtocol/XProtocol.hh"
#include "XProtocol/XPtypes.hh"

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/

class XrdXrootdJob2Do : public XrdJob
{
public:
friend class XrdXrootdJob;

void      DoIt();

enum      JobStatus {Job_Active, Job_Cancel, Job_Done, Job_Waiting};

JobStatus Status;     //    Job Status

          XrdXrootdJob2Do(XrdXrootdJob      *job,
                          int                jnum,
                          const char       **args,
                          XrdXrootdResponse *Resp,
                          int                opts);
         ~XrdXrootdJob2Do();

private:
int          addClient(XrdXrootdResponse *rp, int opts);
void         delClient(XrdXrootdResponse *rp);
XrdOucTList *lstClient(void);
int          verClient(int dodel=0);
void         Destruct(int sendresp);
void         Redrive(void);
void         sendResult(char *lp, int caned=0);

static const int                     maxClients = 8;
struct {XrdLink     *Link;
        unsigned int Inst;
        kXR_char     streamid[2];
        char         isSync;
       }                             Client[maxClients];

       int                           numClients;

       XrdOucStream       jobStream;  // -> Stream for job I/O
       XrdXrootdJob      *theJob;     // -> Job description
       char              *theArgs[5]; // -> Program arguments (see XrdOucProg)
       char              *theResult;  // -> The result
       int                JobNum;     //    Job Number
       char               JobMark;
};
  
/******************************************************************************/
/*                      G l o b a l   F u n c t i o n s                       */
/******************************************************************************/

extern XrdOucTrace     *XrdXrootdTrace;
  
int XrdXrootdJobWaiting(XrdXrootdJob2Do *item, void *arg)
{
    return (item->Status == XrdXrootdJob2Do::Job_Waiting);
}

/******************************************************************************/
/*                 C l a s s   X r d X r o o t d J o b 2 D o                  */
/******************************************************************************/
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdXrootdJob2Do::XrdXrootdJob2Do(XrdXrootdJob      *job,
                                 int                jnum,
                                 const char       **args,
                                 XrdXrootdResponse *resp,
                                 int                opts) 
                                : XrdJob(job->JobName)
{
   int i;
   for (i = 0; i < 5 && args[i]; i++) theArgs[i] = strdup(args[i]);
   for (i = i; i < 5; i++)            theArgs[i] = (char *)0;
   theJob     = job;
   JobNum     = jnum;
   JobMark    = 0;
   numClients = 0;
   theResult  = 0;
   Status     = Job_Waiting;
   addClient(resp, opts);
}

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/

XrdXrootdJob2Do::~XrdXrootdJob2Do()
{
   int i;

   for (i = 0; i < numClients; i++) 
       if (!Client[i].isSync) {sendResult(0, 1); break;}

   for (i = 0; i < 5; i++) 
       if (theArgs[i]) free(theArgs[i]);
}

/******************************************************************************/
/*                                  D o I t                                   */
/******************************************************************************/
  
void XrdXrootdJob2Do::DoIt()
{
   char *lp;
   int i;

// While we were waiting to run we may have been cancelled. Check for this
//
   theJob->myMutex.Lock();
   if (Status == Job_Cancel) {Destruct(1); return;}

// Preform the actual function and get the result
//
   if (theJob->theProg->Run(&jobStream,theArgs[1],theArgs[2],theArgs[3],theArgs[4])) lp=0;
      else {theJob->myMutex.UnLock();
            lp = jobStream.GetLine();
            theJob->myMutex.Lock();
           }

// While we were waiting, this job may have been cancelled. Check it out
//
   if (Status == Job_Cancel) {Destruct(1); return;}

// Jobs is done. Send the result to any async clients waiting for it
//
   Status = Job_Done;
   for (i = 0; i < numClients; i++) 
       if (!Client[i].isSync) {sendResult(lp); break;}

// If the number of jobs > than the max allowed, then redrive a waiting job
//
   if (theJob->numJobs > theJob->maxJobs) Redrive();
   theJob->numJobs--;

// If there are no polling clients left, then delete ourselves
//
   if (numClients) 
      {theResult = lp;
       theJob->myMutex.UnLock();
      } else Destruct(0);
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                             a d d C l i e n t                              */
/******************************************************************************/
  
int XrdXrootdJob2Do::addClient(XrdXrootdResponse *rp, int opts)
{
   XrdLink *lp = rp->theLink();
   unsigned int Inst = lp->Inst();
   int i;

// Remove useless clients
//
   if (numClients >= maxClients) verClient();

// See if we are already here
//
   for (i = 0; i < numClients; i++)
       if (lp == Client[i].Link && Inst == Client[i].Inst) return 0;

// Add the client if we can
//
   if (numClients >= maxClients) return -1;
   Client[numClients].Link = lp;
   Client[numClients].Inst = Inst;
   if (opts & JOB_Sync) Client[numClients].isSync = 1;
      else {rp->StreamID(Client[numClients].streamid);
            Client[numClients].isSync = 0;
           }
   numClients++;
   JobMark = 0;
   return 1;
}

/******************************************************************************/
/*                             d e l C l i e n t                              */
/******************************************************************************/
  
void XrdXrootdJob2Do::delClient(XrdXrootdResponse *rp)
{
   XrdLink *lp = rp->theLink();
   unsigned int Inst = lp->Inst();
   int i, j;

// See if we are already here
//
   for (i = 0; i < numClients; i++)
       if (lp == Client[i].Link && Inst == Client[i].Inst)
          {for (j = i+1; j < numClients; j++) Client[i++] = Client[j];
           numClients--;
           break;
          }
}

/******************************************************************************/
/*                             l s t C l i e n t                              */
/******************************************************************************/
  
// Warning! The size of buff is large enough for the default number of clients
//          per job element.
//
XrdOucTList *XrdXrootdJob2Do::lstClient()
{
   char State, buff[4096], *bp = buff;
   int bsz, i, k;

// Get the state pf the job element
//
   switch(Status)
         {case Job_Active:  State = 'a'; break;
          case Job_Cancel:  State = 'c'; break;
          case Job_Done:    State = 'd'; break;
          case Job_Waiting: State = 'w'; break;
          default:          State = 'u'; break;
         };

// Insert the header (reserve 8 characters for the trailer)
//
   bp = buff + sprintf(buff, "<s>%c</s><conn>", State);
   bsz = sizeof(buff) - (bp - buff) - 8;

// Remove all clients from a job whose network connection is no longer valid
//
   if (!numClients) bp++;
      else for (i = 0; i < numClients; i++)
               if (Client[i].Link && Client[i].Link->isInstance(Client[i].Inst))
                  {if ((k = strlcpy(bp, Client[i].Link->ID, bsz)) >= bsz
                   || (bsz -= k) < 1) {bp++; break;}
                   bp += k; *bp = ' '; bp++; bsz--;
                  }

// Insert trailer
//
   if (*(bp-1) == ' ') bp--;
   strcpy(bp, "</conn>");

// Return the text
//
   return new XrdOucTList(buff, bp-buff+4);
}

/******************************************************************************/
/*                             v e r C l i e n t                              */
/******************************************************************************/
  
int XrdXrootdJob2Do::verClient(int dodel)
{
   int i, j, k;

// Remove all clients from a job whose network connection is no longer valid
//
   for (i = 0; i < numClients; i++)
       if (!Client[i].Link->isInstance(Client[i].Inst))
          {k = i;
           for (j = i+1; j < numClients; j++,k++) Client[k] = Client[j];
           numClients--; i--;
          }

// If no more clients, delete ourselves if safe to do so
//
   if (!numClients && dodel)
      {XrdXrootdJob2Do *jp = theJob->JobTable.Remove(JobNum);
       delete jp;
       return 0;
      }
   return numClients;
}
  
/******************************************************************************/
/*                              D e s t r u c t                               */
/******************************************************************************/
  
void XrdXrootdJob2Do::Destruct(int sendresp)
{
   XrdXrootdJob2Do *jp;

// Indicate this job was cancelled and delete ourselves. Note that the
// caller must make immediate return for this to not segv.
//
   if (sendresp) sendResult(0, 1);
   jp = theJob->JobTable.Remove(JobNum);
   theJob->myMutex.UnLock();
   delete jp;
   return;
}

/******************************************************************************/
/*                               R e d r i v e                                */
/******************************************************************************/
  
void XrdXrootdJob2Do::Redrive()
{
   XrdXrootdJob2Do *jp;

// Find the first waiting job
//

   while ((jp = theJob->JobTable.Apply(XrdXrootdJobWaiting, (void *)0)))
         if (jp->verClient(jp->JobMark > 0)) break;

// Schedule this job if we really have one here
//
   if (jp)
      {jp->Status = Job_Active;
       theJob->Sched->Schedule((XrdJob *)jp);
      }
}

/******************************************************************************/
/*                            s e n d R e s u l t                             */
/******************************************************************************/
  
void XrdXrootdJob2Do::sendResult(char *lp, int caned)
{
   const char *TraceID = "sendResult";
   const kXR_int32 Xcan  = static_cast<kXR_int32>(htonl(kXR_Cancelled));
   const kXR_int32 Xbad  = static_cast<kXR_int32>(htonl(kXR_ServerError));
   XrdXrootdReqID ReqID;
   struct iovec   jobVec[6];
   XResponseType  jobStat;
   const char    *trc, *tre;
   kXR_int32      erc;
   int            j, i, ovhd = 0, dlen = 0, n = 1;

// Format the message to be sent
//
   if (lp)
      {jobStat = kXR_ok; trc = "ok";
       if (theArgs[0])
          {        jobVec[n].iov_base = theArgs[0];                 // 1
           dlen  = jobVec[n].iov_len  = strlen(theArgs[0]); n++;
                   jobVec[n].iov_base = (char *)" ";                // 2
           dlen += jobVec[n].iov_len  = 1;                  n++;
          }
      } else {
       jobStat = kXR_error; trc = "error";
       if (caned) {erc = Xcan; lp = (char *)"Cancelled by admin.";}
          else    {erc = Xbad; lp = (char *)"Program failed.";}
                   jobVec[n].iov_base = (char *)&erc;
           dlen  = jobVec[n].iov_len  = sizeof(erc);        n++;    // 3
           ovhd = 1;
      }
                   jobVec[n].iov_base = lp;                         // 4
           dlen += jobVec[n].iov_len  = strlen(lp)+ovhd;    n++;

// Send the response to each client waiting for it
//
   j = 0;
   for (i = 0; i < numClients; i++)
       {if (!Client[i].isSync)
           {ReqID.setID(Client[i].streamid, 
                        Client[i].Link->FDnum(), Client[i].Link->Inst());
            tre = (XrdXrootdResponse::Send(ReqID, jobStat, jobVec, n, dlen) < 0
                ?  "skipped" : "sent");
            TRACE(RSP, tre <<" async " <<trc <<" to " <<Client[i].Link->ID);
           } else if (i != j) Client[j++] = Client[i];
        }
   numClients = j;
}

/******************************************************************************/
/*                    C l a s s   X r d X r o o t d J o b                     */
/******************************************************************************/
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdXrootdJob::XrdXrootdJob(XrdScheduler *schp, 
                           XrdOucProg   *pgm,
                           const char   *jname,
                           int           maxjobs)
                          : XrdJob("Job Scheduler"),
                            JobTable(maxjobs*3)
{
// Initialize the base member here
//
   Sched      = schp;
   theProg    = pgm;
   JobName    = strdup(jname);
   maxJobs    = maxjobs;
   numJobs    = 0;

// Schedule ourselves to run 15 minutes from now
//
    schp->Schedule((XrdJob *)this, time(0) + (reScan));
}
  
/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/

// Note! There is no reliable way to delete this object because various
// unsynchronized threads may be pending at various break points. Fortunately,
// there really is no need to ever delete an object of this kind.

XrdXrootdJob::~XrdXrootdJob()
{
   if (JobName) free(JobName);
   myMutex.Lock();
   Sched->Cancel((XrdJob *)this);
   myMutex.UnLock();
}

/******************************************************************************/
/*                                C a n c e l                                 */
/******************************************************************************/
  
int XrdXrootdJob::Cancel(const char *jkey, XrdXrootdResponse *resp)
{
   XrdXrootdJob2Do *jp = 0;
   int i, jNum, jNext = 0, numcaned = 0;

// Lock our data
//
   myMutex.Lock();

// Cancel a specific job if a key was passed
//
   if (jkey)
      {if ((jp = JobTable.Find(jkey)))
          {numcaned = 1;
           if (resp) {jp->delClient(resp);
                      if (!jp->numClients) CleanUp(jp);
                     }
               else  CleanUp(jp);
          }
       myMutex.UnLock();
       return numcaned;
      }

// Delete multiple jobs
//
   while((jNum = JobTable.Next(jNext)) >= 0)
        {jp = JobTable.Item(jNum);
         if (resp)
            {i = jp->numClients;
             jp->delClient(resp);
             if (i != jp->numClients) numcaned++;
             if (!jp->numClients) CleanUp(jp);
            } else {
             CleanUp(jp);
             numcaned++;
            }
        }

// All done
//
   myMutex.UnLock();
   return numcaned;
}

/******************************************************************************/
/*                                  D o I t                                   */
/******************************************************************************/
 
void XrdXrootdJob::DoIt()
{
   int jNum, jNext = 0;
   XrdXrootdJob2Do *jp;

// Scan through all of the jobs looking for disconnected clients
//
   while((jNum = JobTable.Next(jNext)) >= 0)
        {myMutex.Lock();
         if ((jp = JobTable.Item(jNum)))
            {if (jp->JobMark) {if (!jp->verClient()) CleanUp(jp);}
                else jp->JobMark = 1;
            }
         myMutex.UnLock();
        }

// Schedule ourselves to run 15 minutes from now
//
    Sched->Schedule((XrdJob *)this, time(0) + (reScan));
}

/******************************************************************************/
/*                                  L i s t                                   */
/******************************************************************************/
  
// Output: <job id="jname">%jobkey<s>%status</s><c>%clientid ...</c> ....</job>
//
XrdOucTList *XrdXrootdJob::List()
{
   char *jkey, buff[1024];
   int tlen, jNum, jNext = 0;
   XrdXrootdJob2Do *jp;
   XrdOucTList *tF = 0, *tL = 0, *tp;

// Scan through all of the jobs listing each, in turn
//
   while((jNum = JobTable.Next(jNext)) >= 0)
        {myMutex.Lock();
         if ((jp = JobTable.Item(jNum, &jkey)) && (tp = jp->lstClient()))
            {tlen = sprintf(buff, "<job id=\"%s\">%s", JobName, jkey);
            if (tL) tL->next = new XrdOucTList(buff, tlen, tp);
               else tF       = new XrdOucTList(buff, tlen, tp);
            tL = tp->next = new XrdOucTList("</job>", 6);
            }
         myMutex.UnLock();
        }

// Return the whole schmear
//
   return tF;
}

/******************************************************************************/
/*                              S c h e d u l e                               */
/******************************************************************************/
  
int XrdXrootdJob::Schedule(const char         *jkey,
                           const char        **args,
                           XrdXrootdResponse  *resp,
                           int                 Opts)
{
   XrdXrootdJob2Do *jp;
   const char *msg = "Job resources currently not available.";
   int jobNum, rc, isSync = Opts & JOB_Sync;

// First find if this is a duplicate or create a new one
//
   myMutex.Lock();
   if (!(Opts & JOB_Unique) && jkey && (jp = JobTable.Find(jkey)))
      {if (jp->Status == XrdXrootdJob2Do::Job_Done)
          {rc = sendResult(resp, args[0], jp);
           myMutex.UnLock();
           return rc;
          }
       if (jp->addClient(resp, Opts) < 0) isSync = 1;
          else msg = "Job scheduled.";
      } else {
       if ((jobNum = JobTable.Alloc()) < 0) isSync = 1;
          else {if ((jp = new XrdXrootdJob2Do(this, jobNum, args, resp, Opts)))
                   {JobTable.Insert(jp, jkey, jobNum);
                    if (numJobs < maxJobs)
                       {Sched->Schedule((XrdJob *)jp);
                        jp->Status = XrdXrootdJob2Do::Job_Active;
                       }
                    numJobs++; msg = "Job Scheduled";
                   }
               }
      }

// Tell the client to wait
//
   if (isSync) rc = resp->Send(kXR_wait, 30, msg);
      else     rc = resp->Send(kXR_waitresp, 600, "Job scheduled.");
   myMutex.UnLock();
   return rc;
}
 
/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                               C l e a n U p                                */
/******************************************************************************/
  
void XrdXrootdJob::CleanUp(XrdXrootdJob2Do *jp)
{
   int theStatus = jp->Status;

// Now we have to be careful. If the job is waiting or completed schedule 
// it for cancellation. If it's active then kill the associated process. The
// thread waiting for the result will see the cancellation. Otherwise, it
// already has been cancelled and is in the scheduled queue.
//
   jp->Status = XrdXrootdJob2Do::Job_Cancel;
        if (theStatus == XrdXrootdJob2Do::Job_Waiting
        ||  theStatus == XrdXrootdJob2Do::Job_Done)
           Sched->Schedule((XrdJob *)jp);
   else if (theStatus == XrdXrootdJob2Do::Job_Active)
           jp->jobStream.Drain();
}

/******************************************************************************/
/*                            s e n d R e s u l t                             */
/******************************************************************************/
  
int XrdXrootdJob::sendResult(XrdXrootdResponse *resp,
                             const char        *rpfx,
                             XrdXrootdJob2Do   *job)
{
   struct iovec jobResp[4];
   int dlen, i, rc;

// Send an error result if no result is present
//
   if (!(job->theResult)) rc = resp->Send(kXR_ServerError,"Program failed");
      else {if (!rpfx) {dlen = 0; i = 1;}
               else {        jobResp[1].iov_base = (char *)rpfx;
                     dlen  = jobResp[1].iov_len  = strlen(rpfx);
                             jobResp[2].iov_base = (char *)" "; 
                     dlen += jobResp[2].iov_len  = 1;
                     i = 3;
                    }
                   jobResp[i].iov_base = job->theResult;
           dlen += jobResp[i].iov_len  = strlen(job->theResult);
           rc = resp->Send(jobResp, i+1, dlen);
          }

// Remove the client from the job. Check if clean-up is required
//
   job->delClient(resp);
   if (!job->numClients) CleanUp(job);

// All done
//
   return rc;
}
