/******************************************************************************/
/*                                                                            */
/*                      X r d C m s P r e p a r e . c c                       */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

// Original Version: 1.11 2007/08/08 19:18:47 abh

const char *XrdCmsPrepareCVSID = "$Id$";
  
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdCms/XrdCmsConfig.hh"
#include "XrdCms/XrdCmsPrepare.hh"
#include "XrdCms/XrdCmsTrace.hh"
#include "XrdFrm/XrdFrmProxy.hh"
#include "XrdNet/XrdNetMsg.hh"
#include "XrdOss/XrdOss.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucMsubs.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdSys/XrdSysError.hh"

using namespace XrdCms;

/******************************************************************************/
/*                        S t a t i c   O b j e c t s                         */
/******************************************************************************/
  
XrdCmsPrepare   XrdCms::PrepQ;

/******************************************************************************/
/*          G l o b a l s   &   E x t e r n a l   F u n c t i o n s           */
/******************************************************************************/

// This function is applied to all prepare queue entries. It checks if the file
// in online and if so, returns a -1 to delete the entry from the queue. O/W
// it returns a zero which keeps the entry in the queue. The key is the LFN.
//
int XrdCmsScrubScan(const char *key, char *cip, void *xargp)
{
   struct stat buf;

// Use oss interface to determine whether the file exists or not
//
   return (Config.ossFS->Stat(key, &buf, XRDOSS_resonly) ? 0 : -1);
}

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdCmsPrepare::XrdCmsPrepare() : XrdJob("Prep cache scrubber"),
                                 prepSched(&Say)
{prepif   = 0;
 preppid  = 0;
 resetcnt = scrub2rst = 3;
 scrubtime= 20*60;
 NumFiles = 0;
 lastemsg = time(0);
 Relay    = new XrdNetMsg(&Say);
 PrepFrm  = 0;
 prepOK   = 0;
}

/******************************************************************************/
/*                                   A d d                                    */
/******************************************************************************/
  
int XrdCmsPrepare::Add(XrdCmsPrepArgs &pargs)
{
   char *pdata[XrdOucMsubs::maxElem+2], prtybuff[8], *pP=prtybuff;
   int rc, pdlen[XrdOucMsubs::maxElem + 2];

// Check if we are using the built-in mechanism
//
   if (PrepFrm)
      {rc = PrepFrm->Add('+',pargs.path,  pargs.opaque,pargs.Ident,pargs.reqid,
                             pargs.notify,pargs.mode,atoi(pargs.prty));
       if (rc) Say.Emsg("Add", rc, "prepare", pargs.path);
          else {PTMutex.Lock();
                if (!PTable.Add(pargs.path, 0, 0, Hash_data_is_key)) NumFiles++;
                PTMutex.UnLock();
               }
       return rc == 0;
      }

// Restart the scheduler if need be
//
   PTMutex.Lock();
   if (!prepif || !prepSched.isAlive())
      {Say.Emsg("Add","No prepare manager; prepare",pargs.reqid,"ignored.");
       PTMutex.UnLock();
       return 0;
      }

// Write out the header line
//
   if (!prepMsg)
      {*pP++ = pargs.prty[0]; *pP = '\0';
       pdata[0] = (char *)"+ ";               pdlen[0] = 2;
       pdata[1] = pargs.reqid;                pdlen[1] = strlen(pargs.reqid);
       pdata[2] = (char *)" ";                pdlen[2] = 1;
       pdata[3] = pargs.notify;               pdlen[3] = strlen(pargs.notify);
       pdata[4] = (char *)" ";                pdlen[4] = 1;
       pdata[5] = prtybuff;                   pdlen[5] = strlen(prtybuff);
       pdata[6] = (char *)" ";                pdlen[6] = 1;
       pdata[7] = pargs.mode;                 pdlen[7] = strlen(pargs.mode);
       pdata[8] = (char *)" ";                pdlen[8] = 1;
       pdata[9] = pargs.path;                 pdlen[9] = strlen(pargs.path);
       pdata[10] = (char *)"\n";              pdlen[10] = 1;
       pdata[11]= 0;                          pdlen[11]= 0;
      if (!(rc = prepSched.Put((const char **)pdata, (const int *)pdlen)))
         if (!PTable.Add(pargs.path, 0, 0, Hash_data_is_key)) NumFiles++;
      } else {
       int Oflag = (index(pargs.mode, (int)'w') ? O_RDWR : 0);
       mode_t Prty = atoi(pargs.prty);
       XrdOucEnv Env(pargs.opaque);
       XrdOucMsubsInfo Info(pargs.Ident, &Env,  N2N,   pargs.path,
                            pargs.notify, Prty, Oflag, pargs.mode, pargs.reqid);
       int k = prepMsg->Subs(Info, pdata, pdlen);
       pdata[k]   = (char *)"\n"; pdlen[k++] = 1;
       pdata[k]   = 0;            pdlen[k]   = 0;
       if (!(rc = prepSched.Put((const char **)pdata, (const int *)pdlen)))
          if (!PTable.Add(pargs.path, 0, 0, Hash_data_is_key)) NumFiles++;
      }

// All done
//
   PTMutex.UnLock();
   return rc == 0;
}

/******************************************************************************/
/*                                   D e l                                    */
/******************************************************************************/
  
int XrdCmsPrepare::Del(char *reqid)
{
   char *pdata[4];
   int rc, pdlen[4];

// Use our built-in mechanism if so wanted
//
   if (PrepFrm)
      {if ((rc = PrepFrm->Del('-', reqid)))
          Say.Emsg("Del", rc, "unprepare", reqid);
       return rc == 0;
      }

// Restart the scheduler if need be
//
   PTMutex.Lock();
   if (!prepif || !prepSched.isAlive())
      {Say.Emsg("Del","No prepare manager; unprepare",reqid,"ignored.");
       PTMutex.UnLock();
       return 0;
      }

// Write out the delete request
//
   pdata[0] = (char *)"- ";
   pdlen[0] = 2;
   pdata[1] = reqid;
   pdlen[1] = strlen(reqid);
   pdata[2] = (char *)"\n";
   pdlen[2] = 1;
   pdata[3] = (char *)0;
   pdlen[3] = 0;
   rc = prepSched.Put((const char **)pdata, (const int *)pdlen);
   PTMutex.UnLock();
   return rc == 0;
}
 
/******************************************************************************/
/*                                  D o I t                                   */
/******************************************************************************/
  
void XrdCmsPrepare::DoIt()
{
// Simply scrub the cache
//
   Scrub();
   Sched->Schedule((XrdJob *)this,scrubtime+time(0));
}

/******************************************************************************/
/*                                E x i s t s                                 */
/******************************************************************************/
  
int  XrdCmsPrepare::Exists(char *path)
{
   int Found;

// Lock the hash table
//
   PTMutex.Lock();

// Look up the entry
//
   Found = (NumFiles ? PTable.Find(path) != 0 : 0);

// All done
//
   PTMutex.UnLock();
   return Found;
}
 
/******************************************************************************/
/*                                  G o n e                                   */
/******************************************************************************/
  
void XrdCmsPrepare::Gone(char *path)
{

// Lock the hash table
//
   PTMutex.Lock();

// Delete the entry
//
   if (NumFiles > 0 && PTable.Del(path) == 0) NumFiles--;

// All done
//
   PTMutex.UnLock();
}

/******************************************************************************/
/*                                I n f o r m                                 */
/******************************************************************************/
  
void XrdCmsPrepare::Inform(const char *cmd, XrdCmsPrepArgs *pargs)
{
   EPNAME("Inform")
   struct iovec Msg[8];
   char *mdest, *minfo;

// See if requestor wants a response
//
   if (!index(pargs->mode, (int)'n')
   ||  strncmp("udp://", pargs->notify, 6)
   ||  !Relay)
      {DEBUG(pargs->Ident <<' ' <<cmd <<' ' <<pargs->reqid <<" not sent to "
                          <<pargs->notify);
       return;
      }

// Extract out destination and argument
//
   mdest = pargs->notify+6;
   if ((minfo = index(mdest, (int)'/')))
      {*minfo = '\0'; minfo++;}
   if (!minfo || !*minfo) minfo = (char *)"*";
   DEBUG("Sending " <<mdest <<": " <<cmd <<' '<<pargs->reqid <<' ' <<minfo);

// Create message to be sent
//
   Msg[0].iov_base = (char *)cmd;  Msg[0].iov_len  = strlen(cmd);
   Msg[1].iov_base = (char *)" ";  Msg[1].iov_len  = 1;
   Msg[2].iov_base = pargs->reqid; Msg[2].iov_len  = strlen(pargs->reqid);
   Msg[3].iov_base = (char *)" ";  Msg[3].iov_len  = 1;
   Msg[4].iov_base = minfo;        Msg[4].iov_len  = strlen(minfo);
   Msg[5].iov_base = (char *)" ";  Msg[5].iov_len  = 1;
   Msg[6].iov_base = pargs->path;  Msg[6].iov_len  = (pargs->pathlen)-1;
   Msg[7].iov_base = (char *)"\n"; Msg[7].iov_len  = 1;

// Send the message and return
//
   Relay->Send(Msg, 8, mdest);
}

/******************************************************************************/
/*                               P r e p a r e                                */
/******************************************************************************/

void XrdCmsPrepare::Prepare(XrdCmsPrepArgs *pargs)
{
   EPNAME("Prepare");
   int rc;

// Check if this file is not online, prepare it
//
   if (!(rc = isOnline(pargs->path)))
      {DEBUG("Preparing " <<pargs->reqid <<' ' <<pargs->notify <<' ' 
                          <<pargs->prty <<' ' <<pargs->mode <<' ' <<pargs->path);
       if (!Config.DiskSS) Say.Emsg("Prepare","staging disallowed; ignoring prep",
                                    pargs->Ident, pargs->reqid);
          else Add(*pargs);
       return;
      }

// If the file is really online, inform the requestor
//
   if (rc > 0) Inform("avail", pargs);
}

/******************************************************************************/
/*                                 R e s e t                                  */
/******************************************************************************/

void XrdCmsPrepare::Reset(const char *iName, const char *aPath, int aMode)
{
   EPNAME("Reset");
   char baseAP[1024], *Slash;

// This is a call from the configurator. No need to do anything if we have
// no interface to initialize.
//
   if (!prepif) return;

// If this is a built-in mechanism, then allocate the prepare interface
// and initialize it. This is a one-time thing and it better work right away.
// In any case, do a standard reset.
//
   if (!*prepif)
      {PrepFrm = new XrdFrmProxy(Say.logger(), iName);
       DEBUG("Initializing internal FRM prepare interface.");
       strcpy(baseAP, aPath); baseAP[strlen(baseAP)-1] = '\0';
       if ((Slash = rindex(baseAP, '/'))) *Slash = '\0';
       if (!(prepOK = PrepFrm->Init(XrdFrmProxy::opStg, baseAP, aMode)))
          {Say.Emsg("Reset", "Built-in prepare init failed; prepare disabled.");
           return;
          }
      }

// Reset the interface and schedule a scrub
//
   Reset();
   if (scrubtime) Sched->Schedule((XrdJob *)this,scrubtime+time(0));

}

/******************************************************************************/
/*                              s e t P a r m s                               */
/******************************************************************************/
  
int XrdCmsPrepare::setParms(int rcnt, int stime, int deco)
{if (rcnt  > 0) resetcnt  = scrub2rst = rcnt;
 if (stime > 0) scrubtime = stime;
 doEcho = deco;
 return 0;
}

int XrdCmsPrepare::setParms(const char *ifpgm, char *ifmsg)
{if (ifpgm)
    {const char *Slash = rindex(ifpgm, '/');
     if (prepif) free(prepif);
     if (Slash && !strcmp(Slash+1, "frm_xfragent")) ifpgm = "";
     prepif = strdup(ifpgm);
    }
 if (ifmsg)
    {if (prepMsg) delete prepMsg;
     prepMsg = new XrdOucMsubs(&Say);
     if (!(prepMsg->Parse("prepmsg", ifmsg)))
        {delete prepMsg; prepMsg = 0; return 1;}
    }
 return 0;
}
 
/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                              i s O n l i n e                               */
/******************************************************************************/
  
int XrdCmsPrepare::isOnline(char *path)
{
   static const int Sopts = XRDOSS_resonly | XRDOSS_updtatm;
   struct stat buf;

// Issue the stat() via oss plugin. If it indicates the file is not there is
// still might be logically here because it's in a staging queue.
//
   if (Config.ossFS->Stat(path, &buf, Sopts))
      {if (Config.DiskSS && Exists(path)) return -1;
          else return 0;
      }
   return 1;
}

/******************************************************************************/
/*                                 R e s e t                                  */
/******************************************************************************/
  
void XrdCmsPrepare::Reset()  // Must be called with PTMutex locked!
{
   char *lp,  *pdata[] = {(char *)"?\n", 0};
   int         pdlen[] = {2, 0};

// Hanlde via built-in mechanism
//
   if (PrepFrm)
      {XrdFrmProxy::Queues State(XrdFrmProxy::opStg);
       char Buff[1024];
       if (prepOK)
          {PTable.Purge(); NumFiles = 0;
           while(PrepFrm->List(State, Buff, sizeof(Buff)))
                {PTable.Add(Buff, 0, 0, Hash_data_is_key); NumFiles++;
                 if (doEcho) Say.Emsg("Reset","Prepare pending for",Buff);
                }
          }
       return;
      }

// Check if we really have an interface to reset
//
   if (!prepif)
      {Say.Emsg("Reset", "Prepare program not specified; prepare disabled.");
       return;
      }

// Do it the slow external way
//
   if (!prepSched.isAlive() && !startIF()) return;
   if (prepSched.Put((const char **)pdata, (const int *)pdlen))
      {Say.Emsg("Prepare", prepSched.LastError(), "write to", prepif);
       prepSched.Drain(); prepOK = 0;
      }
      else {PTable.Purge(); NumFiles = 0;
            while((lp = prepSched.GetLine()) && *lp)
                 {PTable.Add(lp, 0, 0, Hash_data_is_key); NumFiles++;
                  if (doEcho) Say.Emsg("Reset","Prepare pending for",lp);
                 }
           }
}
  
/******************************************************************************/
/*                                 S c r u b                                  */
/******************************************************************************/
  
void XrdCmsPrepare::Scrub()
{
     PTMutex.Lock();
     if (scrub2rst <= 0)
        {Reset();
         scrub2rst = resetcnt;
        }
        else {PTable.Apply(XrdCmsScrubScan, (void *)0);
              scrub2rst--;
             }
     if (!PrepFrm && !prepSched.isAlive()) startIF();
     PTMutex.UnLock();
}

/******************************************************************************/
/*                               s t a r t I F                                */
/******************************************************************************/
  
int XrdCmsPrepare::startIF()  // Must be called with PTMutex locked!
{   
   EPNAME("startIF")

// If we are using a local interface then there is nothing to start.
//
   if (PrepFrm) return prepOK;

// Complain if there is no external prepare program
//
   if (!prepif)
      {Say.Emsg("startIF","Prepare program not specified; prepare disabled.");
       return (prepOK = 0);
      }

// Setup the external program
//
   DEBUG("Prepare: Starting " <<prepif);
   if (prepSched.Exec(prepif, 1))
      {time_t eNow = time(0);
       prepOK = 0;
       if ((eNow - lastemsg) >= 60)
          {lastemsg = eNow;
           Say.Emsg("Prepare", prepSched.LastError(), "start", prepif);
          }
      } else prepOK = 1;

// All done
//
   return prepOK;
}
