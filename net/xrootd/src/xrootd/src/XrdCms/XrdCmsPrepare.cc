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
#include <utime.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdCms/XrdCmsConfig.hh"
#include "XrdCms/XrdCmsPrepare.hh"
#include "XrdCms/XrdCmsTrace.hh"
#include "XrdNet/XrdNetMsg.hh"
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

int XrdCmsScrubScan(const char *key, char *cip, void *xargp)
{
   struct stat buf;
   if (stat(key, &buf)) return 0;
   return -1;
}

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdCmsPrepare::XrdCmsPrepare() : XrdJob("File cache scrubber"),
                                 prepSched(&Say)
{prepif  = 0;
 preppid = 0;
 resetcnt = scrub2rst = 3;
 scrubtime= 20*60;
 NumFiles = 0;
 lastemsg = time(0);
 Relay    = new XrdNetMsg(&Say);
 isFrm = 0;
}

/******************************************************************************/
/*                                   A d d                                    */
/******************************************************************************/
  
int XrdCmsPrepare::Add(XrdCmsPrepArgs &pargs)
{
   char ubuff[256], *pdata[XrdOucMsubs::maxElem+2], prtybuff[8], *pP=prtybuff;
   int rc, pdlen[XrdOucMsubs::maxElem + 2];

// Restart the scheduler if need be
//
   PTMutex.Lock();
   if (!prepif || !prepSched.isAlive())
      {Say.Emsg("Add","No prepare manager; prepare",pargs.reqid,"ignored.");
       PTMutex.UnLock();
       return 0;
      }

// Extract out prty
//
   *pP++ = pargs.prty[0]; *pP = '\0';

// Write out the header line
//
   if (!prepMsg)
      {if (isFrm)
      {pdlen[0] = getID(pargs.Ident,ubuff,sizeof(ubuff)); pdata[0] = ubuff;}
else  {pdata[0] = (char *)"+ ";               pdlen[0] = 2;}
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
   rc = prepSched.Put((const char **)pdata, (const int *)pdlen) == 0;
   PTMutex.UnLock();
   return rc;
}
 
/******************************************************************************/
/*                                  D o I t                                   */
/******************************************************************************/
  
void XrdCmsPrepare::DoIt()
{
// Simply scrub the cache
//
   Scrub();
   if (prepif) Sched->Schedule((XrdJob *)this,scrubtime+time(0));
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
/*                              s e t P a r m s                               */
/******************************************************************************/
  
int XrdCmsPrepare::setParms(int rcnt, int stime, int deco)
{if (rcnt  > 0) resetcnt  = scrub2rst = rcnt;
 if (stime > 0) scrubtime = stime;
 doEcho = deco;
 return 0;
}

int XrdCmsPrepare::setParms(char *ifpgm, char *ifmsg)
{if (ifpgm)
    {if (prepif) free(prepif);
     prepif = strdup(ifpgm);
     isFrm = !strcmp(ifpgm, "frm_pstga");
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
/*                                 g e t I D                                  */
/******************************************************************************/
  
int XrdCmsPrepare::getID(const char *Tid, char *buff, int bsz)
{
   char *bP;
   int n;

// The buffer always starts with a '+'
//
   *buff = '+'; bP = buff+1; bsz -= 3;

// Get the trace id
//
   if (Tid && (n = strlen(Tid)) <= bsz) {strcpy(bP, Tid); bP += n;}

// Insert space
//
   *bP++ = ' '; *bP = '\0';
   return bP - buff;
}

/******************************************************************************/
/*                              i s O n l i n e                               */
/******************************************************************************/
  
int XrdCmsPrepare::isOnline(char *path)
{
   struct stat buf;
   struct utimbuf times;
   char *lclpath, lclbuff[XrdCmsMAX_PATH_LEN+1];

// Generate the true local path
//
   lclpath = path;
   if (Config.lcl_N2N)
      {if (Config.lcl_N2N->lfn2pfn(lclpath,lclbuff,sizeof(lclbuff))) return 0;
          else lclpath = lclbuff;
      }

// Do a stat
//
   if (stat(lclpath, &buf))
      {if (Config.DiskSS && Exists(path)) return -1;
          else return 0;
      }

// Make sure we are doing a stat on a file
//
   if ((buf.st_mode & S_IFMT) == S_IFREG)
      {times.actime = time(0);
       times.modtime = buf.st_mtime;
       utime(lclpath, &times);
       return 1;
      }

// Determine what to return
//
   return 0;
}

/******************************************************************************/
/*                                 R e s e t                                  */
/******************************************************************************/
  
int XrdCmsPrepare::Reset()  // Must be called with PTMutex locked!
{
     char *lp,  *pdata[] = {(char *)"?\n", 0};
     int ok = 0, pdlen[] = {2, 0};

     if (!prepif)
        Say.Emsg("Reset", "Prepare program not specified; prepare disabled.");
        else {scrub2rst = resetcnt;
              if (!prepSched.isAlive() && !startIF()) return 0;
              if (prepSched.Put((const char **)pdata, (const int *)pdlen))
                 {Say.Emsg("Prepare", prepSched.LastError(),
                                 "write to", prepif);
                  prepSched.Drain();
                 }
                 else {PTable.Purge(); ok = 1; NumFiles = 0;
                       while((lp = prepSched.GetLine()) && *lp)
                            {PTable.Add(lp, 0, 0, Hash_data_is_key);
                             NumFiles++;
                             if (doEcho) 
                                Say.Emsg("Reset","Prepare pending for",lp);
                            }
                      }
             }
    return ok;
}

/******************************************************************************/
/*                                 S c r u b                                  */
/******************************************************************************/
  
void XrdCmsPrepare::Scrub()
{
     PTMutex.Lock();
     if (scrub2rst <= 0) Reset();
        else {PTable.Apply(XrdCmsScrubScan, (void *)0);
              scrub2rst--;
             }
     if (!prepSched.isAlive()) startIF();
     PTMutex.UnLock();
}

/******************************************************************************/
/*                               s t a r t I F                                */
/******************************************************************************/
  
int XrdCmsPrepare::startIF()  // Must be called with PTMutex locked!
{   
    EPNAME("startIF")
    int NoGo = 0;

    if (!prepif)
       {Say.Emsg("startIF","Prepare program not specified; prepare disabled.");
        NoGo = 1;
       }
       else {DEBUG("Prepare: Starting " <<prepif);
             if ((NoGo = prepSched.Exec(prepif, 1)))
                {time_t eNow = time(0);
                 if ((eNow - lastemsg) >= 60)
                    {lastemsg = eNow;
                     Say.Emsg("Prepare", prepSched.LastError(),
                                    "start", prepif);
                    }
                }
            }
    return !NoGo;
}
