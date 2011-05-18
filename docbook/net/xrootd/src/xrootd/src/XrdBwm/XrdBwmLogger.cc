/******************************************************************************/
/*                                                                            */
/*                       X r d B w m L o g g e r . c c                        */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

const char *XrdBwmLoggerCVSID = "$Id$";

#include <ctype.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "XrdBwm/XrdBwmLogger.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdOuc/XrdOucProg.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdNet/XrdNetOpts.hh"
#include "XrdNet/XrdNetSocket.hh"
#include "XrdSys/XrdSysPlatform.hh"

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/

class XrdBwmLoggerMsg
{
public:

static const int msgSize = 2048;

XrdBwmLoggerMsg *next;
char             Text[msgSize];
int              Tlen;

             XrdBwmLoggerMsg() : next(0), Tlen(0) {}

            ~XrdBwmLoggerMsg() {}
};

/******************************************************************************/
/*                     E x t e r n a l   L i n k a g e s                      */
/******************************************************************************/
  
void *XrdBwmLoggerSend(void *pp)
{
     XrdBwmLogger *lP = (XrdBwmLogger *)pp;
     lP->sendEvents();
     return (void *)0;
}

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdBwmLogger::XrdBwmLogger(const char *Target)
{

// Set common variables
//
   theTarget = strdup(Target);
   eDest = 0; 
   theProg = 0;
   msgFirst = msgLast = msgFree = 0;
   tid = 0;
   msgFD = 0;
   endIT = 0;
   theEOL= '\n';
}

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/

XrdBwmLogger::~XrdBwmLogger()
{
  XrdBwmLoggerMsg *tp;

// Kill the notification thread. This may cause a msg block to be orphaned
// but, in practice, this object does not really get deleted after being 
// started. So, the problem is moot.
//
   endIT = 1;
   if (tid) XrdSysThread::Kill(tid);

// Release all queued message bocks
//
  qMut.Lock();
  while ((tp = msgFirst)) {msgFirst = tp->next; delete tp;}
  if (theTarget)  free(theTarget);
  if (msgFD >= 0) close(msgFD);
  if (theProg)    delete theProg;
  qMut.UnLock();

// Release all free message blocks
//
  fMut.Lock();
  while ((tp = msgFree)) {msgFree = tp->next; delete tp;}
  fMut.UnLock();
}

/******************************************************************************/
/*                                 E v e n t                                  */
/******************************************************************************/
  
void XrdBwmLogger::Event(Info &eInfo)
{
   static int warnings = 0;
   XrdBwmLoggerMsg *tp;

// Get a message block
//
   if (!(tp = getMsg()))
      {if ((++warnings & 0xff) == 1)
          eDest->Emsg("Notify", "Ran out of logger message objects;",
                                eInfo.Tident, "event not logged.");
          return;
      }

// Format the message
//
   tp->Tlen = snprintf(tp->Text, XrdBwmLoggerMsg::msgSize,
                    "<stats id=\"bwm\"><tid>%s</tid><lfn>%s</lfn>"
                    "<lcl>%s</lcl><rmt>%s</rmt><flow>%c</flow>"
                    "<at>%ld</at><bt>%ld</bt><ct>%ld</ct>"
                    "<iq>%d</iq><oq>%d</oq><xq>%d</xq>"
                    "<sz>%lld<sz><esec>%d</esec></stats>%c",
                    eInfo.Tident, eInfo.Lfn, eInfo.lclNode, eInfo.rmtNode,
                    eInfo.Flow, eInfo.ATime, eInfo.BTime, eInfo.CTime,
                    eInfo.numqIn, eInfo.numqOut, eInfo.numqXeq, eInfo.Size,
                    eInfo.ESec, theEOL);

// Either log this or put the message on the queue and return
//
   tp->next = 0;
   qMut.Lock();
   if (msgLast) {msgLast->next = tp; msgLast = tp;}
      else msgFirst = msgLast = tp;
   qMut.UnLock();
   qSem.Post();
}

/******************************************************************************/
/*                            s e n d E v e n t s                             */
/******************************************************************************/
  
void XrdBwmLogger::sendEvents(void)
{
   XrdBwmLoggerMsg *tp;
   const char *theData[2] = {0,0};
         int   theDlen[2] = {0,0};

// This is an endless loop that just gets things off the event queue and
// send them out. This allows us to only hang a simgle thread should the
// receiver get blocked, instead of the whole process.
//
   while(1)
        {qSem.Wait();
         qMut.Lock();
         if (endIT) break;
         if ((tp = msgFirst) && !(msgFirst = tp->next)) msgLast = 0;
         qMut.UnLock();
         if (tp) 
            {if (!theProg) Feed(tp->Text, tp->Tlen);
                else {theData[0] = tp->Text; theDlen[0] = tp->Tlen;
                      theProg->Feed(theData, theDlen);

                     }
             retMsg(tp);
            }
         }
   qMut.UnLock();
}

/******************************************************************************/
/*                                 S t a r t                                  */
/******************************************************************************/
  
int XrdBwmLogger::Start(XrdSysError *eobj)
{
   int rc;

// Set the error object pointer
//
   eDest = eobj;

// Check if we need to create a socket to a path
//
        if (!strcmp("*", theTarget)) {msgFD = -1; theEOL = '\0';}
   else if (*theTarget == '>')
           {XrdNetSocket *msgSock;
            if (!(msgSock = XrdNetSocket::Create(eobj, theTarget+1, 0, 0660,
                                                 XRDNET_FIFO))) return -1;
            msgFD = msgSock->Detach();
            delete msgSock;
           }
   else    {// Allocate a new program object if we don't have one
            //
            if (theProg) return 0;
            theProg = new XrdOucProg(eobj);

            // Setup the program
            //
            if (theProg->Setup(theTarget, eobj)) return -1;
            if ((rc = theProg->Start()))
               {eobj->Emsg("Logger", rc, "start event collector"); return -1;}
           }

// Now start a thread to get messages and send them to the collector
//
   if ((rc = XrdSysThread::Run(&tid, XrdBwmLoggerSend, static_cast<void *>(this),
                          0, "Log message sender")))
      {eobj->Emsg("Logger", rc, "create log message sender thread");
       return -1;
      }

// All done
//
   return 0;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                                  F e e d                                   */
/******************************************************************************/
  
int XrdBwmLogger::Feed(const char *data, int dlen)
{
   int retc;

// Send message to the log if need be
//
   if (msgFD < 0) {eDest->Say("", data); return 0;}

// Write the data. since this is a udp socket all the data goes or none does
//
  do { retc = write(msgFD, (const void *)data, (size_t)dlen);}
       while (retc < 0 && errno == EINTR);
  if (retc < 0)
     {eDest->Emsg("Feed", errno, "write to logger socket", theTarget);
      return -1;
     }

// All done
//
   return 0;
}

/******************************************************************************/
/*                                g e t M s g                                 */
/******************************************************************************/

XrdBwmLoggerMsg *XrdBwmLogger::getMsg()
{
   XrdBwmLoggerMsg *tp;

// Lock the free queue
//
   fMut.Lock();

// Get message object but don't give out too many
//
   if (msgsInQ >= maxmInQ) tp = 0;
      else {if ((tp = msgFree)) msgFree = tp->next;
               else tp = new XrdBwmLoggerMsg();
            msgsInQ++;
           }

// Unlock and return result
//
   fMut.UnLock();
   return tp;
}

/******************************************************************************/
/*                                r e t M s g                                 */
/******************************************************************************/

void XrdBwmLogger::retMsg(XrdBwmLoggerMsg *tp)
{

// Lock the free queue, return message, unlock the queue
//
   fMut.Lock();
   tp->next = msgFree; 
   msgFree  = tp;
   msgsInQ--;
   fMut.UnLock();
}
