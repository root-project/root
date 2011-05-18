/******************************************************************************/
/*                                                                            */
/*                          X r d O f s E v s . c c                           */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/*             Based on code developed by Derek Feichtinger, CERN.            */
/******************************************************************************/
  
//         $Id$

const char *XrdOfsEvsCVSID = "$Id$";

#include <ctype.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "XrdOfs/XrdOfsEvs.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdOuc/XrdOucProg.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdNet/XrdNetOpts.hh"
#include "XrdNet/XrdNetSocket.hh"
#include "XrdSys/XrdSysPlatform.hh"

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/

class XrdOfsEvsMsg
{
public:

XrdOfsEvsMsg *next;
char         *text;
int           tlen;
int           isBig;

             XrdOfsEvsMsg(char *tval=0, int big=0)
                        {text = tval; tlen=0; isBig = big; next=0;}

            ~XrdOfsEvsMsg() {if (text) free(text);}
};

/******************************************************************************/
/*                     E x t e r n a l   L i n k a g e s                      */
/******************************************************************************/
  
void *XrdOfsEvsSend(void *pp)
{
     XrdOfsEvs *evs = (XrdOfsEvs *)pp;
     evs->sendEvents();
     return (void *)0;
}
  
/******************************************************************************/
/*                    S t a t i c   D e f i n i t i o n s                     */
/******************************************************************************/

XrdOfsEvsFormat XrdOfsEvs::MsgFmt[XrdOfsEvs::nCount];

const int       XrdOfsEvs::minMsgSize;
const int       XrdOfsEvs::maxMsgSize;

/******************************************************************************/
/*                     X r d E v s F o r m a t : : D e f                      */
/******************************************************************************/
  
void XrdOfsEvsFormat::Def(evFlags theFlags, const char *Fmt, ...)
{
   va_list ap;
   int theVal, i = 0;

// Return if already defined
//
   if (Format) return;

// Set flags and format. Prepare the arg vector
//
   Flags = theFlags; 
   Format = Fmt;
   memset(Args, 0, sizeof(Args));

// Pick up all arguments
//
   va_start(ap, Fmt);
   while((theVal = va_arg(ap, int)) >= 0) 
        Args[i++] = static_cast<XrdOfsEvsInfo::evArg>(theVal);
   va_end(ap);
}

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdOfsEvs::XrdOfsEvs(Event theEvents, const char *Target, int minq, int maxq)
{

// Set common variables
//
   enEvents = static_cast<Event>(theEvents & enMask);
   endIT = 0;
   theTarget = strdup(Target);
   eDest = 0; 
   theProg = 0;
   maxMin = minq; maxMax = maxq;
   msgFirst = msgLast = msgFreeMax = msgFreeMin = 0;
   numMax = numMin = 0; 
   tid = 0;
   msgFD = -1;

// Initialize all static format entries that have not been initialized yet.
// Note that format may be specified prior to this object being created!
//
// <tid> chmod  <mode> <path>
//
   MsgFmt[Chmod  & Mask].Def(XrdOfsEvsFormat::cvtMode, "%s chmod %s %s\n",
                             XrdOfsEvsInfo::evTID,
                             XrdOfsEvsInfo::evFMODE, XrdOfsEvsInfo::evLFN1, -1);
// <tid> closer <path>
//
   MsgFmt[Closer & Mask].Def(XrdOfsEvsFormat::Null,    "%s closer %s\n",
                             XrdOfsEvsInfo::evTID,   XrdOfsEvsInfo::evLFN1, -1);
                                              
// <tid> closew <path>
//
   MsgFmt[Closew & Mask].Def(XrdOfsEvsFormat::Null,    "%s closew %s\n",
                             XrdOfsEvsInfo::evTID,   XrdOfsEvsInfo::evLFN1, -1);
                                              
// <tid> create <mode> <path>
//
   MsgFmt[Create & Mask].Def(XrdOfsEvsFormat::cvtMode, "%s create %s %s\n",
                             XrdOfsEvsInfo::evTID,
                             XrdOfsEvsInfo::evFMODE, XrdOfsEvsInfo::evLFN1, -1);
// <tid> mkdir  <mode> <path>
//
   MsgFmt[Mkdir  & Mask].Def(XrdOfsEvsFormat::cvtMode, "%s mkdir %s %s\n",
                             XrdOfsEvsInfo::evTID,
                             XrdOfsEvsInfo::evFMODE, XrdOfsEvsInfo::evLFN1, -1);
// <tid> mv     <path> <path>
//
   MsgFmt[Mv     & Mask].Def(XrdOfsEvsFormat::Null,    "%s mv %s %s\n",
                             XrdOfsEvsInfo::evTID,
                             XrdOfsEvsInfo::evLFN1,  XrdOfsEvsInfo::evLFN2, -1);
// <tid> openr  <path>
//
   MsgFmt[Openr  & Mask].Def(XrdOfsEvsFormat::Null,    "%s openr %s\n",
                             XrdOfsEvsInfo::evTID,   XrdOfsEvsInfo::evLFN1, -1);
                                              
// <tid> openw  <path>
//
   MsgFmt[Openw  & Mask].Def(XrdOfsEvsFormat::Null,    "%s openw %s\n",
                             XrdOfsEvsInfo::evTID,   XrdOfsEvsInfo::evLFN1, -1);
                                              
// <tid> rm     <path>
//
   MsgFmt[Rm     & Mask].Def(XrdOfsEvsFormat::Null,    "%s rm %s\n",
                             XrdOfsEvsInfo::evTID,   XrdOfsEvsInfo::evLFN1, -1);
                                              
// <tid> rmdir  <path>
//
   MsgFmt[Rmdir  & Mask].Def(XrdOfsEvsFormat::Null,    "%s rmdir %s\n",
                             XrdOfsEvsInfo::evTID,   XrdOfsEvsInfo::evLFN1, -1);
                                              
// <tid> trunc  <size>
//
   MsgFmt[Trunc  & Mask].Def(XrdOfsEvsFormat::cvtFSize,"%s trunc %s\n",
                             XrdOfsEvsInfo::evTID,   XrdOfsEvsInfo::evFSIZE,-1);
                                              
// <tid> fwrite <path>
//
   MsgFmt[Fwrite & Mask].Def(XrdOfsEvsFormat::Null,    "%s fwrite %s\n",
                             XrdOfsEvsInfo::evTID,   XrdOfsEvsInfo::evLFN1, -1);
}

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/

XrdOfsEvs::~XrdOfsEvs()
{
  XrdOfsEvsMsg *tp;

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
  if (theTarget) free(theTarget);
  if (msgFD >= 0)close(msgFD);
  if (theProg)   delete theProg;
  qMut.UnLock();

// Release all free message blocks
//
  fMut.Lock();
  while ((tp = msgFreeMax)) {msgFreeMax = tp->next; delete tp;}
  while ((tp = msgFreeMin)) {msgFreeMin = tp->next; delete tp;}
  fMut.UnLock();
}

/******************************************************************************/
/*                                N o t i f y                                 */
/******************************************************************************/
  
void XrdOfsEvs::Notify(Event eID, XrdOfsEvsInfo &Info)
{
   static int warnings = 0;
   XrdOfsEvsFormat *fP;
   XrdOfsEvsMsg *tp;
   char modebuff[8], sizebuff[16];
   int eNum, isBig = (eID & Mv), msgSize = (isBig ? maxMsgSize : minMsgSize);

// Validate event number and set event name
//
   eNum = eID & Mask;
   if (eNum < 0 || eNum >= nCount) return;

// Check if we need to do any conversions
//
   fP = &MsgFmt[eNum];
   if (fP->Flags & XrdOfsEvsFormat::cvtMode)
      {sprintf(modebuff, "%o", static_cast<int>((Info.FMode() & S_IAMB)));
       Info.Set(XrdOfsEvsInfo::evFMODE, modebuff);
      } else Info.Set(XrdOfsEvsInfo::evFMODE, "$FMODE");
   if (fP->Flags & XrdOfsEvsFormat::cvtFSize)
      {sprintf(sizebuff, "%lld", Info.FSize());
       Info.Set(XrdOfsEvsInfo::evFSIZE, sizebuff);
      } else Info.Set(XrdOfsEvsInfo::evFSIZE, "$FSIZE");

// Get a message block
//
   if (!(tp = getMsg(isBig)))
      {if ((++warnings & 0xff) == 1)
          eDest->Emsg("Notify", "Ran out of message objects;", eName(eNum),
                                "event notification not sent.");
          return;
      }

// Format the message
//
   tp->tlen = fP->SNP(Info, tp->text, msgSize);

// Put the message on the queue and return
//
   tp->next = 0;
   qMut.Lock();
   if (msgLast) {msgLast->next = tp; msgLast = tp;}
      else msgFirst = msgLast = tp;
   qMut.UnLock();
   qSem.Post();
}

/******************************************************************************/
/*                                 P a r s e                                  */
/******************************************************************************/
  
int XrdOfsEvs::Parse(XrdSysError &Eroute, XrdOfsEvs::Event eNum, char *mText)
{
    static struct valVar {const char              *vname;
                          XrdOfsEvsInfo::evArg     vnum;
                          XrdOfsEvsFormat::evFlags vopt;}
        Vars[] = {
        {"TID",     XrdOfsEvsInfo::evTID,   XrdOfsEvsFormat::Null},
        {"LFN",     XrdOfsEvsInfo::evLFN1,  XrdOfsEvsFormat::Null},
        {"LFN1",    XrdOfsEvsInfo::evLFN1,  XrdOfsEvsFormat::Null},
        {"CGI",     XrdOfsEvsInfo::evCGI1,  XrdOfsEvsFormat::Null},
        {"CGI1",    XrdOfsEvsInfo::evCGI1,  XrdOfsEvsFormat::Null},
        {"LFN2",    XrdOfsEvsInfo::evLFN2,  XrdOfsEvsFormat::Null},
        {"CGI2",    XrdOfsEvsInfo::evCGI2,  XrdOfsEvsFormat::Null},
        {"FMODE",   XrdOfsEvsInfo::evFMODE, XrdOfsEvsFormat::cvtMode},
        {"FSIZE",   XrdOfsEvsInfo::evFSIZE, XrdOfsEvsFormat::cvtFSize}
       };
   int numvars = sizeof(Vars)/sizeof(struct valVar);
   char parms[1024], *pP = parms;
   char *pE = parms+sizeof(parms)-((XrdOfsEvsInfo::evARGS*2)-8);
   char varbuff[16], *bVar, *eVar;
   int  i, j, aNum = 0, Args[XrdOfsEvsInfo::evARGS] = {0};
   XrdOfsEvsFormat::evFlags ArgOpts = XrdOfsEvsFormat::freeFmt;

// Parse the text
//
   parms[0] = '\0';
   while(*mText && pP < pE)
        {if (*mText == '\\' && *(mText+1) == '$')
            {*pP++ = '$'; mText += 2; continue;}
            else if (*mText != '$') {*pP++ = *mText++; continue;}
         bVar = mText+1;
              if (*mText == '{') {eVar = index(mText, '}'); j = 1;}
         else if (*mText == '[') {eVar = index(mText, ']'); j = 1;}
         else {eVar = bVar; while(isalpha(*eVar)) eVar++;   j = 0;}
         i = eVar - bVar;
         if (i < 1 || i >= (int)sizeof(varbuff))
            {Eroute.Emsg("Parse","Invalid notifymsg variable starting at",mText);
             return 1;
            }
         strncpy(varbuff, bVar, i); varbuff[i] = '\0';
         for (i = 0; i < numvars; i++)
             if (!strcmp(varbuff, Vars[i].vname)) break;
         if (i >= numvars)
            {Eroute.Emsg("Parse", "Unknown notifymsg variable -",varbuff);
             return 1;
            }
         if (aNum >= XrdOfsEvsInfo::evARGS)
            {Eroute.Say("Parse", "Too many notifymsg variables"); return 1;}
         strcpy(pP, "%s"); pP += 2;
         Args[aNum++] = Vars[i].vnum; 
         ArgOpts = static_cast<XrdOfsEvsFormat::evFlags>(ArgOpts|Vars[i].vopt);
         mText = eVar+j;
        }

// Check if we overran the buffer or didn't have any text
//
   if (pP >= pE)
      {Eroute.Emsg("Parse","notifymsg text too long");return 1;}
   if (!parms[0])
      {Eroute.Emsg("Parse","notifymsg text not specified");return 1;}

// Set the format
//
   strcpy(pP, "\n");
   eNum = static_cast<Event>(eNum & Mask);
   MsgFmt[eNum].Set(ArgOpts, strdup(parms), Args);

// All done
//
   return 0;
}

/******************************************************************************/
/*                            s e n d E v e n t s                             */
/******************************************************************************/
  
void XrdOfsEvs::sendEvents(void)
{
   XrdOfsEvsMsg *tp;
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
            {if (!theProg) Feed(tp->text, tp->tlen);
                else {theData[0] = tp->text; theDlen[0] = tp->tlen;
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
  
int XrdOfsEvs::Start(XrdSysError *eobj)
{
   int rc;

// Set the error object pointer
//
   eDest = eobj;

// Check if we need to create a socket to a path
//
   if (*theTarget == '>')
      {XrdNetSocket *msgSock;
       if (!(msgSock = XrdNetSocket::Create(eobj,theTarget+1,0,0660,XRDNET_FIFO)))
          return -1;
       msgFD = msgSock->Detach();
       delete msgSock;

      } else {

      // Allocate a new program object if we don't have one
      //
         if (theProg) return 0;
         theProg = new XrdOucProg(eobj);

     // Setup the program
     //
        if (theProg->Setup(theTarget, eobj)) return -1;
        if ((rc = theProg->Start()))
           {eobj->Emsg("Evs", rc, "start event collector"); return -1;}
    }

// Now start a thread to get messages and send them to the collector
//
   if ((rc = XrdSysThread::Run(&tid, XrdOfsEvsSend, static_cast<void *>(this),
                          0, "Event notification sender")))
      {eobj->Emsg("Evs", rc, "create event notification thread");
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
/*                                 e N a m e                                  */
/******************************************************************************/
  
const char *XrdOfsEvs::eName(int eNum)
{
  static const char *eventName[] = {"Chmod",  "closer", "closew", "create",
                                    "fwrite", "mkdir",  "mv",     "openr",
                                    "opnw",   "rm",     "rmdir",  "trunc"};

  eNum = (eNum & Mask);
  return (eNum < 0 || eNum >= nCount ? "?" : eventName[eNum]);
}

/******************************************************************************/
/*                                  F e e d                                   */
/******************************************************************************/
  
int XrdOfsEvs::Feed(const char *data, int dlen)
{
   int retc;

// Write the data. ince this is a udp socket all the data goes or none does
//
  do { retc = write(msgFD, (const void *)data, (size_t)dlen);}
       while (retc < 0 && errno == EINTR);
  if (retc < 0)
     {eDest->Emsg("EvsFeed", errno, "write to event socket", theTarget);
      return -1;
     }

// All done
//
   return 0;
}

/******************************************************************************/
/*                                g e t M s g                                 */
/******************************************************************************/

XrdOfsEvsMsg *XrdOfsEvs::getMsg(int bigmsg)
{
   XrdOfsEvsMsg *tp;
   int msz = 0;

// Lock the free queue
//
   fMut.Lock();

// Get a free element from the big or small queue, as needed
//
   if (bigmsg)
        if ((tp = msgFreeMax)) msgFreeMax = tp->next;
           else msz = maxMsgSize;
   else if ((tp = msgFreeMin)) msgFreeMin = tp->next;
           else msz = minMsgSize;

// Check if we have to allocate a new item
//
   if (!tp && (numMax + numMin) < (maxMax + maxMin))
      {if ((tp = new XrdOfsEvsMsg((char *)malloc(msz), bigmsg)))
          {if (!(tp->text)) {delete tp; tp = 0;}
              else if (bigmsg) numMax++;
                      else     numMin++;
          }
      }

// Unlock and return result
//
   fMut.UnLock();
   return tp;
}

/******************************************************************************/
/*                                r e t M s g                                 */
/******************************************************************************/

void XrdOfsEvs::retMsg(XrdOfsEvsMsg *tp)
{

// Lock the free queue
//
   fMut.Lock();

// Check if we exceeded the hold quotax
//
   if (tp->isBig)
      if (numMax > maxMax) {delete tp; numMax--;}
         else {tp->next = msgFreeMax; msgFreeMax = tp;}
      else
      if (numMin > maxMin) {delete tp; numMin--;}
         else {tp->next = msgFreeMin; msgFreeMin = tp;}

// Unlock and return
//
   fMut.UnLock();
}
