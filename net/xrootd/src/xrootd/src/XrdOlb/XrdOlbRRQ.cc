/******************************************************************************/
/*                                                                            */
/*                          X r d O l b R R Q . c c                           */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdOlbRRQCVSID = "$Id$";

#include "XrdOlb/XrdOlbManager.hh"
#include "XrdOlb/XrdOlbRRQ.hh"
#include "XrdOlb/XrdOlbRTable.hh"
#include "XrdOlb/XrdOlbServer.hh"
#include "XrdOlb/XrdOlbTrace.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSys/XrdSysTimer.hh"
#include <stdio.h>

using namespace XrdOlb;

// Note: Debugging statements have been commented out. This is time critical
//       code and debugging may only be enabled in standalone testing as the
//       delays introduced by DEBUG() will usually cause timeout failures.
  
/******************************************************************************/
/*       G l o b a l   O b j e c t s   &   S t a t i c   M e m b e r s        */
/******************************************************************************/
  
XrdOlbRRQ             XrdOlb::RRQ;

XrdSysMutex           XrdOlbRRQSlot::myMutex;
XrdOlbRRQSlot        *XrdOlbRRQSlot::freeSlot = 0;
short                 XrdOlbRRQSlot::initSlot = 0;

/******************************************************************************/
/*                    E x t e r n a l   F u n c t i o n s                     */
/******************************************************************************/
  
void *XrdOlbRRQ_StartTimeOut(void *parg) {return RRQ.TimeOut();}

void *XrdOlbRRQ_StartRespond(void *parg) {return RRQ.Respond();}

/******************************************************************************/
/*               X r d O l b R R Q   C l a s s   M e t h o d s                */
/******************************************************************************/
/******************************************************************************/
/*                                   A d d                                    */
/******************************************************************************/
  
short XrdOlbRRQ::Add(short Snum, XrdOlbRRQInfo *Info)
{
// EPNAME("RRQ Add");
   XrdOlbRRQSlot *sp;

// Obtain a slot and fill it in
//
   if (!(sp = XrdOlbRRQSlot::Alloc(Info))) return 0;
// DEBUG("adding slot " <<sp->slotNum);

// If a slot number given, check if it's the right slot and it is still queued.
// If so, piggy-back this request to existing one and make a fast exit
//
   myMutex.Lock();
   if (Snum && Slot[Snum].Info.Key == Info->Key && Slot[Snum].Expire)
      {sp->Cont = Slot[Snum].Cont; 
       Slot[Snum].Cont = sp;
       myMutex.UnLock();
       return Snum;
      }

// Queue this slot to the pending response queue and tell the timeout scheduler
//
   sp->Expire = myClock+1;
   if (waitQ.Singleton()) isWaiting.Post();
   waitQ.Prev()->Insert(&sp->Link);
   myMutex.UnLock();
   return sp->slotNum;
}

/******************************************************************************/
/*                                   D e l                                    */
/******************************************************************************/
  
void XrdOlbRRQ::Del(short Snum, const void *Key)
{
     Ready(Snum, Key, 0);
}

/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/
  
int XrdOlbRRQ::Init(int Tint, int Tdly)
{
   int rc;
   pthread_t tid;

// Set values
//
   if (Tint) Tslice = Tint;
   if (Tdly) Tdelay = Tdly;

// Start the responder thread
//
   if ((rc = XrdSysThread::Run(&tid, XrdOlbRRQ_StartRespond, (void *)0,
                               0, "Request Responder")))
      {Say.Emsg("Config", rc, "create request responder thread");
       return 1;
      }

// Start the timeout thread
//
   if ((rc = XrdSysThread::Run(&tid, XrdOlbRRQ_StartTimeOut, (void *)0,
                               0, "Request Timeout")))
      {Say.Emsg("Config", rc, "create request timeout thread");
       return 1;
      }

// Fill out the try and wait i/o vectors
//
   redr_iov[1].iov_base = (char *)" !try ";
   redr_iov[1].iov_len  = 6;
   redr_iov[2].iov_base = hostbuff;
   wait_iov[1].iov_base = waitbuff;
   wait_iov[1].iov_len  = sprintf(waitbuff," !wait %d\n", Tdelay);

// All done
//
   return 0;
}

/******************************************************************************/
/*                                 R e a d y                                  */
/******************************************************************************/
  
void XrdOlbRRQ::Ready(int Snum, const void *Key, SMask_t mask)
{
// EPNAME("RRQ Ready");
   XrdOlbRRQSlot *sp;

// Check if it's the right slot and it is still queued.
//
   myMutex.Lock();
   sp = &Slot[Snum];
   if (sp->Info.Key != Key || !sp->Expire)
      {myMutex.UnLock();
//     DEBUG("slot " <<Snum <<" no longer valid");
       return;
      }

// Move the element from the waiting queue to the ready queue
//
   sp->Link.Remove();
   if (readyQ.Singleton()) isReady.Post();
   sp->Arg = mask;
   readyQ.Prev()->Insert(&sp->Link);
   myMutex.UnLock();
// DEBUG("readied slot " <<Snum <<" mask " <<mask);
}

/******************************************************************************/
/*                               R e s p o n d                                */
/******************************************************************************/
  
void *XrdOlbRRQ::Respond()
{
// EPNAME("RRQ Respond");
   XrdOlbRRQSlot *sp, *cp;
   int i, doredir;

// In an endless loop, process all ready elements
//
   while(1)
        {isReady.Wait();
//       DEBUG("responder awoken");
         while(1)
              {myMutex.Lock();
               if (readyQ.Singleton()) {myMutex.UnLock(); break;}
               sp = readyQ.Next()->Item();
               sp->Link.Remove();
               sp->Expire = 0;
               myMutex.UnLock();
               cp = sp;
               while(cp && cp->Info.isLU) cp = sendLocInfo(cp);
               if (cp)
                  {if ((doredir = (cp->Arg &&
                       Manager.SelServer(cp->Info.isRW,cp->Arg,hostbuff))))
                      {i = strlen(hostbuff);
                       hostbuff[i] = '\n';
                       redr_iov[2].iov_len = i+1;
                      }
                   sendResponse(&cp->Info, doredir);
                   do {if (cp->Info.isLU) cp = sendLocInfo(cp);
                           else {sendResponse(&cp->Info, doredir);
                                 cp = cp->Cont;
                                }
                      } while(cp);
                  }
               sp->Recycle();
              }
        }

// Keep the compiler happy
//
   return (void *)0;
}

/******************************************************************************/
/*                           s e n d L o c I n f o                            */
/******************************************************************************/
  
XrdOlbRRQSlot *XrdOlbRRQ::sendLocInfo(XrdOlbRRQSlot *Sp)
{
   XrdOlbServer *sP;
   SMask_t myArg = Sp->Arg;

// Handle delays or responses
//
   if (myArg) do {RTable.Lock();
                  if ((sP = RTable.Find(Sp->Info.Rnum, Sp->Info.Rinst)))
                     sP->do_Locate(Sp->Info.ID,Sp->Info.ID,myArg,Sp->Info.Arg);
                  RTable.UnLock();
                 } while((Sp = Sp->Cont) && Sp->Info.isLU);
      else    do {sendResponse(&(Sp->Info), 0);}
                   while((Sp = Sp->Cont) && Sp->Info.isLU);

// Return propogated slot
//
   if (Sp) Sp->Arg = myArg;
   return Sp;
}
  
/******************************************************************************/
/*                          s e n d R e s p o n s e                           */
/******************************************************************************/
  
void XrdOlbRRQ::sendResponse(XrdOlbRRQInfo *Info, int doredir)
{
// EPNAME("sendResponse");
   XrdOlbServer *sp;

// Find the redirector and send the message
//
   RTable.Lock();
   if ((sp = RTable.Find(Info->Rnum, Info->Rinst)))
      {if (doredir){redr_iov[0].iov_base = Info->ID;
                    redr_iov[0].iov_len  = strlen(Info->ID);
                    sp->Send(redr_iov, redr_iov_cnt);
//                  DEBUG("Fast redirect " <<sp->Name() <<" -> " <<hostbuff);
                   }
              else {wait_iov[0].iov_base = Info->ID;
                    wait_iov[0].iov_len  = strlen(Info->ID);
                    sp->Send(wait_iov, wait_iov_cnt);
//                  DEBUG("Redirect delay " <<sp->Name() <<' ' <<Tdelay);
                   }
      } 
//    else {DEBUG("redirector " <<Info->Rnum <<'.' <<Info->Rinst <<"not found");}
   RTable.UnLock();
}

/******************************************************************************/
/*                               T i m e O u t                                */
/******************************************************************************/
  
void *XrdOlbRRQ::TimeOut()
{
// EPNAME("RRQ TimeOut");
   XrdOlbRRQSlot *sp;

// We measure 133ms (default) intervals to timeout waiting requests
//
   while(1)
        {isWaiting.Wait();
         myMutex.Lock();
         while(1)
              {myClock++;
               myMutex.UnLock();
               XrdSysTimer::Wait(Tslice);
               myMutex.Lock();
               while((sp=waitQ.Next()->Item()) && sp->Expire < myClock)
                    {sp->Link.Remove();
                     if (readyQ.Singleton()) isReady.Post();
                     sp->Arg = 0;
//                   DEBUG("expired slot " <<sp->slotNum);
                     readyQ.Prev()->Insert(&sp->Link);
                    }
               if (waitQ.Singleton()) break;
              }
         myMutex.UnLock();
        }

// Keep the compiler happy
//
   return (void *)0;
}

/******************************************************************************/
/*           X r d O l b R R Q S l o t   C l a s s   M e t h o d s            */
/******************************************************************************/
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdOlbRRQSlot::XrdOlbRRQSlot() : Link(this)
{

slotNum  = initSlot++;
if (slotNum)
   {Cont     = freeSlot;
    freeSlot = this;
   } else Cont = 0;
Arg      = 0;
Info.Key = 0;
}

/******************************************************************************/
/*                                 A l l o c                                  */
/******************************************************************************/
  
XrdOlbRRQSlot *XrdOlbRRQSlot::Alloc(XrdOlbRRQInfo *theInfo)
{
   XrdOlbRRQSlot *sp;

   myMutex.Lock();
   if ((sp = freeSlot))
      {sp->Info = *theInfo;
       freeSlot = sp->Cont;
       sp->Cont = 0;
       sp->Arg  = 0;
      }
   myMutex.UnLock();
   return sp;
}

/******************************************************************************/
/*                               R e c y c l e                                */
/******************************************************************************/
  
void XrdOlbRRQSlot::Recycle()
{
   XrdOlbRRQSlot *sp, *np = Cont;

   myMutex.Lock();
   if (!Link.Singleton()) Link.Remove();
   while((sp = np))
        {np           = sp->Cont;
         sp->Cont     = freeSlot;
         freeSlot     = sp;
         sp->Info.Key = 0;
        }
   Info.Key = 0;
   Cont     = freeSlot;
   freeSlot = this;
   myMutex.UnLock();
}
