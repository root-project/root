/******************************************************************************/
/*                                                                            */
/*                         X r d F r m P s t g . c c                          */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

const char *XrdFrmPstgCVSID = "$Id$";

#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmPstg.hh"
#include "XrdFrm/XrdFrmPstgReq.hh"
#include "XrdFrm/XrdFrmPstgXfr.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdNet/XrdNetMsg.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysPthread.hh"

using namespace XrdFrm;
  
/******************************************************************************/
/* Public:                         A g e n t                                  */
/******************************************************************************/
  
int XrdFrmPstg::Agent(char *c2sFN)
{
   EPNAME("Agent");
   XrdNetMsg    udpMsg(&Say, c2sFN);
   XrdOucStream Request;
   struct stat buf;
   char *tp, theMsg[] = {'!','a','t','t','n','\n'};
   int  lenMsg = sizeof(theMsg);

// Attach stdin to the Request stream
//
   Request.Attach(STDIN_FILENO, 8*1024);

// Each frm request comes in as:
//
// +[<traceid>] <npath> <prty> <mode> <path> [. . .]
// - <requestid>
// ?
//
   while((tp = Request.GetLine()))
        {DEBUG("Request: '" <<tp <<"'");
         if ((tp = Request.GetToken()))
            {     if (*tp == '+'){Agent_Add(Request, tp+1);
                                  if (!stat(c2sFN, &buf))
                                     udpMsg.Send(theMsg, lenMsg);
                                 }
             else if (*tp == '-') Agent_Del(Request, tp+1);
             else if (*tp == '?') Agent_Lst(Request, tp+1);
             else if (*tp == '!') udpMsg.Send(theMsg, lenMsg);
             else Say.Emsg("Agent", "Invalid request, '", tp, "'.");
            }
        }

// If we exit then we lost the connection
//
   Say.Emsg("Agent", "Exiting; lost request connection!");
   return 8;
}

/******************************************************************************/
/* Private:                    A g e n t _ A d d                              */
/******************************************************************************/
  
void XrdFrmPstg::Agent_Add(XrdOucStream &Request, char *Tok)
{
   XrdFrmPstgReq::Request myReq;
   const char *Miss = 0;
   char *tp, *op;

// Handle: +[<traceid>] <requestid> <npath> <prty> <mode> <path> [. . .]
//
   memset(&myReq, 0, sizeof(myReq));

       if (*Tok) strlcpy(myReq.User, Tok, sizeof(myReq.User));
          else   strlcpy(myReq.User, Config.myProg, sizeof(myReq.User));

       if (!(tp = Request.GetToken())) Miss = "request id";
          else strlcpy(myReq.ID, tp, sizeof(myReq.ID));

   if (!Miss)
      {if (!(tp = Request.GetToken())) Miss = "notify path";
          else strlcpy(myReq.Notify, tp, sizeof(myReq.Notify));
      }

   if (!Miss)
      {if (!(tp = Request.GetToken())) Miss = "priority";
          else {myReq.Prty = atoi(tp);
                if (myReq.Prty < 0) myReq.Prty = 0;
                   else if (myReq.Prty > XrdFrmPstgReq::maxPrty)
                            myReq.Prty = XrdFrmPstgReq::maxPrty;
               }
      }

   if (!Miss)
      {if (!(tp = Request.GetToken())) Miss = "mode";
          else {if (index(tp,'w')) myReq.Options |= XrdFrmPstgReq::stgRW;
                if (*myReq.Notify != '-')
                   {if (index(tp,'s') ||  index(tp,'n'))
                       myReq.Options |= XrdFrmPstgReq::msgSucc;
                    if (index(tp,'f') || !index(tp,'q'))
                       myReq.Options |= XrdFrmPstgReq::msgFail;
                   }
               }
      }

   if (!Miss && !(tp = Request.GetToken())) Miss = "path";

// Check for any errors
//
   if (Miss) {Say.Emsg("Agent_Add", Miss, "missing in '+' request.");
              return;
             }

// Add all paths in the request
//
   myReq.addTOD = time(0);
   do {strlcpy(myReq.LFN, tp, sizeof(myReq.LFN));
       if ((op = index(tp, '?'))) myReq.Opaque = op-tp;
          else myReq.Opaque = 0;
       rQueue[myReq.Prty]->Add(&myReq);
       if ((tp = Request.GetToken())) memset(myReq.LFN, 0, sizeof(myReq.LFN));
      } while(tp);
}

/******************************************************************************/
/* Private:                    A g e n t _ D e l                              */
/******************************************************************************/
  
void XrdFrmPstg::Agent_Del(XrdOucStream &Request, char *Tok)
{
   XrdFrmPstgReq::Request myReq;
   char *tp;
   int i;

// Handle: - <requestid>
//
   memset(&myReq, 0, sizeof(myReq));

   if ((tp = Request.GetToken())) strlcpy(myReq.ID, tp, sizeof(myReq.ID));
      else {Say.Emsg("Agent_Del", "request id missing in '-' request.");
            return;
           }
  
// Remove all pending requests for this id
//
   for (i = 0; i <= XrdFrmPstgReq::maxPrty; i++) rQueue[i]->Can(&myReq);
}

/******************************************************************************/
/* Private:                    A g e n t _ L s t                              */
/******************************************************************************/
  
void XrdFrmPstg::Agent_Lst(XrdOucStream &Request, char *Tok)
{
   static const int maxItems = 8;
   static struct ITypes {const char *IName; XrdFrmPstgReq::Item ICode;}
                 ITList[] = {{"lfn",    XrdFrmPstgReq::getLFN},
                             {"lfncgi", XrdFrmPstgReq::getLFNCGI},
                             {"mode",   XrdFrmPstgReq::getMODE},
                             {"prty",   XrdFrmPstgReq::getPRTY},
                             {"qwt",    XrdFrmPstgReq::getQWT},
                             {"rid",    XrdFrmPstgReq::getRID},
                             {"tod",    XrdFrmPstgReq::getTOD},
                             {"note",   XrdFrmPstgReq::getNOTE},
                             {"tid",    XrdFrmPstgReq::getUSER}};
   static int ITNum = sizeof(ITList)/sizeof(struct ITypes);

   char myLfn[4096];
   XrdFrmPstgReq::Item Items[maxItems];
   int n = 0, i, Offs;
   char *tp;

   while((tp = Request.GetToken()) && n < maxItems)
        {for (i = 0; i < ITNum; i++)
             if (!strcmp(tp, ITList[i].IName))
                {Items[n++] = ITList[i].ICode; break;}
        }

// List entries in each priority queue
//
   for (i = 0; i <= XrdFrmPstgReq::maxPrty; i++)
       {Offs = 0;
        while(rQueue[i]->List(myLfn,sizeof(myLfn),Offs, Items, n))
             cout <<myLfn <<endl;
       }
   cout <<endl;
}

/******************************************************************************/
/* Public:                        S e r v e r                                 */
/******************************************************************************/
  
int XrdFrmPstg::Server(int udpFD)
{
   EPNAME("Server");
   XrdOucStream Request(&Say);
   char *tp;

// Hookup to the udp socket as a stream
//
   Request.Attach(udpFD, 64*1024);

// Now simply get requests (see Agent() for details)
//
   while((tp = Request.GetLine()))
        {DEBUG("Request: '" <<tp <<"'");
         if ((tp = Request.GetToken()))
            {     if (*tp == '+'){Agent_Add(Request, tp+1);
                                  Server_Driver(1);
                                 }
             else if (*tp == '-') Agent_Del(Request, tp+1);
             else if (*tp == '?') {}
             else if (*tp == '!') Server_Driver(1);
             else Say.Emsg("Server", "Invalid request, '", tp, "'.");
            }
        }

// We should never get here (but....)
//
   Say.Emsg("Server", "Exiting; lost request connection!");
   return 8;
}

/******************************************************************************/
/* Public:                 S e r v e r _ D r i v e r                          */
/******************************************************************************/
  
void XrdFrmPstg::Server_Driver(int PushIt)
{
   static XrdSysMutex     rqMutex;
   static XrdSysSemaphore rqReady;
   static int isPosted = 0;

// If this is a PushIt then see if we need to push the binary semaphore
//
   if (PushIt) {rqMutex.Lock();
                if (!isPosted) {rqReady.Post(); isPosted = 1;}
                rqMutex.UnLock();
               }
      else     {rqMutex.Lock(); isPosted = 0; rqMutex.UnLock();
                rqReady.Wait();
                isPosted = 0;  // Advisory, lock not needed for this one
               }
}
  
/******************************************************************************/
/* Public:                  S e r v e r _ S t a g e                           */
/******************************************************************************/
  
void XrdFrmPstg::Server_Stage()
{
   XrdFrmPstgReq::Request myReq;
   int i, rc, numXfr, numPull;;

// Perform staging in an endless loop
//
do{Server_Driver(0);
   do{numXfr = 0;
      for (i = XrdFrmPstgReq::maxPrty; i >= 0; i--)
          {numPull = i+1;
           while(numPull && (rc = rQueue[i]->Get(&myReq)))
                {numPull -= XrdFrmPstgXfr::Queue(&myReq, i);
                 numXfr++;
                 if (rc < 0) break;
                }
          }
     } while(numXfr);
  } while(1);
}
