/******************************************************************************/
/*                                                                            */
/*                     X r d F r m X f r A g e n t . c c                      */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

const char *XrdFrmXfrAgentCVSID = "$Id$";

#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmRequest.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdFrm/XrdFrmUtils.hh"
#include "XrdFrm/XrdFrmXfrAgent.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysPlatform.hh"

using namespace XrdFrm;

/******************************************************************************/
/*                      S t a t i c   V a r i a b l e s                       */
/******************************************************************************/

XrdFrmReqAgent XrdFrmXfrAgent::GetAgent("getf", XrdFrmRequest::getQ);

XrdFrmReqAgent XrdFrmXfrAgent::MigAgent("migr", XrdFrmRequest::migQ);

XrdFrmReqAgent XrdFrmXfrAgent::StgAgent("pstg", XrdFrmRequest::stgQ);

XrdFrmReqAgent XrdFrmXfrAgent::PutAgent("putf", XrdFrmRequest::putQ);
  
/******************************************************************************/
/* Private:                          A d d                                    */
/******************************************************************************/
  
void XrdFrmXfrAgent::Add(XrdOucStream   &Request, char *Tok,
                         XrdFrmReqAgent &Server)
{
   XrdFrmRequest myReq;
   const char *Miss = 0;
   char *tp, *op;

// Handle: op[<traceid>] <requestid> <npath> <prty> <mode> <path> [. . .]
//
// op: + | & | ^ | < | = | >
//
   memset(&myReq, 0, sizeof(myReq));
   myReq.OPc = *Tok;
   if (*Tok == '=' || *Tok == '^') myReq.Options |= XrdFrmRequest::Purge;
   Tok++;

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
                   else if (myReq.Prty > XrdFrmRequest::maxPrty)
                            myReq.Prty = XrdFrmRequest::maxPrty;
               }
      }

   if (!Miss)
      {if (!(tp = Request.GetToken())) Miss = "mode";
          else myReq.Options = XrdFrmUtils::MapM2O(myReq.Notify, tp);
      }

   if (!Miss && !(tp = Request.GetToken())) Miss = "path";

// Check for any errors
//
   if (Miss) {Say.Emsg("Agent_Add", Miss, "missing in '+' request.");
              return;
             }

// Add all paths in the request
//
   do {strlcpy(myReq.LFN, tp, sizeof(myReq.LFN));
       if ((op = index(tp, '?'))) {myReq.Opaque = op-tp+1; *op = '\0';}
          else myReq.Opaque = 0;
       myReq.LFO = 0;
       if (myReq.LFN[0] != '/' && !(myReq.LFO = XrdFrmUtils::chkURL(myReq.LFN)))
          Say.Emsg("Agent_Add", "Invalid url -", myReq.LFN);
          else Server.Add(myReq);
       if ((tp = Request.GetToken())) memset(myReq.LFN, 0, sizeof(myReq.LFN));
      } while(tp);
}

/******************************************************************************/
/* Private:                        A g e n t                                  */
/******************************************************************************/

XrdFrmReqAgent *XrdFrmXfrAgent::Agent(char bType)
{

// Return the agent corresponding to the type
//
   switch(bType)
         {case 0  : return &StgAgent;
          case '+': return &StgAgent;
          case '^':
          case '&': return &MigAgent;
          case '<': return &GetAgent;
          case '=':
          case '>': return &PutAgent;
          default:  break;
         }
   return 0;
}

/******************************************************************************/
/* Private:                          D e l                                    */
/******************************************************************************/
  
void XrdFrmXfrAgent::Del(XrdOucStream  &Request, char *Tok,
                         XrdFrmReqAgent &Server)
{
   XrdFrmRequest myReq;

// If the requestid is adjacent to the operation, use it o/w get it
//
   if (!(*Tok) && (!(Tok = Request.GetToken()) || !(*Tok)))
      {Say.Emsg("Del", "request id missing in cancel request.");
       return;
      }

// Copy the request ID into the request and remove it from peer server
//
   memset(&myReq, 0, sizeof(myReq));
   strlcpy(myReq.ID, Tok, sizeof(myReq.ID));
   Server.Del(myReq);
}

/******************************************************************************/
/* Private:                         L i s t                                   */
/******************************************************************************/
  
void XrdFrmXfrAgent::List(XrdOucStream &Request, char *Tok)
{
   XrdFrmRequest::Item Items[XrdFrmRequest::getLast];
   XrdFrmReqAgent *agentP;
   int n = 0;
   char *tp;

   while((tp = Request.GetToken()) && n < XrdFrmRequest::getLast)
        {if (XrdFrmUtils::MapV2I(tp, Items[n])) n++;}

// List entries queued for specific servers
//
   if (!(*Tok)) {StgAgent.List(Items, n); GetAgent.List(Items, n);}
      else do {if ((agentP = Agent(*Tok))) agentP->List(Items, n);
              } while(*(++Tok));
   cout <<endl;
}

/******************************************************************************/
/* Public:                       P r o c e s s                                */
/******************************************************************************/
  
void XrdFrmXfrAgent::Process(XrdOucStream &Request)
{
   char *tp;

// Each frm request comes in as:
//
// Copy in:    <[<traceid>] <reqid> <npath> <prty> <mode> <path> [. . .]
// Copy purge: =[<traceid>] <reqid> <npath> <prty> <mode> <path> [. . .]
// Copy out:   >[<traceid>] <reqid> <npath> <prty> <mode> <path> [. . .]
// Migrate:    &[<traceid>] <reqid> <npath> <prty> <mode> <path> [. . .]
// Migr+Purge: ^[<traceid>] <reqid> <npath> <prty> <mode> <path> [. . .]
// Stage:      +[<traceid>] <reqid> <npath> <prty> <mode> <path> [. . .]
// Cancel in:  - <requestid>
// Cancel out: ~ <requestid>
// List:       ?[<][+][&][>]
// Wakeup:     ![<][+][&][>]
//
   if ((tp = Request.GetToken()))
      switch(*tp)
            {case '+':  Add(Request, tp,   StgAgent); break;
             case '<':  Add(Request, tp,   GetAgent); break;
             case '=':
             case '>':  Add(Request, tp,   PutAgent); break;
             case '&':
             case '^':  Add(Request, tp,   MigAgent); break;
             case '-':  Del(Request, tp+1, StgAgent);
                        Del(Request, tp+1, GetAgent);
                        break;
             case '~':  Del(Request, tp+1, MigAgent);
                        Del(Request, tp+1, PutAgent);
                        break;
             case '?': List(Request, tp+1);           break;
             case '!': GetAgent.Ping(tp);             break;
             default: Say.Emsg("Agent", "Invalid request, '", tp, "'.");
            }
}

/******************************************************************************/
/* Public:                         S t a r t                                  */
/******************************************************************************/
  
int XrdFrmXfrAgent::Start()
{
   EPNAME("Agent");
   XrdOucStream Request;
   char *tp;

// Prepare our agents
//
   if (!StgAgent.Start(Config.QPath, Config.AdminMode)
   ||  !MigAgent.Start(Config.QPath, Config.AdminMode)
   ||  !GetAgent.Start(Config.QPath, Config.AdminMode)
   ||  !PutAgent.Start(Config.QPath, Config.AdminMode)) return 2;

// Attach stdin to the Request stream
//
   Request.Attach(STDIN_FILENO, 8*1024);

// Process all input
//
   while((tp = Request.GetLine()))
        {DEBUG ("Request: '" <<tp <<"'");
         Process(Request);
        }

// If we exit then we lost the connection
//
   Say.Emsg("Agent", "Exiting; lost request connection!");
   return 8;
}
