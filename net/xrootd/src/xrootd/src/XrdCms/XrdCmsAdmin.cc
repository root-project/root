/******************************************************************************/
/*                                                                            */
/*                        X r d C m s A d m i n . c c                         */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

// Original Version: 1.20 2007/07/31 02:25:11 abh

const char *XrdCmsAdminCVSID = "$Id$";

#include <stdio.h>
#include <limits.h>
#include <unistd.h>
#include <sys/types.h>

#include "XProtocol/YProtocol.hh"

#include "XrdCms/XrdCmsAdmin.hh"
#include "XrdCms/XrdCmsConfig.hh"
#include "XrdCms/XrdCmsManager.hh"
#include "XrdCms/XrdCmsPrepare.hh"
#include "XrdCms/XrdCmsState.hh"
#include "XrdCms/XrdCmsTrace.hh"
#include "XrdNet/XrdNetSocket.hh"
#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucName2Name.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"

using namespace XrdCms;
 
/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/
  
namespace XrdCms
{
class AdminReq
{
public:

       AdminReq *Next;
const  char     *Req;
const  char     *Path;
       CmsRRHdr  Hdr;
       char     *Data;
       int       Dlen;
static int       numinQ;
static const int maxinQ = 1024;

static AdminReq *getReq() {AdminReq *arP;
                           do {QPresent.Wait();
                               QMutex.Lock();
                               if ((arP = First))
                                  {if (!(First = arP->Next)) Last = 0;
                                   numinQ--;
                                  }
                               QMutex.UnLock();
                              } while (!arP);
                           return arP;
                          }

       void      Requeue() {QMutex.Lock();
                            Next=First; First=this; QPresent.Post(); numinQ++;
                            QMutex.UnLock();
                           }

          AdminReq(const char *req, XrdCmsRRData &RRD) 
                  : Next(0), Req(req), Path(RRD.Path ? RRD.Path : ""),
                    Hdr(RRD.Request), Data(RRD.Buff), Dlen(RRD.Dlen)
                  {RRD.Buff = 0;
                   QMutex.Lock();
                   if (Last) {Last->Next = this; Last = this;}
                      else    First=Last = this;
                   QPresent.Post();
                   numinQ++;
                   QMutex.UnLock();
                  }

     ~AdminReq() {if (Data) free(Data);}

private:

static XrdSysSemaphore QPresent;
static XrdSysMutex     QMutex;
static AdminReq       *First;
static AdminReq       *Last;
};
};

/******************************************************************************/
/*                     G l o b a l s   &   S t a t i c s                      */
/******************************************************************************/

       XrdSysSemaphore  AdminReq::QPresent(0);
       XrdSysMutex      AdminReq::QMutex;
       AdminReq        *AdminReq::First = 0;
       AdminReq        *AdminReq::Last  = 0;
       int              AdminReq::numinQ= 0;

       XrdSysMutex      XrdCmsAdmin::myMutex;
       XrdSysSemaphore *XrdCmsAdmin::SyncUp = 0;
       int              XrdCmsAdmin::POnline= 0;

/******************************************************************************/
/*            E x t e r n a l   T h r e a d   I n t e r f a c e s             */
/******************************************************************************/
  
void *XrdCmsAdminLogin(void *carg)
      {XrdCmsAdmin *Admin = new XrdCmsAdmin();
       Admin->Login(*(int *)carg);
       delete Admin;
       return (void *)0;
      }

void *XrdCmsAdminSend(void *carg)
      {XrdCmsAdmin::Relay(0,0);
       return (void *)0;
      }
 
/******************************************************************************/
/*                                 L o g i n                                  */
/******************************************************************************/
  
void XrdCmsAdmin::Login(int socknum)
{
   const char *epname = "Admin_Login";
   char *request, *tp;

// Attach the socket FD to a stream
//
   Stream.Attach(socknum);

// The first request better be "login"
//
   if ((request = Stream.GetLine()))
      {DEBUG("initial request: '" <<request <<"'");
       if (!(tp = Stream.GetToken()) || strcmp("login", tp) || !do_Login())
          {Say.Emsg(epname, "Invalid admin login sequence");
           return;
          }
       } else {Say.Emsg(epname, "No admin login specified");
               return;
              }

// Start receiving requests on this stream
//
   while((request = Stream.GetLine()))
        {DEBUG("received request: '" <<request <<"'");
         if ((tp = Stream.GetToken()))
            {     if (!strcmp("resume",   tp))
                      CmsState.Update(XrdCmsState::Active, 1);
             else if (!strcmp("rmdid",    tp)) do_RmDid();   // via lfn
             else if (!strcmp("newfn",    tp)) do_RmDud();   // via lfn
             else if (!strcmp("suspend",  tp)) 
                     {CmsState.Update(XrdCmsState::Active, 0);
                      Say.Emsg("Notes","suspend requested by",Stype,Sname);
                     }
             else Say.Emsg(epname, "invalid admin request,", tp);
            }
        }

// The socket disconnected
//
   Say.Emsg("Login", Stype, Sname, "logged out");

// If this is a primary, we must suspend but do not record this event!
//
   if (Primary) 
      {CmsState.Update(XrdCmsState::FrontEnd, 0, -1);
       myMutex.Lock();
       POnline = 0;
       Relay(1,-1);
       myMutex.UnLock();
      }
   return;
}

/******************************************************************************/
/*                                 N o t e s                                  */
/******************************************************************************/
  
void *XrdCmsAdmin::Notes(XrdNetSocket *AnoteSock)
{
   const char *epname = "Notes";
   char *request, *tp;
   int rc;

// Bind the udp socket to a stream
//
   Stream.Attach(AnoteSock->Detach());
   Sname = strdup("anon");

// Accept notifications in an endless loop
//
   do {while((request = Stream.GetLine()))
            {DEBUG("received notification: '" <<request <<"'");
             if ((tp = Stream.GetToken()))
                {     if (!strcmp("gone",    tp)) do_RmDid(1); // via pfn
                 else if (!strcmp("rmdid",   tp)) do_RmDid(0); // via lfn
                 else if (!strcmp("have",    tp)) do_RmDud(1); // via pfn
                 else if (!strcmp("newfn",   tp)) do_RmDud(0); // via lfn
                 else if (!strcmp("nostage", tp))
                         {CmsState.Update(XrdCmsState::Stage, 0);
                          Say.Emsg("Notes","nostage requested by",Stype,Sname);
                         }
                 else if (!strcmp("stage",   tp))
                          CmsState.Update(XrdCmsState::Stage, 1);
                 else Say.Emsg(epname, "invalid notification,", tp);
                }
            }
       if ((rc = Stream.LastError())) break;
       rc = Stream.Detach(); Stream.Attach(rc);
      } while(1);

// We should never get here
//
   Say.Emsg(epname, rc, "accept notification");
   return (void *)0;
}

/******************************************************************************/
/*                                 R e l a y                                  */
/******************************************************************************/
  
void XrdCmsAdmin::Relay(int setSock, int newSock)
{
   const char            *epname = "Admin_Relay";
   static const int       HdrSz = sizeof(CmsRRHdr);
   static XrdSysMutex     SMutex;
   static XrdSysSemaphore SReady(0);
   static int             curSock = -1;
   AdminReq              *arP;
   int                    retc, mySock = -1;

// Set socket for writing (called from admin thread when pimary logs on)
   if (setSock)
      {SMutex.Lock();
       if (curSock >= 0) close(curSock);
          else if (newSock >= 0) SReady.Post();
       if (newSock < 0) curSock = -1;
          else {curSock = dup(newSock); XrdNetSocket::setOpts(curSock, 0);}
       SMutex.UnLock();
       return;
      }

// This is just an endless loop
//
   do {while(mySock < 0)
            {SMutex.Lock();
             if (curSock < 0) {SMutex.UnLock(); SReady.Wait(); SMutex.Lock();}
             mySock = curSock; curSock = -1;
             SMutex.UnLock();
            }

       do {arP = AdminReq::getReq();

           if ((retc = write(mySock, &arP->Hdr, HdrSz))     != HdrSz
           ||  (retc = write(mySock, arP->Data, arP->Dlen)) != arP->Dlen)
              retc = (retc < 0 ? errno : ECANCELED);
              else {DEBUG("sent " <<arP->Req <<' ' <<arP->Path);
                    delete arP; retc = 0;
                   }
          } while(retc == 0);

       if (retc) Say.Emsg("AdminRelay", retc, "relay", arP->Req);
       arP->Requeue();
       close(mySock);
       mySock = -1;
      } while(1);
}

/******************************************************************************/
/*                                  S e n d                                   */
/******************************************************************************/
  
void XrdCmsAdmin::Send(const char *Req, XrdCmsRRData &Data)
{
   AdminReq *arP;

   if (AdminReq::numinQ < AdminReq::maxinQ) arP = new AdminReq(Req, Data);
      else Say.Emsg("Send", "Queue full; ignoring", Req, Data.Path);
}

/******************************************************************************/
/*                                 S t a r t                                  */
/******************************************************************************/
  
void *XrdCmsAdmin::Start(XrdNetSocket *AdminSock)
{
   const char *epname = "Start";
   int InSock;
   pthread_t tid;

// Start the relay thread
//
   if (XrdSysThread::Run(&tid,XrdCmsAdminSend,(void *)0))
      Say.Emsg(epname, errno, "start admin relay");

// If we are in independent mode then let the caller continue
//
   if (Config.doWait)
      {Say.Emsg(epname, "Waiting for primary server to login.");}
       else if (SyncUp) {SyncUp->Post(); SyncUp = 0;}

// Accept connections in an endless loop
//
   while(1) if ((InSock = AdminSock->Accept()) >= 0)
               {XrdNetSocket::setOpts(InSock, 0);
                if (XrdSysThread::Run(&tid,XrdCmsAdminLogin,(void *)&InSock))
                   {Say.Emsg(epname, errno, "start admin");
                    close(InSock);
                   }
               } else Say.Emsg(epname, errno, "accept connection");
   return (void *)0;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                              d o _ L o g i n                               */
/******************************************************************************/
  
int XrdCmsAdmin::do_Login()
{
   const char *emsg;
   char buff[64], *tp, Ltype = 0;
   int Port = 0;

// Process: login {p | P | s | u} <name> [port <port>]
//
   if (!(tp = Stream.GetToken()))
      {Say.Emsg("do_Login", "login type not specified");
       return 0;
      }

   Ltype = *tp;
   if (*(tp+1) == '\0')
      switch (*tp)
             {case 'p': Stype = "Primary server"; break;
              case 'P': Stype = "Proxy server";   break;
              case 's': Stype = "Server";         break;
              case 'u': Stype = "Admin";          break;
              default:  Ltype = 0;                break;
             }

   if (!Ltype)
      {Say.Emsg("do_Login", "Invalid login type,", tp);
       return 0;
      } else Ltype = *tp;

   if (!(tp = Stream.GetToken()))
      {Say.Emsg("do_Login", "login name not specified");
       return 0;
      } else Sname = strdup(tp);

// Get any additional options
//
   while((tp = Stream.GetToken()))
        {     if (!strcmp(tp, "port"))
                 {if (!(tp = Stream.GetToken()))
                     {Say.Emsg("do_Login", "login port not specified");
                      return 0;
                     }
                  if (XrdOuca2x::a2i(Say,"login port",tp,&Port,0))
                     return 0;
                 }
         else    {Say.Emsg("do_Login", "invalid login option -", tp);
                  return 0;
                 }
        }

// If this is not a primary, we are done. Otherwise there is much more. We
// must make sure we are compatible with the login
//
   if (Ltype != 'p' && Ltype != 'P') return 1;
        if (Ltype == 'p' &&  Config.asProxy()) emsg = "only accepts proxies";
   else if (Ltype == 'P' && !Config.asProxy()) emsg = "does not accept proxies";
   else                                        emsg = 0;
   if (emsg) 
      {Say.Emsg("do_login", "Server login rejected; configured role", emsg);
       return 0;
      }

// Discard login if this is a duplicate primary server
//
   myMutex.Lock();
   if (POnline)
      {myMutex.UnLock();
       Say.Emsg("do_Login", "Primary server already logged in; login of", 
                                   tp, "rejected.");
       return 0;
      }

// Indicate we have a primary
//
   Primary = 1;
   POnline = 1;
   Relay(1, Stream.FDNum());
   CmsState.Update(XrdCmsState::FrontEnd, 1, Port);

// Check if this is the first primary login and resume if we must
//
   if (SyncUp) {SyncUp->Post(); SyncUp = 0;}
   myMutex.UnLock();

// Document the login
//
   sprintf(buff, "logged in; data port is %d", Port);
   Say.Emsg("do_Login:", Stype, Sname, buff);
   return 1;
}
 
/******************************************************************************/
/*                              d o _ R m D i d                               */
/******************************************************************************/
  
void XrdCmsAdmin::do_RmDid(int isPfn)
{
   const char *epname = "do_RmDid";
   char  *tp, *thePath, apath[XrdCmsMAX_PATH_LEN];
   int   rc;

   if (!(tp = Stream.GetToken()))
      {Say.Emsg(epname,"removed path not specified by",Stype,Sname);
       return;
      }

// Handle prepare queue removal
//
   if (PrepQ.isOK())
      {if (!isPfn && Config.lcl_N2N)
          if ((rc = Config.lcl_N2N->lfn2pfn(tp, apath, sizeof(apath))))
             {Say.Emsg(epname, rc, "determine pfn for removed path", tp);
              thePath = 0;
             } else thePath = apath;
          else thePath = tp;
       if (thePath) PrepQ.Gone(thePath);
      }

// If we have a pfn then we must get the lfn to inform our manager about the file
//
   if (isPfn && Config.lcl_N2N)
      {if ((rc = Config.lcl_N2N->pfn2lfn(tp, apath, sizeof(apath))))
          {Say.Emsg(epname, rc, "determine lfn for removed path", tp);
           return;
          } else tp = apath;
      }

   DEBUG("sending managers gone " <<tp);
   Manager.Inform(kYR_gone, kYR_raw, tp, strlen(tp)+1);
}
 
/******************************************************************************/
/*                              d o _ R m D u d                               */
/******************************************************************************/
  
void XrdCmsAdmin::do_RmDud(int isPfn)
{
   const char *epname = "do_RmDud";
   char *tp, *pp, apath[XrdCmsMAX_PATH_LEN];
   int   rc, Mods = kYR_raw;

   if (!(tp = Stream.GetToken()))
      {Say.Emsg(epname,"added path not specified by",Stype,Sname);
       return;
      }

   if ((pp = Stream.GetToken()) && *pp == 'p') Mods |= CmsHaveRequest::Pending;
      else Mods |= CmsHaveRequest::Online;

   if (isPfn && Config.lcl_N2N)
      {if ((rc = Config.lcl_N2N->pfn2lfn(tp, apath, sizeof(apath))))
          {Say.Emsg(epname, rc, "determine lfn for added path", tp);
           return;
          } else tp = apath;
      }

   DEBUG("sending managers have online " <<tp);
   Manager.Inform(kYR_have, Mods, tp, strlen(tp)+1);
}
