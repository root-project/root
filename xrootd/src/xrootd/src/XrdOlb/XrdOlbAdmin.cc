/******************************************************************************/
/*                                                                            */
/*                        X r d O l b A d m i n . c c                         */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

const char *XrdOlbAdminCVSID = "$Id$";

#include <fcntl.h>
#include <limits.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdOlb/XrdOlbAdmin.hh"
#include "XrdOlb/XrdOlbConfig.hh"
#include "XrdOlb/XrdOlbManager.hh"
#include "XrdOlb/XrdOlbPrepare.hh"
#include "XrdOlb/XrdOlbTrace.hh"
#include "XrdOuc/XrdOuca2x.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdOuc/XrdOucName2Name.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdNet/XrdNetSocket.hh"

using namespace XrdOlb;
 
/******************************************************************************/
/*                     G l o b a l s   &   S t a t i c s                      */
/******************************************************************************/

       XrdSysMutex      XrdOlbAdmin::myMutex;
       XrdSysSemaphore *XrdOlbAdmin::SyncUp = 0;
       int              XrdOlbAdmin::POnline= 0;

/******************************************************************************/
/*            E x t e r n a l   T h r e a d   I n t e r f a c e s             */
/******************************************************************************/
  
void *XrdOlbLoginAdmin(void *carg)
      {XrdOlbAdmin *Admin = new XrdOlbAdmin();
       Admin->Login(*(int *)carg);
       delete Admin;
       return (void *)0;
      }
 
/******************************************************************************/
/*                                 L o g i n                                  */
/******************************************************************************/
  
void XrdOlbAdmin::Login(int socknum)
{
   const char *epname = "Admin_Login";
   char *request, *tp;

// Attach the socket FD to a stream
//
   Stream.Attach(socknum);

// The first request better be "login"
//
   if ((request = Stream.GetLine()))
      {DEBUG("Initial admin request: '" <<request <<"'");
       if (!(tp = Stream.GetToken()) || strcmp("login", tp) || !do_Login())
          {Say.Emsg(epname, "Invalid admin login sequence");
           return;
          }
       } else {Say.Emsg(epname, "No admin login specified");
               return;
              }

// Document the login
//
   Say.Emsg(epname, Stype, Sname, "logged in");

// Start receiving requests on this stream
//
   while((request = Stream.GetLine()))
        {DEBUG("received admin request: '" <<request <<"'");
         if ((tp = Stream.GetToken()))
            {     if (!strcmp("resume",   tp)) do_Resume();
             else if (!strcmp("rmdid",    tp)) do_RmDid();   // via lfn
             else if (!strcmp("newfn",    tp)) do_RmDud();   // via lfn
             else if (!strcmp("suspend",  tp)) do_Suspend();
             else Say.Emsg(epname, "invalid admin request,", tp);
            }
        }

// The socket disconnected
//
   Say.Emsg("Login", Stype, Sname, "logged out");

// If this is a primary, we must suspend but do not record this event!
//
   if (Primary) 
      {myMutex.Lock();
       Manager.Suspend();
       POnline = 0;
       myMutex.UnLock();
      }
   return;
}

/******************************************************************************/
/*                                 N o t e s                                  */
/******************************************************************************/
  
void *XrdOlbAdmin::Notes(XrdNetSocket *AnoteSock)
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
                 else if (!strcmp("nostage", tp)) do_NoStage();
                 else if (!strcmp("stage",   tp)) do_Stage();
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
/*                                 S t a r t                                  */
/******************************************************************************/
  
void *XrdOlbAdmin::Start(XrdNetSocket *AdminSock)
{
   const char *epname = "Start";
   int InSock;
   pthread_t tid;

// If we are in independent mode then let the caller continue
//
   if (Config.doWait && Config.asServer() || Config.asSolo())
      Say.Emsg(epname, "Waiting for primary server to login.");
      else if (SyncUp) {SyncUp->Post(); SyncUp = 0;}

// Accept connections in an endless loop
//
   while(1) if ((InSock = AdminSock->Accept()) >= 0)
               {if (XrdSysThread::Run(&tid,XrdOlbLoginAdmin,(void *)&InSock))
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
  
int XrdOlbAdmin::do_Login()
{
   const char *emsg;
   char *tp, Ltype = 0;
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
   if (Config.doWait) Manager.setPort(Port);

// Check if this is the first primary login and resume if we must
//
   if (SyncUp)
      {SyncUp->Post();
       SyncUp = 0;
       myMutex.UnLock();
       return 1;
      }
   Manager.Resume();
   myMutex.UnLock();

   return 1;
}
 
/******************************************************************************/
/*                            d o _ N o S t a g e                             */
/******************************************************************************/
  
void XrdOlbAdmin::do_NoStage()
{
   Say.Emsg("do_NoStage", "nostage requested by", Stype, Sname);
   Manager.Stage(0);
   close(open(Config.NoStageFile, O_WRONLY|O_CREAT, S_IRUSR|S_IWUSR));
}
 
/******************************************************************************/
/*                             d o _ R e s u m e                              */
/******************************************************************************/
  
void XrdOlbAdmin::do_Resume()
{
   Say.Emsg("do_Resume", "resume requested by", Stype, Sname);
   unlink(Config.SuspendFile);
   Manager.Resume();
}
 
/******************************************************************************/
/*                              d o _ R m D i d                               */
/******************************************************************************/
  
void XrdOlbAdmin::do_RmDid(int isPfn)
{
   const char *epname = "do_RmDid";
   const char *cmd = "gone ";
   const int   cmdl= strlen(cmd);
   char  *tp, *thePath, apath[XrdOlbMAX_PATH_LEN];
   int   rc;

   if (!(tp = Stream.GetToken()))
      {Say.Emsg(epname,"removed path not specified by",Stype,Sname);
       return;
      }

// Handle prepare queue removal
//
   if (Config.PrepOK)
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
      if ((rc = Config.lcl_N2N->pfn2lfn(tp, apath, sizeof(apath))))
         {Say.Emsg(epname, rc, "determine lfn for removed path", tp);
          return;
         } else tp = apath;

   DEBUG("Sending managers " <<cmd <<tp);
   Manager.Inform(cmd, cmdl, tp, 0);
}
 
/******************************************************************************/
/*                              d o _ R m D u d                               */
/******************************************************************************/
  
void XrdOlbAdmin::do_RmDud(int isPfn)
{
   const char *epname = "do_RmDud";
   const char *cmd = "have ? ";
   const int   cmdl= strlen(cmd);
   char *tp, apath[XrdOlbMAX_PATH_LEN];
   int   rc;

   if (!(tp = Stream.GetToken()))
      {Say.Emsg(epname,"added path not specified by",Stype,Sname);
       return;
      }

   if (isPfn && Config.lcl_N2N)
      if ((rc = Config.lcl_N2N->pfn2lfn(tp, apath, sizeof(apath))))
         {Say.Emsg(epname, rc, "determine lfn for added path", tp);
          return;
         } else tp = apath;

   DEBUG("Sending managers " <<cmd <<tp);
   Manager.Inform(cmd, cmdl, tp, 0);
}
 
/******************************************************************************/
/*                              d o _ S t a g e                               */
/******************************************************************************/
  
void XrdOlbAdmin::do_Stage()
{
   Say.Emsg("do_Stage", "stage requested by", Stype, Sname);
   Manager.Stage(1);
   unlink(Config.NoStageFile);
}
  
/******************************************************************************/
/*                            d o _ S u s p e n d                             */
/******************************************************************************/
  
void XrdOlbAdmin::do_Suspend()
{
   Say.Emsg("do_Suspend", "suspend requested by", Stype, Sname);
   Manager.Suspend();
   close(open(Config.SuspendFile, O_WRONLY|O_CREAT, S_IRUSR|S_IWUSR));
}
