/******************************************************************************/
/*                                                                            */
/*                    X r d F r m X f r D a e m o n . c c                     */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

const char *XrdFrmXfrDaemonCVSID = "$Id$";

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmMigrate.hh"
#include "XrdFrm/XrdFrmRequest.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdFrm/XrdFrmTransfer.hh"
#include "XrdFrm/XrdFrmUtils.hh"
#include "XrdFrm/XrdFrmXfrAgent.hh"
#include "XrdFrm/XrdFrmXfrDaemon.hh"
#include "XrdNet/XrdNetOpts.hh"
#include "XrdNet/XrdNetSocket.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSys/XrdSysTimer.hh"

using namespace XrdFrm;

/******************************************************************************/
/*                      S t a t i c   V a r i a b l e s                       */
/******************************************************************************/

XrdFrmReqBoss XrdFrmXfrDaemon::GetBoss("getf", XrdFrmRequest::getQ);

XrdFrmReqBoss XrdFrmXfrDaemon::MigBoss("migr", XrdFrmRequest::migQ);

XrdFrmReqBoss XrdFrmXfrDaemon::StgBoss("pstg", XrdFrmRequest::stgQ);

XrdFrmReqBoss XrdFrmXfrDaemon::PutBoss("putf", XrdFrmRequest::putQ);

/******************************************************************************/
/* Private:                         B o s s                                   */
/******************************************************************************/

XrdFrmReqBoss *XrdFrmXfrDaemon::Boss(char bType)
{

// Return the boss corresponding to the type
//
   switch(bType)
         {case 0  :
          case '+': return &StgBoss;
          case '^':
          case '&': return &MigBoss;
          case '<': return &GetBoss;
          case '=':
          case '>': return &PutBoss;
          default:  break;
         }
   return 0;
}

/******************************************************************************/
/* Public:                          I n i t                                   */
/******************************************************************************/

int XrdFrmXfrDaemon::Init()
{
   char buff[80];

// Make sure we are the only daemon running
//
   sprintf(buff, "%s/frm_xfrd.lock", Config.QPath);
   if (!XrdFrmUtils::Unique(buff, Config.myProg)) return 0;

// Initiliaze the transfer processor (it need to be active now)
//
   if (!XrdFrmTransfer::Init()) return 0;

// Fix up some values that might not make sense
//
   if (Config.WaitMigr < Config.IdleHold) Config.WaitMigr = Config.IdleHold;

// Check if it makes any sense to migrate and, if so, initialize migration
//
   if (Config.pathList)
      {if (!Config.xfrOUT)
          Say.Emsg("Config","Output copy command not specified; "
                            "auto-migration disabled!");
          else XrdFrmMigrate::Migrate();
      } else Say.Emsg("Config","No migratable paths; "
                               "auto-migration disabled!");

// Start the external interfaces
//
   if (!StgBoss.Start(Config.QPath, Config.AdminMode)
   ||  !MigBoss.Start(Config.QPath, Config.AdminMode)
   ||  !GetBoss.Start(Config.QPath, Config.AdminMode)
   ||  !PutBoss.Start(Config.QPath, Config.AdminMode)) return 0;

// All done
//
   return 1;
}
  
/******************************************************************************/
/* Public:                          P o n g                                   */
/******************************************************************************/
  
void *XrdFrmXfrDaemonPong(void *parg)
{
    XrdFrmXfrDaemon::Pong();
    return (void *)0;
}
  
void XrdFrmXfrDaemon::Pong()
{
   EPNAME("Pong");
   static int udpFD = -1;
   XrdOucStream Request(&Say);
   XrdFrmReqBoss *bossP;
   char *tp;

// Get a UDP socket for the server if we haven't already done so and start
// a thread to re-enter this code and wait for messages from an agent.
//
   if (udpFD < 0)
      {XrdNetSocket *udpSock;
       pthread_t tid;
       int retc;
       if ((udpSock = XrdNetSocket::Create(&Say, Config.QPath,
                   "xfrd.udp", Config.AdminMode, XRDNET_UDPSOCKET)))
          {udpFD = udpSock->Detach(); delete udpSock;
           if ((retc = XrdSysThread::Run(&tid, XrdFrmXfrDaemonPong, (void *)0,
                                         XRDSYSTHREAD_BIND, "Pong")))
              Say.Emsg("main", retc, "create udp listner");
          }
       return;
      }

// Hookup to the udp socket as a stream
//
   Request.Attach(udpFD, 64*1024);

// Now simply get requests (see XrdFrmXfrDaemon for details). Here we screen
// out ping and list requests.
//
   while((tp = Request.GetLine()))
        {DEBUG(": '" <<tp <<"'");
         switch(*tp)
               {case '?': break;
                case '!': if ((tp = Request.GetToken()))
                             while(*tp++)
                                  {if ((bossP = Boss(*tp))) bossP->Wakeup(1);}
                          break;
                default:  XrdFrmXfrAgent::Process(Request);
               }
        }

// We should never get here (but....)
//
   Say.Emsg("Server", "Lost udp connection!");
}

/******************************************************************************/
/* Public:                         S t a r t                                  */
/******************************************************************************/
  
int XrdFrmXfrDaemon::Start()
{

// Start the ponger
//
   Pong();

// Now start nudging
//
   while(1)
        {StgBoss.Wakeup(); GetBoss.Wakeup();
         MigBoss.Wakeup(); PutBoss.Wakeup();
         XrdSysTimer::Snooze(Config.WaitQChk);
        }

// We should never get here
//
   return 0;
}
