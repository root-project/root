// @(#)root/proofd:$Id$
// Author: G. Ganis March 2011

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdLauncher                                                    //
//                                                                      //
// Author: G. Ganis, CERN, 2011                                         //
//                                                                      //
// Class describing the proofserv launcher interface                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <unistd.h>

#include "XrdProofdLauncher.h"

#include "rpdconn.h"
#include "XrdProofdAux.h"
#include "XrdProofdManager.h"
#include "XrdProofdProofServ.h"
#include "XrdProofdProtocol.h"
#include "XrdProofdTrace.h"
#include "XrdNet/XrdNetPeer.hh"
#include "XrdROOT.h"

//_________________________________________________________________________________
XrdNetPeer *XrdProofdLauncher::Launch(ProofdLaunch_t *in, int &pid)
{
   // Launch a session, establishing the UNIX connection over 'sockp';
   // retrieve the process id; returns the peer object describing the connection
   // and the pid. Or NULL in case of failure.
   XPDLOC(SMGR, "Launcher::Launch")

   XrdNetPeer *peer = 0;
   pid = -1;

   // Parse inputs
   if (!in) {
      TRACE(XERR, "undefined inputs!");
      return peer;
   }
   XrdProofdProofServ *xps = in->fPS;
   
   // Log prefix
   XrdOucString npfx;
   XPDFORM(npfx, "%s-%s:", (xps->SrvType() == kXPD_MasterWorker) ? "wrk" : "mst", xps->Ordinal());

   XrdOucString emsg;
   // The path to the executable
   if (!fClient->ROOT()) {
      TRACE(XERR, "ROOT version undefined!");
      // Dump to the log file
      XrdProofdAux::LogEmsgToFile(in->fErrLog.c_str(), "ROOT version undefined!", npfx.c_str());
      return peer;
   }
   XrdOucString pexe;
   XPDFORM(pexe, "%s/proofexecv", fClient->ROOT()->BinDir());
   if (access(pexe.c_str(), X_OK) != 0) {
      XPDFORM(emsg, "path '%s' does not exist or is not executable (errno: %d)", pexe.c_str(), (int)errno);
      TRACE(XERR, emsg);      
      XrdProofdAux::LogEmsgToFile(in->fErrLog.c_str(), emsg.c_str(), npfx.c_str());
      return peer;
   }
   
   // Create server socket to get the call back
   rpdunixsrv *unixsrv = new rpdunixsrv(xps->UNIXSockPath());
   if (!unixsrv || !unixsrv->isvalid(0)) {
      XPDFORM(emsg, "could not start unix server connection on path '%s' (errno: %d)",
                    xps->UNIXSockPath(), (int)errno);
      TRACE(XERR, emsg);      
      XrdProofdAux::LogEmsgToFile(in->fErrLog.c_str(), emsg.c_str(), npfx.c_str());
      return peer;
   }

   // Start the proofexecv
   XrdOucString cmd, exp;
   XPDFORM(exp, ". %s; export ROOTBINDIR=\"%s\"; %s %d %d", in->fEnvFile.c_str(), fClient->ROOT()->BinDir(),
                                                            pexe.c_str(), xps->SrvType(), in->fDbgLevel);
   XPDFORM(cmd, "%s %s \"%s\" %s %s &", exp.c_str(), fClient->User(), in->fSessionDir.c_str(),
                                        xps->UNIXSockPath(), in->fErrLog.c_str());
   TRACE(ALL, cmd);
   if (system(cmd.c_str()) == -1) {
      XPDFORM(emsg, "failure from 'system' (errno: %d)", (int)errno);
      TRACE(XERR, emsg);
      XrdProofdAux::LogEmsgToFile(in->fErrLog.c_str(), emsg.c_str(), npfx.c_str());
      return peer;
   }

   //
   // Accept a connection from the second server
   int err;
   rpdunix *uconn = unixsrv->accept(in->fIntWait, &err);
   if (!uconn || !uconn->isvalid(0)) {
      XPDFORM(emsg, "failure accepting callback (errno: %d)", -err);
      TRACE(XERR, emsg);
      XrdProofdAux::LogEmsgToFile(in->fErrLog.c_str(), emsg.c_str(), npfx.c_str());
      return peer;
   }
   TRACE(ALL, "proofserv connected!");

   //   
   // Setup the peer
   return SetupPeer(in, pid, uconn);
}

//_________________________________________________________________________________
XrdNetPeer *XrdProofdLauncher::SetupPeer(ProofdLaunch_t *in, int &pid, rpdunix *uconn)
{
   // Launch a session, establishing the UNIX connection over 'sockp';
   // retrieve the process id; returns the peer object describing the connection
   // and the pid. Or NULL in case of failure.
   XPDLOC(SMGR, "Launcher::SetupPeer")

   XrdNetPeer *peer = 0;
   pid = -1;
   XrdOucString emsg;

   // Parse inputs
   if (!in || !uconn || (uconn && !uconn->isvalid(1))) {
      TRACE(XERR, "undefined or invalid inputs!");
      return peer;
   }
   XrdProofdManager   *mgr = in->fMgr;
   XrdProofdProofServ *xps = in->fPS;
   
   // Log prefix
   XrdOucString npfx;
   XPDFORM(npfx, "%s-%s:", (xps->SrvType() == kXPD_MasterWorker) ? "wrk" : "mst", xps->Ordinal());

   //
   // Receive pid
   int rcc = 0;
   if ((rcc = uconn->recv(pid)) != 0) {
      XPDFORM(emsg, "failure receiving pid from the child (errno: %d)", -rcc);
      TRACE(XERR, emsg);
      XrdProofdAux::LogEmsgToFile(in->fErrLog.c_str(), emsg.c_str(), npfx.c_str());
      return peer;
   }
   TRACE(ALL, "child pid: " << pid);

   // Send admin path and executable
   rpdmsg msg;
   std::string srvadmin(mgr->AdminPath()), adminpath(xps->AdminPath()),
               pspath(fClient->ROOT()->PrgmSrv());
   int ppid = (int)getpid();
   msg << srvadmin << adminpath << pspath << ppid;
   if ((rcc = uconn->send(msg)) != 0) {
      XPDFORM(emsg, "ERROR: failure sending admin path and executable to child (errno: %d)", -rcc);
      TRACE(XERR, emsg);
      XrdProofdAux::LogEmsgToFile(in->fErrLog.c_str(), emsg.c_str(), npfx.c_str());
      return peer;
   }
   
   // Send information about dataset, data dir(s), credentials ...
   std::string creds(fClient->Sandbox()->Dir());
   creds += "/.creds";
   msg.reset();
   msg << (int) XrdProofdProtocol::EUidAtStartup() << std::string(fClient->Group()) << creds
       << std::string(xps->Ordinal()) << std::string(mgr->DataDir())
       << std::string(mgr->DataDirOpts()) << std::string(mgr->DataSetExp());
   if ((rcc = uconn->send(msg)) != 0) {
      XPDFORM(emsg, "ERROR: failure sending information about dataset, data dir ... to child (errno: %d)", -rcc);
      TRACE(XERR, emsg);
      XrdProofdAux::LogEmsgToFile(in->fErrLog.c_str(), emsg.c_str(), npfx.c_str());
      return peer;
   }
       
   // Wait for something to read on the socket
   int pollrc = uconn->pollrd(in->fIntWait);
   if (pollrc <= 0) {
      emsg = "ERROR: timeout while waiting for handshake information";
      if (pollrc < 0)
         XPDFORM(emsg, "ERROR: failure while waiting for handshake information (errno: %d)", errno);
      TRACE(XERR, emsg);
      XrdProofdAux::LogEmsgToFile(in->fErrLog.c_str(), emsg.c_str(), npfx.c_str());
      return peer;
   }
   TRACE(ALL, "information ready to be read: " << pollrc);

   // Create the peer object
   peer = new XrdNetPeer();
   peer->fd = uconn->exportfd();
   memcpy(&(peer->InetAddr), uconn->address(), sizeof(peer->InetAddr));
   peer->InetName = 0;
   delete uconn;
   
   // Done
   return peer;
}

