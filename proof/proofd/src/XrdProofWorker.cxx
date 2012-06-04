// @(#)root/proofd:$Id$
// Author: Gerardo Ganis  June 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofWorker                                                       //
//                                                                      //
// Authors: G. Ganis, CERN, 2007                                        //
//                                                                      //
// Class with information about a potential worker.                     //
// A list of instances of this class is built using the config file or  //
// or the information collected from the resource discoverers.          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <time.h>

#include "XrdProofdAux.h"
#include "XrdProofWorker.h"
#include "XrdProofdProofServ.h"
#include "XrdClient/XrdClientUrlInfo.hh"
#include "XpdSysDNS.h"
#include "XProofProtocol.h"

// Tracing utilities
#include "XrdProofdTrace.h"

//______________________________________________________________________________
XrdProofWorker::XrdProofWorker(const char *str)
   : fExport(256), fType('W'), fPort(-1), fPerfIdx(100), fActive(1)
{
   // Constructor from a config file-like string

   fMutex = new XrdSysRecMutex;

   // Make sure we got something to parse
   if (!str || strlen(str) <= 0)
      return;

   // The actual work is done by Reset()
   Reset(str);
}
//__________________________________________________________________________
XrdProofWorker::~XrdProofWorker()
{
   // Destructor

   SafeDel(fMutex);
}

//______________________________________________________________________________
void XrdProofWorker::Reset(const char *str)
{
   // Set content from a config file-like string
   XPDLOC(NMGR, "Worker::Reset")


   // Reinit vars
   fExport = "";
   fType = 'W';
   fHost = "";
   fPort = XPD_DEF_PORT;
   fPerfIdx = 100;
   fImage = "";
   fWorkDir = "";
   fMsd = "";
   fId = "";

   // Make sure we got something to parse
   if (!str || strlen(str) <= 0)
      return;

   // Tokenize the string
   XrdOucString s(str);

   // First token is the type
   XrdOucString tok;
   XrdOucString typestr = "master|submaster|worker|slave";
   int from = s.tokenize(tok, 0, ' ');
   if (from == STR_NPOS || typestr.find(tok) == STR_NPOS)
      return;
   if (tok == "submaster")
      fType = 'S';
   else if (tok == "master")
      fType = 'M';

   // Next token is the user@host:port string, make sure it is a full qualified host name
   if ((from = s.tokenize(tok, from, ' ')) == STR_NPOS)
      return;
   XrdClientUrlInfo ui(tok.c_str());
   // Take the user name, if specified
   fUser = ui.User;
   char *err;
   char *fullHostName = XrdSysDNS::getHostName((char *)ui.Host.c_str(), &err);
   if (!fullHostName || !strcmp(fullHostName, "0.0.0.0")) {
      TRACE(XERR, "DNS could not resolve '" << ui.Host << "'");
      return;
   }
   fHost = fullHostName;
   SafeFree(fullHostName);
   // Take the port, if specified
   fPort = (ui.Port > 0) ? ui.Port : fPort;

   // and then the remaining options
   while ((from = s.tokenize(tok, from, ' ')) != STR_NPOS) {
      if (tok.beginswith("workdir=")) {
         // Working dir
         tok.replace("workdir=", "");
         fWorkDir = tok;
      } else if (tok.beginswith("image=")) {
         // Image
         tok.replace("image=", "");
         fImage = tok;
      } else if (tok.beginswith("msd=")) {
         // Mass storage domain
         tok.replace("msd=", "");
         fMsd = tok;
      } else if (tok.beginswith("port=")) {
         // Port
         tok.replace("port=", "");
         fPort = strtol(tok.c_str(), (char **)0, 10);
      } else if (tok.beginswith("perf=")) {
         // Performance index
         tok.replace("perf=", "");
         fPerfIdx = strtol(tok.c_str(), (char **)0, 10);
      } else if (!tok.beginswith("repeat=")) {
         // Unknown
         TRACE(XERR, "ignoring unknown option '" << tok << "'");
      }
   }
}

//______________________________________________________________________________
bool XrdProofWorker::Matches(const char *host)
{
   // Check compatibility of host with this instance.
   // return 1 if compatible.

   return ((fHost.matches(host)) ? 1 : 0);
}

//______________________________________________________________________________
bool XrdProofWorker::Matches(XrdProofWorker *wrk)
{
   // Set content from a config file-like string

   // Check if 'wrk' is on the same node that 'this'; used to find the unique
   // worker nodes.
   // return 1 if the node is the same.

   if (wrk) {
      // Check Host names
      if (wrk->fHost == fHost) {
         // Check ports
         int pa = (fPort > 0) ? fPort : XPD_DEF_PORT;
         int pb = (wrk->fPort > 0) ? wrk->fPort : XPD_DEF_PORT;
         if (pa == pb)
            return 1;
      }
   }

   // They do not match
   return 0;
}

//______________________________________________________________________________
const char *XrdProofWorker::Export(const char *ord)
{
   // Export current content in a form understood by parsing algorithms
   // inside the PROOF session, i.e.
   // <type>|<user@host>|<port>|<ord>|-|<perfidx>|<img>|<workdir>|<msd>
   XPDLOC(NMGR, "Worker::Export")

   fExport = fType;

   // Add user@host
   fExport += '|' ;
   if (fUser.length() > 0) {
      fExport += fUser;
      fExport += "@";
   }
   fExport += fHost;

   // Add port
   if (fPort > 0) {
      fExport += '|' ;
      fExport += fPort;
   } else
      fExport += "|-";

   // Ordinal only if passed as argument
   if (ord && strlen(ord) > 0) {
      // Add ordinal
      fExport += '|' ;
      fExport += ord;
   } else {
      // No ordinal at this level
      fExport += "|-";
   }
   // ID at this level
   fExport += "|-";

   // Add performance index
   fExport += '|' ;
   fExport += fPerfIdx;

   // Add image
   if (fImage.length() > 0) {
      fExport += '|' ;
      fExport += fImage;
   } else
      fExport += "|-";

   // Add workdir
   if (fWorkDir.length() > 0) {
      fExport += '|' ;
      fExport += fWorkDir;
   } else
      fExport += "|-";

   // Add mass storage domain
   if (fMsd.length() > 0) {
      fExport += '|' ;
      fExport += fMsd;
   } else
      fExport += "|-";

   // We are done
   TRACE(DBG, "sending: " << fExport);
   return fExport.c_str();
}

//______________________________________________________________________________
int XrdProofWorker::GetNActiveSessions()
{
   // Calculate the number of workers existing on this node which are
   // currently running.
   // TODO: optimally, one could contact the packetizer and count the
   // opened files.

   int myRunning = 0;
   std::list<XrdProofdProofServ *>::iterator iter;
   XrdSysMutexHelper mhp(fMutex);
   for (iter = fProofServs.begin(); iter != fProofServs.end(); ++iter) {
      if (*iter) {
         if ((*iter)->Status() == kXPD_running)
            myRunning++;
      }
   }
   return myRunning;
}

//______________________________________________________________________________
void XrdProofWorker::MergeProofServs(const XrdProofWorker &other)
{
   // Merge session objects from the other worker object in order to merge all
   // the objects in only one. This was added to support hybrid satatically and
   // dinamically Bonjour workers discovery.

   std::list<XrdProofdProofServ *>::const_iterator iter;
   XrdSysMutexHelper mhp(fMutex);
   for (iter = other.fProofServs.begin(); iter != other.fProofServs.end(); ++iter) {
      this->fProofServs.push_back(*iter);
   }
}

//______________________________________________________________________________
void XrdProofWorker::Sort(std::list<XrdProofWorker *> *lst,
                          bool (*f)(XrdProofWorker *&lhs, XrdProofWorker *&rhs))
{
   // Sort ascendingly the list according to the comparing algorithm defined
   // by 'f'; 'f' should return 'true' if 'rhs' > 'lhs'.
   // This is implemented because on Solaris where std::list::sort() does not
   // support an alternative comparison algorithm.

   // Check argument
   if (!lst)
      return;

   // If empty or just one element, nothing to do
   if (lst->size() < 2)
      return;

   // Fill a temp array with the current status
   XrdProofWorker **ta = new XrdProofWorker *[lst->size() - 1];
   std::list<XrdProofWorker *>::iterator i = lst->begin();
   i++; // skip master
   int n = 0;
   for (; i != lst->end(); ++i)
      ta[n++] = *i;

   // Now start the loops
   XrdProofWorker *tmp = 0;
   bool notyet = 1;
   int jold = 0;
   while (notyet) {
      int j = jold;
      while (j < n - 1) {
         if (f(ta[j], ta[j+1]))
            break;
         j++;
      }
      if (j >= n - 1) {
         notyet = 0;
      } else {
         jold = j + 1;
         XPDSWAP(ta[j], ta[j+1], tmp);
         int k = j;
         while (k > 0) {
            if (!f(ta[k], ta[k-1])) {
               XPDSWAP(ta[k], ta[k-1], tmp);
            } else {
               break;
            }
            k--;
         }
      }
   }

   // Empty the original list
   XrdProofWorker *mst = lst->front();
   lst->clear();
   lst->push_back(mst);

   // Fill it again
   while (n--)
      lst->push_back(ta[n]);

   // Clean up
   delete[] ta;
}
