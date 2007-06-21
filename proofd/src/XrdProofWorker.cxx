// @(#)root/proofd:$Name:  $:$Id:$
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
#include "XrdProofServProxy.h"
#include "XrdNet/XrdNetDNS.hh"

// Tracing utilities
#include "XrdProofdTrace.h"
static const char *gTraceID = " ";
extern XrdOucTrace *XrdProofdTrace;
#define TRACEID gTraceID

//__________________________________________________________________________
XrdProofWorker::XrdProofWorker(const char *str)
               : fActive (0), fSuspended(0),
                 fExport(256), fType('W'), fPort(-1), fPerfIdx(100)
{
   // Constructor from a config file-like string

   // Make sure we got something to parse
   if (!str || strlen(str) <= 0)
      return;

   // The actual work is done by Reset()
   Reset(str);
}

//__________________________________________________________________________
void XrdProofWorker::Reset(const char *str)
{
   // Set content from a config file-like string

   // Reinit vars
   fActive = 0;
   fSuspended = 0;
   fExport = "";
   fType = 'W';
   fHost = "";
   fPort = -1;
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

   // Next token is the user@host string
   if ((from = s.tokenize(tok, from, ' ')) == STR_NPOS)
      return;
   fHost = tok;

   // and then the remaining options
   while ((from = s.tokenize(tok, from, ' ')) != STR_NPOS) {
      if (tok.beginswith("workdir=")) {
         // Working dir
         tok.replace("workdir=","");
         fWorkDir = tok;
      } else if (tok.beginswith("image=")) {
         // Image
         tok.replace("image=","");
         fImage = tok;
      } else if (tok.beginswith("msd=")) {
         // Mass storage domain
         tok.replace("msd=","");
         fMsd = tok;
      } else if (tok.beginswith("port=")) {
         // Port
         tok.replace("port=","");
         fPort = strtol(tok.c_str(), (char **)0, 10);
      } else if (tok.beginswith("perf=")) {
         // Performance index
         tok.replace("perf=","");
         fPerfIdx = strtol(tok.c_str(), (char **)0, 10);
      } else {
         // Unknown
         TRACE(XERR, "XrdProofWorker::Reset: unknown option "<<tok);
      }
   }

   // Default image is the host name
   if (fImage.length() <= 0)
      fImage.assign(fHost, fHost.find('@')+1);
}

//__________________________________________________________________________
bool XrdProofWorker::Matches(const char *host)
{
   // Check compatibility of host with this instance.
   // return 1 if compatible.

   XrdOucString thishost;
   thishost.assign(fHost, fHost.find('@'));
   char *h = XrdNetDNS::getHostName(thishost.c_str());
   thishost = (h ? h : "");
   SafeFree(h);

   return ((thishost.matches(host)) ? 1 : 0);
}

//__________________________________________________________________________
const char *XrdProofWorker::Export()
{
   // Export current content in a form understood by parsing algorithms
   // inside the PROOF session, i.e.
   // <type>|<host@user>|<port>|-|-|<perfidx>|<img>|<workdir>|<msd>

   fExport = fType;

   // Add user@host
   fExport += '|' ; fExport += fHost;

   // Add port
   if (fPort > 0) {
      fExport += '|' ; fExport += fPort;
   } else
      fExport += "|-";

   // No ordinal and ID at this level
   fExport += "|-|-";

   // Add performance index
   fExport += '|' ; fExport += fPerfIdx;

   // Add image
   if (fImage.length() > 0) {
      fExport += '|' ; fExport += fImage;
   } else
      fExport += "|-";

   // Add workdir
   if (fWorkDir.length() > 0) {
      fExport += '|' ; fExport += fWorkDir;
   } else
      fExport += "|-";

   // Add mass storage domain
   if (fMsd.length() > 0) {
      fExport += '|' ; fExport += fMsd;
   } else
      fExport += "|-";

   // We are done
   TRACE(DBG, "XrdProofWorker::Export: sending: "<<fExport);
   return fExport.c_str();
}

//__________________________________________________________________________
int XrdProofWorker::GetNActiveSessions()
{
   // calculate the number of workers existing on this node which are
   // currently running.
   // TODO: optimally, one could contact the packetizer and count the
   // opened files.
   int myRunning = 0;
   std::list<XrdProofServProxy *>::iterator iter;
   for (iter = fProofServs.begin(); iter != fProofServs.end(); ++iter) {
      if (*iter) {
         if ((*iter)->Status() == kXPD_running);
            myRunning++;
      }
   }
   return myRunning;
}
//__________________________________________________________________________
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
   for ( ; i != lst->end(); ++i)
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
