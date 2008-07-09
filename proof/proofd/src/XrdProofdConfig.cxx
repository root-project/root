// @(#)root/proofd:$Id$
// Author: G. Ganis Jan 2008

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdConfig                                                      //
//                                                                      //
// Author: G. Ganis, CERN, 2008                                         //
//                                                                      //
// Implementation of the XrdProofdManager operations related to         //
// configuration.                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include "XrdProofdPlatform.h"

#ifdef OLDXRDOUC
#  include "XrdOuc/XrdOucError.hh"
#  include "XrdOuc/XrdOucLogger.hh"
#else
#  include "XrdSys/XrdSysError.hh"
#  include "XrdSys/XrdSysLogger.hh"
#endif
#include "XrdNet/XrdNetDNS.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucString.hh"

#include "XrdProofdConfig.h"

// Tracing utilities
#include "XrdProofdTrace.h"

XrdOucString XrdProofdConfig::fgHost;

//__________________________________________________________________________
XrdProofdConfig::XrdProofdConfig(const char *fn, XrdSysError *edest)
                : fCfgFile(fn), fEDest(edest)
{
   // Main constructor

   SetCfgEDest(fn, edest);
}

//__________________________________________________________________________
void XrdProofdConfig::SetCfgEDest(const char *fn, XrdSysError *edest)
{
   // Set config file and error handler

   fEDest = edest;
   if (fn && fCfgFile.fName != fn) {
      fCfgFile.fName = fn;
      XrdProofdAux::Expand(fCfgFile.fName);
   }
   fCfgFile.fMtime = 0;
}

//__________________________________________________________________________
bool XrdProofdConfig::ReadFile()
{
   // Return true if the file has never been read or did change since last
   // reading, false otherwise.
   XPDLOC(ALL, "Config::ReadFile")

   // Must have a file
   if (fCfgFile.fName.length() <= 0) {
      TRACE(XERR, "no config file!");
      return -1;
   }

   // Get the modification time
   struct stat st;
   if (stat(fCfgFile.fName.c_str(), &st) != 0)
      return -1;
   TRACE(DBG, "file: " << fCfgFile.fName);
   TRACE(DBG, "time of last modification: " << st.st_mtime);

   // File should be loaded only once
   if (st.st_mtime <= fCfgFile.fMtime)
      return 0;

   // Save the modification time
   fCfgFile.fMtime = st.st_mtime;

   // Never read or changed: read it again
   return 1;
}

//__________________________________________________________________________
int XrdProofdConfig::ParseFile(bool rcf)
{
   // Parse config file for the registered directives. The flag 'rcf' is 0
   // on the first call, 1 on successive calls.
   // Returns 0 on success, -1 otherwise
   XPDLOC(ALL, "Config::ParseFile")

   XrdOucString mp;

   // Check if the config file changed since last read, if any
   if (!ReadFile()) {
      TRACE(DBG, "config file already parsed ");
      return 0;
   }

   // Local FQDN
   if (fgHost.length() <= 0) {
      char *host = XrdNetDNS::getHostName();
      fgHost = host ? host : "";
      SafeFree(host);
   }

   // Communicate the host name to the config directives, so that the (deprecated)
   // old style 'if' condition can be handled
   fDirectives.Apply(SetHostInDirectives, (void *)fgHost.c_str());

   // Open the config file
   int cfgFD;
   const char *cfn = fCfgFile.fName.c_str();
   if ((cfgFD = open(cfn, O_RDONLY, 0)) < 0) {
      TRACE(XERR, "unable to open : " << cfn);
      return -1;
   }

   // Create the stream and attach to the file
   XrdOucStream cfg(fEDest, getenv("XRDINSTANCE"));
   cfg.Attach(cfgFD);

   // Process items
   char *var = 0, *val = 0;
   while ((var = cfg.GetMyFirstWord())) {
      if (!(strncmp("xpd.", var, 4)) && var[4]) {
         // xpd directive: process it
         var += 4;
         // Get the value
         val = cfg.GetToken();
         // Get the directive
         XrdProofdDirective *d = fDirectives.Find(var);
         if (d)
            // Process it
            d->DoDirective(val, &cfg, rcf);
      } else if (var[0]) {
         // Check if we are interested in this non-xpd directive
         XrdProofdDirective *d = fDirectives.Find(var);
         if (d)
            // Process it
            d->DoDirective(0, &cfg, rcf);
      }
   }

   // Done
   return 0;
}
