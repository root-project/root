// @(#)root/proofd:$Id$
// Author: G. Ganis Jan 2008

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdConfig
#define ROOT_XrdProofdConfig

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdConfig                                                      //
//                                                                      //
// Author: G. Ganis, CERN, 2008                                         //
//                                                                      //
// Template to parse config directives                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XrdOuc/XrdOucHash.hh"

#include "XrdProofdAux.h"

class XrdSysError;

class XrdProofdConfig {

private:
   XrdProofdFile                  fCfgFile;       // Config file
   XrdOucHash<XrdProofdDirective> fDirectives;   // Registered directives

   static XrdOucString            fgHost;

protected:
   XrdSysError                   *fEDest;

   int    ParseFile(bool rcf = false);
   bool   ReadFile(bool update = true);
   void   Register(const char *dname, XrdProofdDirective *d)
                                            { fDirectives.Add(dname, d); }
public:
   XrdProofdConfig(const char *cfg = 0, XrdSysError *edest = 0);
   virtual ~XrdProofdConfig() { }

   void   SetCfgEDest(const char *cfg, XrdSysError *edest);
   const char *CfgFile() const { return fCfgFile.fName.c_str(); }

   virtual int  Config(bool rcf = 0) { return ParseFile(rcf); }
   virtual int  DoDirective(XrdProofdDirective *, char *,
                            XrdOucStream *, bool) { return 0; }
   virtual void RegisterDirectives() { }
};

#endif
