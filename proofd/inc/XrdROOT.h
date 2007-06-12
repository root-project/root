// @(#)root/proofd:$Name:  $:$Id:$
// Author: G. Ganis  June 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdROOT
#define ROOT_XrdROOT

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdProtocol                                                    //
//                                                                      //
// Authors: G. Ganis, CERN, 2007                                        //
//                                                                      //
// Class describing a ROOT version                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include "Xrd/XrdProtocol.hh"
#include "XProofProtocol.h"
#include "XrdOuc/XrdOucString.hh"

class XrdROOT {
private:
   int          fStatus;
   XrdOucString fDir;
   XrdOucString fTag;
   XrdOucString fExport;
   XrdOucString fPrgmSrv;
   kXR_int16    fSrvProtVers;

   int          GetROOTVersion(const char *dir, XrdOucString &version);
   int          ValidatePrgmSrv();

public:
   XrdROOT(const char *dir, const char *tag);
   ~XrdROOT() { }

   const char *Dir() const { return fDir.c_str(); }
   const char *Export() const { return fExport.c_str(); }
   bool        IsValid() const { return ((fStatus == 1) ? 1: 0); }
   bool        IsInvalid() const { return ((fStatus == -1) ? 1: 0); }
   bool        Match(const char *dir, const char *tag)
                          { return ((fTag == tag && fDir == dir) ? 1 : 0); }
   bool        MatchTag(const char *tag) { return ((fTag == tag) ? 1 : 0); }
   const char *PrgmSrv() const { return fPrgmSrv.c_str(); }
   void        SetValid() { fStatus = 1; }
   kXR_int16   SrvProtVers() const { return fSrvProtVers; }
   const char *Tag() const { return fTag.c_str(); }
   bool        Validate();
};

#endif
