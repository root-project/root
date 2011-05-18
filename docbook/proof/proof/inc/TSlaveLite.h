// @(#)root/proof:$Id$
// Author: G. Ganis Mar 2008

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSlaveLite
#define ROOT_TSlaveLite


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSlaveLite                                                           //
//                                                                      //
// This is the version of TSlave for local worker servers.              //
// See TSlave for details.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TSlave
#include "TSlave.h"
#endif

class TObjString;
class TSocket;
class TSignalHandler;

class TSlaveLite : public TSlave {

friend class TProof;

private:
   Bool_t   fValid;
   TSignalHandler *fIntHandler;     //interrupt signal handler (ctrl-c)

   void  Init();

public:
   TSlaveLite(const char *ord, Int_t perf,
              const char *image, TProof *proof, Int_t stype,
              const char *workdir, const char *msd);
   virtual ~TSlaveLite();

   void   Close(Option_t *opt = "");
   void   DoError(int level, const char *location, const char *fmt,
                  va_list va) const;

   void   Print(Option_t *option="") const;
   Int_t  SetupServ(Int_t stype, const char *conffile);

   ClassDef(TSlaveLite, 0)  //PROOF lite worker server
};

#endif
