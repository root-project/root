// @(#)root/r:$Id$
// Author: Omar Zapata   11/06/2014


/*************************************************************************
 * Copyright (C)  2014, Omar Andres Zapata Mesa                          *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_R_TRSYSTEM
#define ROOT_R_TRSYSTEM

#ifndef ROOT_R_RExports
#include<RExports.h>
#endif

#ifndef ROOT_TSystem
#include<TSystem.h>
#endif

#ifndef ROOT_TThread
#include<TThread.h>
#endif

#ifndef ROOT_TApplication
#include<TApplication.h>
#endif

//________________________________________________________________________________________________________
/**
   This is a gSystem wrap for R


   @ingroup R
*/
namespace ROOT {
   namespace R {

      class TRSystem: public TObject {
      private:
         TThread *th;
      public:
         TRSystem();
         ~TRSystem() {
            if (th) delete th;
         }
         void ProcessEventsLoop();
         Int_t   Load(TString module);
      };
   }
}

ROOTR_EXPOSED_CLASS_INTERNAL(TRSystem)


//______________________________________________________________________________
ROOT::R::TRSystem::TRSystem(): TObject()
{
   th = nullptr;
}

//______________________________________________________________________________
void ROOT::R::TRSystem::ProcessEventsLoop()
{
   if (!gApplication) {
      Error("TRSystem", "Running ProcessEventsLoop without global object gApplication.");
      return;
   }
   th = new TThread([](void * args) {
      while (kTRUE) {
         gSystem->ProcessEvents();
         gSystem->Sleep(100);
      }
   }, (void *)this);
   th->Run();
}

//______________________________________________________________________________
Int_t ROOT::R::TRSystem::Load(TString module)
{
   return gSystem->Load(module.Data());
}

ROOTR_MODULE(ROOTR_TRSystem)
{

   ROOT::R::class_<ROOT::R::TRSystem>("TRSystem", "TSystem class to manipulate ROOT's Process.")
   .constructor()
   .method("ProcessEventsLoop", &ROOT::R::TRSystem::ProcessEventsLoop)
   .method("Load", (Int_t(ROOT::R::TRSystem::*)(TString))&ROOT::R::TRSystem::Load)
   ;
}

#endif
