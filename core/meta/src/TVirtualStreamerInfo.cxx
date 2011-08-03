// @(#)root/base:$Id$
// Author: Rene Brun   05/02/2007
/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualStreamerInfo   Abstract Interface class                      //
//                                                                      //
// Abstract Interface describing Streamer information for one class.    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TSystem.h"
#include "TClass.h"
#include "TVirtualStreamerInfo.h"
#include "TPluginManager.h"
#include "TStreamerElement.h"
#include "TError.h"


TVirtualStreamerInfo *TVirtualStreamerInfo::fgInfoFactory    = 0;

Bool_t  TVirtualStreamerInfo::fgCanDelete        = kTRUE;
Bool_t  TVirtualStreamerInfo::fgOptimize         = kTRUE;
Bool_t  TVirtualStreamerInfo::fgStreamMemberWise = kTRUE;

ClassImp(TVirtualStreamerInfo)

//______________________________________________________________________________
TVirtualStreamerInfo::TVirtualStreamerInfo() : fOptimized(kFALSE), fIsBuilt(kFALSE)
{
   // Default constructor.

}

//______________________________________________________________________________
TVirtualStreamerInfo::TVirtualStreamerInfo(TClass *cl)
   : TNamed(cl->GetName(),""), fOptimized(kFALSE), fIsBuilt(kFALSE)
{
   // Default constructor.

}

//______________________________________________________________________________
TVirtualStreamerInfo::TVirtualStreamerInfo(const TVirtualStreamerInfo& info)
  : TNamed(info), fOptimized(kFALSE), fIsBuilt(kFALSE)
{ 
   //copy constructor
}

//______________________________________________________________________________
TVirtualStreamerInfo& TVirtualStreamerInfo::operator=(const TVirtualStreamerInfo& info)
{
   //assignment operator
   if(this!=&info) {
      TNamed::operator=(info);
   } 
   return *this;
}

//______________________________________________________________________________
TVirtualStreamerInfo::~TVirtualStreamerInfo()
{
   // Destructor

}

//______________________________________________________________________________
Bool_t TVirtualStreamerInfo::CanDelete()
{
   // static function returning true if ReadBuffer can delete object
   return fgCanDelete;
}

//______________________________________________________________________________
Bool_t TVirtualStreamerInfo::CanOptimize()
{
   // static function returning true if optimization can be on
   return fgOptimize;
}

//______________________________________________________________________________
TStreamerBasicType *TVirtualStreamerInfo::GetElementCounter(const char *countName, TClass *cl)
{
   // Get pointer to a TStreamerBasicType in TClass *cl
   //static function

   TObjArray *sinfos = cl->GetStreamerInfos();
   TVirtualStreamerInfo *info = (TVirtualStreamerInfo *)sinfos->At(cl->GetClassVersion());

   if (!info || !info->IsBuilt()) {
      // Even if the streamerInfo exist, it could still need to be 'build'
      // It is important to figure this out, because
      //   a) if it is not build, we need to build
      //   b) if is build, we should not build it (or we could end up in an
      //      infinite loop, if the element and its counter are in the same
      //      class!

      info = cl->GetStreamerInfo();
   }
   if (!info) return 0;
   TStreamerElement *element = (TStreamerElement *)info->GetElements()->FindObject(countName);
   if (!element) return 0;
   if (element->IsA() == TStreamerBasicType::Class()) return (TStreamerBasicType*)element;
   return 0;
}

//______________________________________________________________________________
Bool_t TVirtualStreamerInfo::GetStreamMemberWise()
{
   // Return whether the TStreamerInfos will save the collections in
   // "member-wise" order whenever possible.    The default is to store member-wise.
   // kTRUE indicates member-wise storing
   // kFALSE inddicates object-wise storing
   //
   // A collection can be saved member wise when it contain is guaranteed to be
   // homogeneous.  For example std::vector<THit> can be stored member wise,
   // while std::vector<THit*> can not (possible use of polymorphism).

   return fgStreamMemberWise;
}

//______________________________________________________________________________
void TVirtualStreamerInfo::Optimize(Bool_t opt)
{
   //  This is a static function.
   //  Set optimization option.
   //  When this option is activated (default), consecutive data members
   //  of the same type are merged into an array (faster).
   //  Optimization must be off in TTree split mode.

   fgOptimize = opt;
}

//______________________________________________________________________________
TVirtualStreamerInfo *TVirtualStreamerInfo::Factory()
{
   // Static function returning a pointer to a new TVirtualStreamerInfo object.
   // If the Info factory does not exist, it is created via the plugin manager.
   // In reality the factory is an empty TStreamerInfo object.

   if (!fgInfoFactory) {
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualStreamerInfo","TStreamerInfo"))) {
         if (h->LoadPlugin() == -1)
            return 0;
         fgInfoFactory = (TVirtualStreamerInfo*) h->ExecPlugin(0);
      } else {
         TString filename("$ROOTSYS/etc/plugins/TVirtualStreamerInfo");
         gSystem->ExpandPathName(filename);
         if (gSystem->AccessPathName(filename) > 0) {            
            ::Fatal("TVirtualStreamerInfo::Factory",
                    "Cannot find the plugin handlers for TVirtualStreamerInfo! "
                    "$ROOTSYS/etc/plugins/TVirtualStreamerInfo does not exist "
                    "or is inaccessible.");
         } else {
            ::Fatal("TVirtualStreamerInfo::Factory",
                    "Cannot find the plugin handler for TVirtualStreamerInfo! "
                    "However $ROOTSYS/etc/plugins/TVirtualStreamerInfo is accessible, "
                    "Check the content of this directory!");
         }
      }
   }

   if (fgInfoFactory) return fgInfoFactory;
   return 0;
}

//______________________________________________________________________________
void TVirtualStreamerInfo::SetCanDelete(Bool_t opt)
{
   //  This is a static function.
   //  Set object delete option.
   //  When this option is activated (default), ReadBuffer automatically
   //  delete objects when a data member is a pointer to an object.
   //  If your constructor is not presetting pointers to 0, you must
   //  call this static function TStreamerInfo::SetCanDelete(kFALSE);

   fgCanDelete = opt;
}

//______________________________________________________________________________
void TVirtualStreamerInfo::SetFactory(TVirtualStreamerInfo *factory)
{
   //static function: Set the StreamerInfo factory
   fgInfoFactory = factory;
}

//______________________________________________________________________________
Bool_t TVirtualStreamerInfo::SetStreamMemberWise(Bool_t enable)
{
   // Set whether the TStreamerInfos will save the collections in
   // "member-wise" order whenever possible.  The default is to store member-wise.
   // kTRUE indicates member-wise storing
   // kFALSE inddicates object-wise storing
   // This function returns the previous value of fgStreamMemberWise.

   // A collection can be saved member wise when it contain is guaranteed to be
   // homogeneous.  For example std::vector<THit> can be stored member wise,
   // while std::vector<THit*> can not (possible use of polymorphism).

   Bool_t prev = fgStreamMemberWise;
   fgStreamMemberWise = enable;
   return prev;
}

//______________________________________________________________________________
void TVirtualStreamerInfo::Streamer(TBuffer &R__b)
{
   // Stream an object of class TVirtualStreamerInfo.

   TNamed::Streamer(R__b);
}
