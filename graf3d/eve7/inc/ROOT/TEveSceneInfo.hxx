// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveSceneInfo_hxx
#define ROOT_TEveSceneInfo_hxx

#include "ROOT/TEveElement.hxx"

namespace ROOT { namespace Experimental
{

class TEveViewer;
class TEveScene;

class TEveSceneInfo : public TEveElement,
                      public TNamed
{
private:
   TEveSceneInfo(const TEveSceneInfo&);            // Not implemented
   TEveSceneInfo& operator=(const TEveSceneInfo&); // Not implemented

protected:
   TEveViewer       *fViewer;
   TEveScene        *fScene;

public:
   TEveSceneInfo(TEveViewer* viewer, TEveScene* scene);
   virtual ~TEveSceneInfo() {}

   Int_t   WriteCoreJson(nlohmann::json& j, Int_t rnr_offset); // override;

   TEveViewer   * GetViewer()      const { return fViewer; }
   TEveScene    * GetScene()       const { return fScene;  }

   virtual Bool_t SingleRnrState() const { return kTRUE; }

   virtual void   AddStamp(UChar_t bits);

   virtual Bool_t AcceptElement(TEveElement* el);
   virtual Bool_t HandleElementPaste(TEveElement* el);

   ClassDef(TEveSceneInfo, 0); // Scene in a viewer.
};

}}

#endif
