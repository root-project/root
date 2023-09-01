// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveSceneInfo
#define ROOT_TEveSceneInfo

#include "TEveElement.h"

class TGLSceneBase;
class TGLSceneInfo;

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
   TGLSceneInfo     *fGLSceneInfo;

public:
   TEveSceneInfo(TEveViewer* viewer, TEveScene* scene, TGLSceneInfo* sinfo);
   ~TEveSceneInfo() override {}

   TEveViewer   * GetViewer()      const { return fViewer; }
   TEveScene    * GetScene()       const { return fScene;  }
   TGLSceneInfo * GetGLSceneInfo() const { return fGLSceneInfo; }
   TGLSceneBase * GetGLScene()     const;

   Bool_t SingleRnrState() const override { return kTRUE; }

   void   AddStamp(UChar_t bits) override;

   Bool_t AcceptElement(TEveElement* el) override;
   Bool_t HandleElementPaste(TEveElement* el) override;

   ClassDefOverride(TEveSceneInfo, 0); // TEveUtil representation of TGLSceneInfo.
};

#endif
