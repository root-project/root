// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
* Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#ifndef ROOT_TEveProjectionAxes
#define ROOT_TEveProjectionAxes

#include "TNamed.h"
#include "TAtt3D.h"
#include "TAttBBox.h"
#include "TAttAxis.h"

#include "TEveElement.h"

class TEveProjectionManager;

class TEveProjectionAxes : public TEveElement,
                           public TNamed,
                           public TAtt3D,
                           public TAttBBox,
                           public TAttAxis
{
   friend class TEveProjectionAxesGL;

public:
   enum ELabMode { kPosition, kValue };
   enum EAxesMode { kHorizontal, kVertical, kAll};

private:
   TEveProjectionAxes(const TEveProjectionAxes&);            // Not implemented
   TEveProjectionAxes& operator=(const TEveProjectionAxes&); // Not implemented

protected:
   TEveProjectionManager*  fManager;  // Model object.

   Bool_t  fUseColorSet;

   ELabMode  fLabMode;                // Division of distorted space.
   EAxesMode fAxesMode;               // Axis vertical/hotrizontal orientation.

   Bool_t fDrawCenter;           // Draw center of distortion.
   Bool_t fDrawOrigin;           // Draw origin.


public:
   TEveProjectionAxes(TEveProjectionManager* m, Bool_t useColorSet = kTRUE);
   virtual ~TEveProjectionAxes();

   TEveProjectionManager* GetManager()      { return fManager; }

   void            SetLabMode(ELabMode x)   { fLabMode = x; }
   ELabMode        GetLabMode()   const     { return fLabMode;}
   void            SetAxesMode(EAxesMode x) { fAxesMode = x; }
   EAxesMode       GetAxesMode()   const    { return fAxesMode; }

   void            SetDrawCenter(Bool_t x)   { fDrawCenter = x; }
   Bool_t          GetDrawCenter() const     { return fDrawCenter; }
   void            SetDrawOrigin(Bool_t x)   { fDrawOrigin = x; }
   Bool_t          GetDrawOrigin() const     { return fDrawOrigin; }

   virtual void    Paint(Option_t* option="");
   virtual void    ComputeBBox();

   virtual const   TGPicture* GetListTreeIcon(Bool_t open=kFALSE);

   ClassDef(TEveProjectionAxes, 1); // Class to draw scales in non-linear projections.
};

#endif
