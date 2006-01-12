// @(#)root/gl:$Name:  $:$Id: TGLPixmap.h,v 1.3 2005/11/23 14:48:02 couet Exp $
// Author: Timur Pocheptsov 18/08/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLPixmap
#define ROOT_TGLPixmap

#include <utility>

#ifndef ROOT_TVirtualViewer3D
#include "TVirtualViewer3D.h"
#endif
#ifndef ROOT_TArcBall
#include "TArcBall.h"
#endif
#ifndef ROOT_TPoint
#include "TPoint.h"
#endif

class GLSelection;
class GLSceneObject;
class GLCamera;
class TBuffer3D;
class TVirtualPad;
class TGLRender;

class TGLPixmap : public TVirtualViewer3D {
private:
   GLCamera         *fCamera;
   Double_t          fViewVolume[4];
   Double_t          fZoom[4];
   Int_t             fActiveViewport[4];
   Int_t             fLightMask;
   TGLRender         *fRender;

   typedef std::pair<Double_t, Double_t> PDD_t;
   PDD_t             fRangeX, fRangeY, fRangeZ, fLastPosRot;
   Double_t          fXc, fYc, fZc;
   Double_t          fRad;

   Bool_t            fPressed;
   TArcBall          fArcBall;

   UInt_t            fNbShapes;
   TPoint            fLastPos;

   Int_t             fGLDevice;

   GLSceneObject    *fSelectedObj;
   enum EAction{kNoAction, kRotating, kPicking, kZooming};
   EAction fAction;
   Bool_t            fBuildingScene;
   TVirtualPad      *fPad;
   Bool_t            fFirstScene;

public:
   TGLPixmap(TVirtualPad * pad);
   ~TGLPixmap();

   // TVirtualViewer3D interface
   virtual Bool_t PreferLocalFrame() const;
   virtual void   BeginScene();
   virtual Bool_t BuildingScene() const { return fBuildingScene; }
   virtual void   EndScene();
   Int_t          AddObject(const TBuffer3D &buffer, Bool_t *addChild);
   Int_t          AddObject(UInt_t, const TBuffer3D &, Bool_t *) {return 0;}
   virtual Bool_t   OpenComposite(const TBuffer3D & buffer, Bool_t * addChildren = 0);
   virtual void   CloseComposite();
   virtual void   AddCompositeOp(UInt_t operation);
   TObject *SelectObject(Int_t x, Int_t y);
   Int_t          DistancetoPrimitive(Int_t px, Int_t py);

/////////////////////////////////////////
   void ZoomIn();// *MENU*
   void ZoomOut();// *MENU*
   void PrintObjects();// *MENU*

   void ExecuteEvent(Int_t type, Int_t px, Int_t py);
   void DrawViewer();
private:
   void CreateViewer();
   void DrawObjects()const;
   void MakeCurrent()const;
   void SwapBuffers()const;
   void UpdateRange(const GLSelection *box);
   void CalculateViewports();
   void CalculateViewvolumes();
   void CreateCameras();

   //non-copyable class
   TGLPixmap(const TGLPixmap &);
   TGLPixmap & operator = (const TGLPixmap &);

   ClassDef(TGLPixmap, 0)
};

#endif
