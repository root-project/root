// @(#)root/gl:$Name:  $:$Id: TGLViewer.h,v 1.6 2005/06/21 16:54:17 brun Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLViewer
#define ROOT_TGLViewer

#ifndef ROOT_TVirtualViewer3D
#include "TVirtualViewer3D.h"
#endif
#ifndef ROOT_TGLScene
#include "TGLScene.h"
#endif
#ifndef ROOT_TGLPerspectiveCamera
#include "TGLPerspectiveCamera.h"
#endif
#ifndef ROOT_TGLOrthoCamera
#include "TGLOrthoCamera.h"
#endif
#ifndef ROOT_TTimer
#include "TTimer.h"
#endif
#ifndef ROOT_CsgOps
#include "CsgOps.h"
#endif

#include <vector>

class TGLFaceSet;
class TGLRedrawTimer;

/*************************************************************************
 * TGLViewer - TODO
 *
 *
 *
 *************************************************************************/
class TGLViewer : public TVirtualViewer3D
{
public:
   enum ECamera { kPerspective, kXOY, kYOZ, kXOZ };

private:
   ///////////////////////////////////////////////////////////////////////
   // Fields
   ///////////////////////////////////////////////////////////////////////

   // Cameras
   // TODO: Put in vector and allow external creation
   TGLPerspectiveCamera fPerspectiveCamera;  //!
   TGLOrthoCamera       fOrthoXOYCamera;     //!
   TGLOrthoCamera       fOrthoYOZCamera;     //!
   TGLOrthoCamera       fOrthoXOZCamera;     //!
   TGLCamera          * fCurrentCamera;      //!

   // Scene management
   Bool_t            fInternalRebuild;       //! internal scene rebuild in progress?
   Bool_t            fAcceptedAllPhysicals;  //! did we take all physicals offered in AddObject()
   Bool_t            fInternalPIDs;          //! using internal physical IDs
   UInt_t            fNextInternalPID;       //! next internal physical ID (from 1 - 0 reserved)

   // Composite shape specific
   typedef std::pair<UInt_t, RootCsg::BaseMesh *> CSPart_t;
   mutable TGLFaceSet     *fComposite; //! Paritally created composite
   UInt_t                  fCSLevel;
   std::vector<CSPart_t>   fCSTokens;

   // Debug tracing (for scene rebuilds)
   UInt_t                  fAcceptedPhysicals;
   UInt_t                  fRejectedPhysicals;

   ///////////////////////////////////////////////////////////////////////
   // Methods
   ///////////////////////////////////////////////////////////////////////

   void PreDraw();
   void PostDraw();

   // Scene management
   Int_t              ValidateObjectBuffer(const TBuffer3D & buffer, Bool_t logical) const;
   TGLLogicalShape  * CreateNewLogical(const TBuffer3D & buffer) const;
   TGLPhysicalShape * CreateNewPhysical(UInt_t physicalID, const TBuffer3D & buffer, 
                                        const TGLLogicalShape & logical) const;
   RootCsg::BaseMesh *BuildComposite();

   // Non-copyable class
   TGLViewer(const TGLViewer &);
   TGLViewer & operator=(const TGLViewer &);

protected:
   ///////////////////////////////////////////////////////////////////////
   // Fields
   ///////////////////////////////////////////////////////////////////////
   // Move back to private when gVirtualGL removed
   TGLRedrawTimer     * fRedrawTimer;        //!
   UInt_t               fNextSceneLOD;       //!

   // Scene is created/owned internally.
   // In future it will be shaped between multiple viewers
   TGLScene       fScene;          //! the GL scene - owned by viewer at present
   TGLRect        fViewport;       //! viewport - drawn area
   TGLPlane       fClipPlane;      //! current clip plane
   Bool_t         fUseClipPlane;   //! use current clipping plane
   Bool_t         fDrawAxes;       //! draw scene axes
   Bool_t         fInitGL;         //! has GL been initialised?
   Bool_t         fDebugMode;      //! viewer in debug mode (forced rebuild + draw scene/frustum/interest boxes)

   ///////////////////////////////////////////////////////////////////////
   // Methods
   ///////////////////////////////////////////////////////////////////////
   
   // Concrete class must implement - TGLManager will replace most
   // fPad call in FillScene
   virtual void   InitGL()                            = 0;
   virtual void   MakeCurrent()  const                = 0;
   virtual void   SwapBuffers()  const                = 0;
   virtual void   FillScene()                         = 0;

   // Scene management
   Bool_t         RebuildScene();

   // Viewport and Camera
   void         SetViewport(Int_t x, Int_t y, UInt_t width, UInt_t height);
   void         SetupCameras(const TGLBoundingBox & box);
   void         SetCurrentCamera(ECamera camera);
   TGLCamera &  CurrentCamera() const { return *fCurrentCamera; }

   // Coordinate conversion
   void WindowToGL(TGLRect & rect)      const { rect.Y() = fViewport.Height() - rect.Y(); }
   void WindowToGL(TGLVertex3 & vertex) const { vertex.Y() = fViewport.Height() - vertex.Y(); }

public:
   TGLViewer();
   virtual ~TGLViewer();

   // TVirtualViewer3D interface
   virtual Bool_t PreferLocalFrame() const;
   virtual void   BeginScene();
   virtual Bool_t BuildingScene() const { return fScene.CurrentLock() == TGLScene::kModifyLock; }
   virtual void   EndScene();
   virtual Int_t  AddObject(const TBuffer3D & buffer, Bool_t * addChildren = 0);
   virtual Int_t  AddObject(UInt_t physicalID, const TBuffer3D & buffer, Bool_t * addChildren = 0);
   virtual Bool_t OpenComposite(const TBuffer3D & buffer, Bool_t * addChildren = 0);
   virtual void   CloseComposite();
   virtual void   AddCompositeOp(UInt_t operation);

   // Once TVirtualGL dropped these can move back to protected
   void   Draw();
   Bool_t Select(const TGLRect & rect); // Window coords origin top left

   // TODO: Once better solution to TGLRedrawTimer found make this
   // protected again.
   virtual void Invalidate(UInt_t redrawLOD = kMed) = 0;

   ClassDef(TGLViewer,0) // GL viewer generic base class
};

// TODO: Find a better place/way to do this
class TGLRedrawTimer : public TTimer
{
   private:
      TGLViewer & fViewer;
      UInt_t      fRedrawLOD;
   public:
      TGLRedrawTimer(TGLViewer & viewer) : fViewer(viewer), fRedrawLOD(100) {};
      ~TGLRedrawTimer() {};
      void   RequestDraw(Int_t milliSec, UInt_t redrawLOD) {
         fRedrawLOD = redrawLOD;
         TTimer::Start(milliSec, kTRUE);
      }
      Bool_t Notify() { TurnOff(); fViewer.Invalidate(kHigh); return kTRUE; }
};

#endif // ROOT_TGLViewer
