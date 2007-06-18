// @(#)root/gl:$Name:  $:$Id: TGLViewer.h,v 1.37 2007/06/11 19:56:33 brun Exp $
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

#include "TGLViewerBase.h"
#include "TGLRnrCtx.h"
#include "TGLSelectRecord.h"

#include "TVirtualViewer3D.h"

#include "TGLScene.h"
#include "TGLPerspectiveCamera.h"
#include "TGLOrthoCamera.h"

#include "TTimer.h"
#include "TPoint.h"

#include "CsgOps.h"

#include "GuiTypes.h"
#include "RQ_OBJECT.h"

#include <vector>

class TGLFaceSet;
class TGLRedrawTimer;
class TGLViewerEditor;
//class TGLWindow; // Remove - TGLManager
class TGLWidget;
class TGLLightSet;
class TGLClipSet;
class TGLManipSet;
class TGLCameraMarkupStyle;

class TContextMenu;


class TGLViewer : public TVirtualViewer3D,
                  public TGLViewerBase

{
   RQ_OBJECT("TGLViewer")
   friend class TGLOutput;
public:

   enum ECameraType { kCameraPerspXOZ, kCameraPerspYOZ, kCameraPerspXOY,
                      kCameraOrthoXOY, kCameraOrthoXOZ, kCameraOrthoZOY };

private:
   TGLViewer(const TGLViewer &);             // Not implemented
   TGLViewer & operator=(const TGLViewer &); // Not implemented

   void InitSecondaryObjects();

protected:
   // External handles
   TVirtualPad  * fPad;         //! external pad - remove replace with signal

   // GUI Handles
   TContextMenu * fContextMenu; //!

   // Cameras
   // TODO: Put in vector and allow external creation
   TGLPerspectiveCamera fPerspectiveCameraXOZ; //!
   TGLPerspectiveCamera fPerspectiveCameraYOZ; //!
   TGLPerspectiveCamera fPerspectiveCameraXOY; //!
   TGLOrthoCamera       fOrthoXOYCamera;       //!
   TGLOrthoCamera       fOrthoXOZCamera;       //!
   TGLOrthoCamera       fOrthoZOYCamera;       //!
   TGLCamera          * fCurrentCamera;        //!

   // Lights
   TGLLightSet        * fLightSet;             //!
   // Clipping
   TGLClipSet         * fClipSet;              //!
   // Selected physical
   TGLSelectRecord      fCurrentSelRec;        //! select record in use as selected
   TGLSelectRecord      fSelRec;               //! select record from last select (should go to context)
   TGLSelectRecord      fSecSelRec;            //! select record from last secondary select (should go to context)
   TGLManipSet        * fSelectedPShapeRef;    //!
   // Overlay
   TGLOverlayElement  * fCurrentOvlElm;        //! current overlay element
   TGLOvlSelectRecord   fOvlSelRec;            //! select record from last overlay select

   // Scene management for fScene
   Bool_t            fInternalRebuild;       //! scene rebuild triggered internally/externally?
   Bool_t            fPostSceneBuildSetup;   //! setup viewer after (re)build complete?
   Bool_t            fAcceptedAllPhysicals;  //! did we take all physicals offered in AddObject()
   Bool_t            fForceAcceptAll;        //! force taking of all logicals/physicals in AddObject()
   Bool_t            fInternalPIDs;          //! using internal physical IDs
   UInt_t            fNextInternalPID;       //! next internal physical ID (from 1 - 0 reserved)

   // Composite shape specific - to TGLPadScene or helper object?
   typedef std::pair<UInt_t, RootCsg::TBaseMesh *> CSPart_t;
   mutable TGLFaceSet     *fComposite; //! Paritally created composite
   UInt_t                  fCSLevel;
   std::vector<CSPart_t>   fCSTokens;

   // Mouse ineraction
   enum EDragAction   { kDragNone,
                        kDragCameraRotate, kDragCameraTruck, kDragCameraDolly,
                        kDragOverlay };

   EDragAction          fAction;
   TPoint               fLastPos;
   UInt_t               fActiveButtonID;

   // Redraw timer
   TGLRedrawTimer     * fRedrawTimer; //!

   // Scene is created/owned internally.
   // In future it will be shared between multiple viewers
   TGLScene       fScene;          //! the default GL scene (filled via VirtualViewer3D API)
   TGLRect        fViewport;       //! viewport - drawn area
   Color_t        fClearColor;     //! clear-color
   Int_t          fAxesType;       //! axes type
   Bool_t         fReferenceOn;    //! reference marker on?
   TGLVertex3     fReferencePos;   //! reference position
   TGLCameraMarkupStyle * fCameraMarkup; //! markup size of viewport in scene units

   Bool_t         fInitGL;         //! has GL been initialised?
   Bool_t         fSmartRefresh;   //! cache logicals during scene rebuilds, use TAtt3D time-stamp to determine if they are still valid

   // Debug tracing (for scene rebuilds)
   Bool_t         fDebugMode;             //! debug mode (forced rebuild + draw scene/frustum/interest boxes)
   UInt_t         fAcceptedPhysicals;     //! number of physicals accepted in last rebuild
   UInt_t         fRejectedPhysicals;     //! number of physicals rejected in last rebuild
   Bool_t         fIsPrinting;

   ///////////////////////////////////////////////////////////////////////
   // Methods
   ///////////////////////////////////////////////////////////////////////
   // Drawing - can tidy up/remove lots when TGLManager added
   void InitGL();
   void PreDraw();
   void PostDraw();
   void MakeCurrent() const;
   void SwapBuffers() const;

   // Scene management - to TGLPadScene or helper object?
   Bool_t             RebuildScene();
   Int_t              ValidateObjectBuffer(const TBuffer3D & buffer, Bool_t includeRaw) const;
   TGLLogicalShape  * CreateNewLogical(const TBuffer3D & buffer) const;
   TGLPhysicalShape * CreateNewPhysical(UInt_t physicalID, const TBuffer3D & buffer,
                                        const TGLLogicalShape & logical) const;
   RootCsg::TBaseMesh *BuildComposite();

   // Cameras
   void        SetViewport(Int_t x, Int_t y, Int_t width, Int_t height);
   void        SetupCameras(Bool_t reset);

protected:
   TGLWidget         *fGLWindow;
   Int_t              fGLDevice; //!for embedded gl viewer

   TGLViewerEditor *fPadEditor;

   std::map<TClass*, TClass*> fDirectRendererMap; //!
   TClass*          FindDirectRendererClass(TClass* cls);
   TGLLogicalShape* AttemptDirectRenderer(TObject* id);

   // Updata/camera-reset behaviour
   Bool_t           fIgnoreSizesOnUpdate;      // ignore sizes of bounding-boxes on update
   Bool_t           fResetCamerasOnUpdate;     // reposition camera on each update
   Bool_t           fResetCamerasOnNextUpdate; // reposition camera on next update
   Bool_t           fResetCameraOnDoubleClick; // reposition camera on double-click

public:
   TGLViewer(TVirtualPad * pad, Int_t x, Int_t y, Int_t width, Int_t height);
   TGLViewer(TVirtualPad * pad);
   virtual ~TGLViewer();

   // TVirtualViewer3D interface
   virtual Int_t  DistancetoPrimitive(Int_t px, Int_t py);
   virtual void   ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual Bool_t PreferLocalFrame() const;

   virtual void   BeginScene();
   virtual Bool_t BuildingScene() const { return fScene.CurrentLock() == kModifyLock; }
   virtual void   EndScene();

   virtual Int_t  AddObject(const TBuffer3D & buffer, Bool_t * addChildren = 0);
   virtual Int_t  AddObject(UInt_t physicalID, const TBuffer3D & buffer, Bool_t * addChildren = 0);
   virtual Bool_t OpenComposite(const TBuffer3D & buffer, Bool_t * addChildren = 0);
   virtual void   CloseComposite();
   virtual void   AddCompositeOp(UInt_t operation);

   virtual void   PrintObjects();
   virtual void   ResetCameras()                { SetupCameras(kTRUE); }
   virtual void   ResetCamerasAfterNextUpdate() { fResetCamerasOnNextUpdate = kTRUE; }

   virtual void   RefreshPadEditor(TObject* = 0) {}

   Int_t   GetDev()          const           { return fGLDevice; }
   Color_t GetClearColor()   const           { return fClearColor; }
   void    SetClearColor(Color_t col)        { fClearColor = col; }
   Bool_t  GetSmartRefresh() const           { return fSmartRefresh; }
   void    SetSmartRefresh(Bool_t smart_ref) { fSmartRefresh = smart_ref; }

   TGLLightSet* GetLightSet() const { return fLightSet; }
   TGLClipSet * GetClipSet()  const { return fClipSet; }

   // External GUI component interface
   TGLCamera & CurrentCamera() const { return *fCurrentCamera; }
   void SetCurrentCamera(ECameraType camera);
   void SetOrthoCamera(ECameraType camera, Double_t left, Double_t right, Double_t top, Double_t bottom);
   void SetPerspectiveCamera(ECameraType camera, Double_t fov, Double_t dolly,
                             Double_t center[3], Double_t hRotate, Double_t vRotate);
   void GetGuideState(Int_t & axesType, Bool_t & referenceOn, Double_t referencePos[3]) const;
   void SetGuideState(Int_t axesType, Bool_t referenceOn, const Double_t referencePos[3]);
   TGLCameraMarkupStyle* GetCameraMarkup() const { return fCameraMarkup; }
   void SetCameraMarkup(TGLCameraMarkupStyle* m) { fCameraMarkup = m; }

   const TGLPhysicalShape * GetSelected() const;

   // Draw and selection
   // Request methods post cross thread request via TROOT::ProcessLineFast().
   void RequestDraw(Short_t LOD = TGLRnrCtx::kLODMed); // Cross thread draw request
   virtual void PreRender();
   void DoDraw();

   void DrawGuides();
   void DrawCameraMarkup();
   void DrawDebugInfo();

   Bool_t RequestSelect(Int_t x, Int_t y, Bool_t trySecSel=kFALSE); // Cross thread select request
   Bool_t DoSelect(Int_t x, Int_t y, Bool_t trySecSel=kFALSE);      // Window coords origin top left
   void   ApplySelection();

   Bool_t RequestOverlaySelect(Int_t x, Int_t y); // Cross thread select request
   Bool_t DoOverlaySelect(Int_t x, Int_t y);      // Window coords origin top left

   // Update/camera-reset
   void   UpdateScene();
   Bool_t GetIgnoreSizesOnUpdate() const        { return fIgnoreSizesOnUpdate; }
   void   SetIgnoreSizesOnUpdate(Bool_t v)      { fIgnoreSizesOnUpdate = v; }
   void   ResetCurrentCamera();
   Bool_t GetResetCamerasOnUpdate() const       { return fResetCamerasOnUpdate; }
   void   SetResetCamerasOnUpdate(Bool_t v)     { fResetCamerasOnUpdate = v; }
   Bool_t GetResetCameraOnDoubleClick() const   { return fResetCameraOnDoubleClick; }
   void   SetResetCameraOnDoubleClick(Bool_t v) { fResetCameraOnDoubleClick = v; }

   virtual void PostSceneBuildSetup(Bool_t resetCameras);

   virtual void SelectionChanged(); // *SIGNAL*

   // Interaction - events to ExecuteEvent are passed on to these
   Bool_t HandleEvent(Event_t *ev);
   Bool_t HandleButton(Event_t *ev);
   Bool_t HandleDoubleClick(Event_t *ev);
   Bool_t HandleConfigureNotify(Event_t *ev);
   Bool_t HandleKey(Event_t *ev);
   Bool_t HandleMotion(Event_t *ev);
   Bool_t HandleExpose(Event_t *ev);

   void SetPadEditor(TGLViewerEditor *ed){fPadEditor = ed;}

   ClassDef(TGLViewer,0) // Standard ROOT GL viewer.
};



// TODO: Find a better place/way to do this
class TGLRedrawTimer : public TTimer
{
private:
   TGLViewer & fViewer;
   Short_t     fRedrawLOD;
   Bool_t      fPending;
public:
   TGLRedrawTimer(TGLViewer & viewer) :
      fViewer(viewer), fRedrawLOD(TGLRnrCtx::kLODHigh), fPending(kFALSE) {}
   ~TGLRedrawTimer() {}
   void RequestDraw(Int_t milliSec, Short_t redrawLOD=TGLRnrCtx::kLODHigh)
   {
      if (fPending) TurnOff(); else fPending = kTRUE;
      if (redrawLOD > fRedrawLOD) fRedrawLOD = redrawLOD;
      TTimer::Start(milliSec, kTRUE);
   }
   virtual void Stop()
   {
      if (fPending) { TurnOff(); fPending = kFALSE; }
   }
   Bool_t Notify()
   {
      TurnOff();
      fPending = kFALSE;
      fViewer.RequestDraw(fRedrawLOD);
      return kTRUE;
   }
};


#endif // ROOT_TGLViewer
