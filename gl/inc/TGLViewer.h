// @(#)root/gl:$Name:  $:$Id: TGLViewer.h,v 1.33 2006/10/11 10:26:23 rdm Exp $
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
#ifndef ROOT_TGLDrawFlags
#include "TGLDrawFlags.h"
#endif
#ifndef ROOT_TTimer
#include "TTimer.h"
#endif
#ifndef ROOT_TPoint
#include "TPoint.h"
#endif
#ifndef ROOT_CsgOps
#include "CsgOps.h"
#endif
#ifndef ROOT_GuiTypes
#include "GuiTypes.h"
#endif
#ifndef ROOT_RQ_OBJECT
#include "RQ_OBJECT.h"
#endif
#include <vector>

class TGLFaceSet;
class TGLRedrawTimer;
class TGLViewerEditor;
class TGLWindow; // Remove - TGLManager
class TContextMenu;
class TGLCameraMarkupStyle;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLViewer                                                            //
//                                                                      //
// Base GL viewer object - used by both standalone and embedded (in pad)//
// GL. Contains core viewer objects :                                   //
//                                                                      //
// GL scene (fScene) - collection of main drawn objects - see TGLScene  //
// Cameras (fXXXXCamera) - ortho and perspective cameras - see TGLCamera//
// Clipping (fClipXXXX) - collection of clip objects - see TGLClip      //
// Manipulators (fXXXXManip) - collection of manipulators - see TGLManip//
//                                                                      //
// It maintains the current active draw styles, clipping object,        //
// manipulator, camera etc.                                             //
//                                                                      //
// TGLViewer is 'GUI free' in that it does not derive from any ROOT GUI //
// TGFrame etc - see TGLSAViewer for this. However it contains GUI      //
// GUI style methods HandleButton() etc to which GUI events can be      //
// directed from standalone frame or embedding pad to perform           //
// interaction.                                                         //
//                                                                      //
// For embedded (pad) GL this viewer is created directly by plugin      //
// manager. For standalone the derived TGLSAViewer is.                  //
//////////////////////////////////////////////////////////////////////////


class TGLViewer : public TVirtualViewer3D
{
   RQ_OBJECT("TGLViewer")
   friend class TGLOutput;
public:

   enum ECameraType { kCameraPerspXOZ, kCameraPerspYOZ, kCameraPerspXOY,
                      kCameraOrthoXOY, kCameraOrthoXOZ, kCameraOrthoZOY };

   enum ELight      { kLightFront =  0x00000001, 
                      kLightTop   =  0x00000002, 
                      kLightBottom = 0x00000004,
                      kLightLeft   = 0x00000008,
                      kLightRight  = 0x00000010, 
                      kLightMask   = 0x0000001f }; 

   enum EAxesType  { kAxesNone, kAxesEdge, kAxesOrigin };

private:
protected:
   // TODO: Consider what to push up to protected, pull out to TGLScene
   // TGLCamera or other external helpers

   ///////////////////////////////////////////////////////////////////////
   // Fields
   ///////////////////////////////////////////////////////////////////////

   // External handles
   TVirtualPad  * fPad;            //! external pad - remove replace with signal
   
   // GUI Handles
   TContextMenu       * fContextMenu; //!

   // Cameras
   // TODO: Put in vector and allow external creation
   TGLPerspectiveCamera fPerspectiveCameraXOZ; //!
   TGLPerspectiveCamera fPerspectiveCameraYOZ; //!
   TGLPerspectiveCamera fPerspectiveCameraXOY; //!
   TGLOrthoCamera       fOrthoXOYCamera;       //!
   TGLOrthoCamera       fOrthoXOZCamera;       //!
   TGLOrthoCamera       fOrthoZOYCamera;       //!
   TGLCamera          * fCurrentCamera;        //!

   // Scene management - to TGLScene or helper object?
   Bool_t            fInternalRebuild;       //! scene rebuild triggered internally/externally?
   Bool_t            fPostSceneBuildSetup;   //! setup viewer after (re)build complete?
   Bool_t            fAcceptedAllPhysicals;  //! did we take all physicals offered in AddObject()
   Bool_t            fForceAcceptAll;        //! force taking of all logicals/physicals in AddObject()
   Bool_t            fInternalPIDs;          //! using internal physical IDs
   UInt_t            fNextInternalPID;       //! next internal physical ID (from 1 - 0 reserved)

   // Composite shape specific - to TGLScene or helper object?
   typedef std::pair<UInt_t, RootCsg::TBaseMesh *> CSPart_t;
   mutable TGLFaceSet     *fComposite; //! Paritally created composite
   UInt_t                  fCSLevel;
   std::vector<CSPart_t>   fCSTokens;

   // Current camera ineraction - no kZoom as zoom is either key or mouse wheel
   // action so instantaneous
   // TODO: Move all this into TGLCamera? Would need to process TEvents itself
   enum ECameraAction   { kCameraNone, kCameraRotate, kCameraTruck, kCameraDolly };
   ECameraAction        fAction;
   TPoint               fLastPos;
   UInt_t               fActiveButtonID;

   // Drawing
   TGLDrawFlags         fDrawFlags;          //! next draw flags - passed to scene
   TGLRedrawTimer     * fRedrawTimer;        //!

   // Scene is created/owned internally.
   // In future it will be shaped between multiple viewers
   TGLScene       fScene;          //! the GL scene - owned by viewer at present
   TGLRect        fViewport;       //! viewport - drawn area
   Color_t        fClearColor;     //! clear-color
   UInt_t         fLightState;     //! light states (on/off) mask
   EAxesType      fAxesType;       //! axes type
   Bool_t         fReferenceOn;    //! reference marker on?
   TGLVertex3     fReferencePos;   //! reference position
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

   // Scene management - to TGLScene or helper object?
   Bool_t             RebuildScene();
   Int_t              ValidateObjectBuffer(const TBuffer3D & buffer, Bool_t includeRaw) const;
   TGLLogicalShape  * CreateNewLogical(const TBuffer3D & buffer) const;
   TGLPhysicalShape * CreateNewPhysical(UInt_t physicalID, const TBuffer3D & buffer, 
                                        const TGLLogicalShape & logical) const;
   RootCsg::TBaseMesh *BuildComposite();

   // Cameras
   void        SetViewport(Int_t x, Int_t y, UInt_t width, UInt_t height);
   void        SetupCameras(Bool_t reset);

   // Lights
   void        SetupLights();

   // Non-copyable class
   TGLViewer(const TGLViewer &);
   TGLViewer & operator=(const TGLViewer &);

protected:
   TGLWindow       *fGLWindow;    //! remove - replace with TGLManager
   Int_t            fGLDevice; //!for embedded gl viewer
   TGLViewerEditor *fPadEditor;

   std::map<TClass*, TClass*> fDirectRendererMap; //!
   TClass*          FindDirectRendererClass(TClass* cls);
   TGLLogicalShape* AttemptDirectRenderer(TObject* id);

   // Updata/camera-reset behaviour
   Bool_t           fIgnoreSizesOnUpdate;      // ignore sizes of bounding-boxes on update
   Bool_t           fResetCamerasOnUpdate;     // reposition camera on each update
   Bool_t           fResetCamerasOnNextUpdate; // reposition camera on next update
   Bool_t           fResetCameraOnDoubleClick; // reposition camera on double-click

   // Overloadable 
   virtual void PostSceneBuildSetup(Bool_t resetCameras);
   virtual void SelectionChanged(); // *SIGNAL*
   virtual void ClipChanged();      // *SIGNAL*

public:
   TGLViewer(TVirtualPad * pad, Int_t x, Int_t y, UInt_t width, UInt_t height);
   TGLViewer(TVirtualPad * pad);
   virtual ~TGLViewer();

   // TVirtualViewer3D interface
   virtual Int_t  DistancetoPrimitive(Int_t px, Int_t py);
   virtual void   ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual Bool_t PreferLocalFrame() const;
   virtual void   BeginScene();
   virtual Bool_t BuildingScene() const { return fScene.CurrentLock() == TGLScene::kModifyLock; }
   virtual void   EndScene();
   virtual Int_t  AddObject(const TBuffer3D & buffer, Bool_t * addChildren = 0);
   virtual Int_t  AddObject(UInt_t physicalID, const TBuffer3D & buffer, Bool_t * addChildren = 0);
   virtual Bool_t OpenComposite(const TBuffer3D & buffer, Bool_t * addChildren = 0);
   virtual void   CloseComposite();
   virtual void   AddCompositeOp(UInt_t operation);
   virtual void   PrintObjects();
   virtual void   ResetCameras()                { SetupCameras(kTRUE); }
   virtual void   ResetCamerasAfterNextUpdate() { fResetCamerasOnNextUpdate = kTRUE; }

   virtual void   RefreshPadEditor(TObject* =0) {}

   Int_t   GetDev()const{return fGLDevice;}
   Color_t GetClearColor() const             { return fClearColor; }
   void    SetClearColor(Color_t col)        { fClearColor = col; }
   Bool_t  GetSmartRefresh() const           { return fSmartRefresh; }
   void    SetSmartRefresh(Bool_t smart_ref) { fSmartRefresh = smart_ref; }

   // External GUI component interface
   void SetDrawStyle(TGLDrawFlags::EStyle style);
   TGLCamera & CurrentCamera() const { return *fCurrentCamera; }
   void SetCurrentCamera(ECameraType camera);
   void SetOrthoCamera(ECameraType camera, Double_t left, Double_t right, Double_t top, Double_t bottom);
   void SetPerspectiveCamera(ECameraType camera, Double_t fov, Double_t dolly, 
                             Double_t center[3], Double_t hRotate, Double_t vRotate);
   void ToggleLight(ELight light);
   void SetLight(ELight light, Bool_t on);
   UInt_t  GetLightState(){return fLightState;}
   void GetGuideState(EAxesType & axesType, Bool_t & referenceOn, Double_t referencePos[3]) const;
   void SetGuideState(EAxesType axesType, Bool_t referenceOn, const Double_t referencePos[3]);
   void GetClipState(EClipType type, Double_t data[6]) const;
   void SetClipState(EClipType type, const Double_t data[6]);
   void GetCurrentClip(EClipType & type, Bool_t & edit) const;
   void SetCurrentClip(EClipType type, Bool_t edit);
   TGLCameraMarkupStyle* GetCameraMarkup() const { return fScene.GetCameraMarkup(); }
   void SetCameraMarkup(TGLCameraMarkupStyle* m) { fScene.SetCameraMarkup(m); }
   void SetSelectedColor(const Float_t rgba[17]);
   void SetColorOnSelectedFamily(const Float_t rgba[17]);
   void SetSelectedGeom(const TGLVertex3 & trans, const TGLVector3 & scale);
   const TGLPhysicalShape * GetSelected() const { return fScene.GetSelected(); }
   
   // Draw and selection
   // Request methods post cross thread request via TVirtualGL / TGLKernel
   // to ensure correct thread and hence valid GL context under Win32.
   // Can be removed when TGLManager is used
   void RequestDraw(Short_t LOD = TGLDrawFlags::kLODMed); // Cross thread draw request
   void DoDraw();
   Bool_t RequestSelect(UInt_t x, UInt_t y); // Cross thread select request
   Bool_t DoSelect(const TGLRect & rect); // Window coords origin top left
   void   ApplySelection();

   // Update/camera-reset
   void   UpdateScene();
   Bool_t GetIgnoreSizesOnUpdate() const        { return fIgnoreSizesOnUpdate; }
   void   SetIgnoreSizesOnUpdate(Bool_t v)      { fIgnoreSizesOnUpdate = v; }
   void   ResetCurrentCamera();
   Bool_t GetResetCamerasOnUpdate() const       { return fResetCamerasOnUpdate; }
   void   SetResetCamerasOnUpdate(Bool_t v)     { fResetCamerasOnUpdate = v; }
   Bool_t GetResetCameraOnDoubleClick() const   { return fResetCameraOnDoubleClick; }
   void   SetResetCameraOnDoubleClick(Bool_t v) { fResetCameraOnDoubleClick = v; }

   // Interaction - events to ExecuteEvent are passed on to these
   Bool_t HandleEvent(Event_t *ev);
   Bool_t HandleButton(Event_t *ev);
   Bool_t HandleDoubleClick(Event_t *ev);
   Bool_t HandleConfigureNotify(Event_t *ev);
   Bool_t HandleKey(Event_t *ev);
   Bool_t HandleMotion(Event_t *ev);
   Bool_t HandleExpose(Event_t *ev);
   
   void SetPadEditor(TGLViewerEditor *ed){fPadEditor = ed;}

   ClassDef(TGLViewer,0) // GL viewer generic base class
};

//______________________________________________________________________________
inline void TGLViewer::GetClipState(EClipType type, Double_t data[6]) const
{
   fScene.GetClipState(type, data);
}

//______________________________________________________________________________
inline void TGLViewer::SetClipState(EClipType type, const Double_t data[6])
{
   fScene.SetClipState(type, data);
   RequestDraw();
}

//______________________________________________________________________________
inline void TGLViewer::GetCurrentClip(EClipType & type, Bool_t & edit) const
{
   fScene.GetCurrentClip(type, edit);
}

//______________________________________________________________________________
inline void TGLViewer::SetCurrentClip(EClipType type, Bool_t edit)
{
   fScene.SetCurrentClip(type, edit);
   RequestDraw();
}

// TODO: Find a better place/way to do this
class TGLRedrawTimer : public TTimer
{
   private:
      TGLViewer & fViewer;
      Short_t     fRedrawLOD;
   public:
      TGLRedrawTimer(TGLViewer & viewer) : fViewer(viewer), fRedrawLOD(TGLDrawFlags::kLODHigh) {};
      ~TGLRedrawTimer() {};
      void   RequestDraw(Int_t milliSec, Short_t redrawLOD) {
         fRedrawLOD = redrawLOD;
         TTimer::Start(milliSec, kTRUE);
      }
      Bool_t Notify() { TurnOff(); fViewer.RequestDraw(fRedrawLOD); return kTRUE; }
};



// 
// Wrapper class for TGLPhysicalShape class editor
//

class TGLPShapeObj : public TObject
{
public:
   TGLPhysicalShape *fPShape;
   TGLViewer        *fViewer;

   TGLPShapeObj() : TObject(), fPShape(0), fViewer(0) {}
   TGLPShapeObj(TGLPhysicalShape* sh,TGLViewer* v) :
      TObject(), fPShape(sh), fViewer(v) { }
   virtual ~TGLPShapeObj() {}

   virtual const char* GetName() const { return "Selected"; }

private:
   TGLPShapeObj(const TGLPShapeObj &); // Not implemented
   TGLPShapeObj& operator=(const TGLPShapeObj &); // Not implemented

   ClassDef(TGLPShapeObj, 0); // This object wraps TGLPhysicalShape (not a TObject) so 
};


#endif // ROOT_TGLViewer
