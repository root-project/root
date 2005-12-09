// @(#)root/gl:$Name:  $:$Id: TGLViewer.h,v 1.18 2005/12/05 17:34:44 brun Exp $
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
class TGLWindow; // Remove - TGLManager
class TContextMenu;
class TGLClip;
class TGLClipPlane;
class TGLClipBox;
class TGLManip;
class TGLTransManip;
class TGLScaleManip;
class TGLRotateManip;

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

   // TODO: Put this into a proper draw style flag UInt_t
   // seperated into viewer/scene/physical/logical sections
   // modify TGLDrawable to cache on shape subset
   enum EDrawStyle { kFill, kOutline, kWireFrame };
   enum EClipType  { kClipNone, kClipPlane, kClipBox };
   enum EAxesType  { kAxesNone, kAxesEdge, kAxesOrigin };

private:
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
   Bool_t            fInternalPIDs;          //! using internal physical IDs
   UInt_t            fNextInternalPID;       //! next internal physical ID (from 1 - 0 reserved)

   // Composite shape specific - to TGLScene or helper object?
   typedef std::pair<UInt_t, RootCsg::BaseMesh *> CSPart_t;
   mutable TGLFaceSet     *fComposite; //! Paritally created composite
   UInt_t                  fCSLevel;
   std::vector<CSPart_t>   fCSTokens;

   // Interaction - push most into TGLCamera
   enum EAction         { kNone, kRotate, kTruck, kDolly, kDrag };
   EAction              fAction;
   TPoint               fStartPos;
   TPoint               fLastPos;
   UInt_t               fActiveButtonID;

   // Drawing
   EDrawStyle           fDrawStyle;          //! current draw style (Fill/Outline/WireFrame)  
   TGLRedrawTimer     * fRedrawTimer;        //!
   UInt_t               fNextSceneLOD;       //!

   // Scene is created/owned internally.
   // In future it will be shaped between multiple viewers
   TGLScene       fScene;          //! the GL scene - owned by viewer at present
   TGLRect        fViewport;       //! viewport - drawn area
   UInt_t         fLightState;     //! light states (on/off) mask
   EAxesType      fAxesType;       //! axes type
   Bool_t         fReferenceOn;    //! reference marker on?
   TGLVertex3     fReferencePos;   //! reference position
   Bool_t         fInitGL;         //! has GL been initialised?

   // Clipping
   TGLClipPlane   * fClipPlane;
   TGLClipBox     * fClipBox;
   TGLClip        * fCurrentClip;  //! the current clipping shape
   Bool_t           fClipEdit;

   // Object manipulators - physical + clipping shapes
   TGLTransManip  * fTransManip;    //! translation manipulator
   TGLScaleManip  * fScaleManip;    //! scaling manipulator
   TGLRotateManip * fRotateManip;   //! rotation manipulator 
   TGLManip       * fCurrentManip;  //! current manipulator
    
   // Debug tracing (for scene rebuilds)
   Bool_t         fDebugMode;             //! debug mode (forced rebuild + draw scene/frustum/interest boxes)
   UInt_t         fAcceptedPhysicals;     //! number of physicals accepted in last rebuild
   UInt_t         fRejectedPhysicals;     //! number of physicals rejected in last rebuild
   Bool_t         fIsPrinting;

   ///////////////////////////////////////////////////////////////////////
   // Methods
   ///////////////////////////////////////////////////////////////////////
   // Drawing - can tidy up/remove lots when TGLManager added
   void PreDraw();
   void PostDraw();
   void InitGL();
   void MakeCurrent()  const;
   void SwapBuffers()  const;

   // Scene management - to TGLScene or helper object?
   Bool_t             RebuildScene();
   Int_t              ValidateObjectBuffer(const TBuffer3D & buffer, Bool_t includeRaw) const;
   TGLLogicalShape  * CreateNewLogical(const TBuffer3D & buffer) const;
   TGLPhysicalShape * CreateNewPhysical(UInt_t physicalID, const TBuffer3D & buffer, 
                                        const TGLLogicalShape & logical) const;
   RootCsg::BaseMesh *BuildComposite();

   // Cameras
   void        SetViewport(Int_t x, Int_t y, UInt_t width, UInt_t height);
   void        SetupCameras();
   TGLCamera & CurrentCamera() const { return *fCurrentCamera; }

   // Lights
   void        SetupLights();

   // Clipping
   void SetupClips();
   void ClearClips();

   // Non-copyable class
   TGLViewer(const TGLViewer &);
   TGLViewer & operator=(const TGLViewer &);

protected:
   TGLWindow * fGLWindow;    //! remove - replace with TGLManager

public:
   TGLViewer(TVirtualPad * pad, Int_t x, Int_t y, UInt_t width, UInt_t height);
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

   // External GUI component interface
   void  SetDrawStyle(EDrawStyle drawStyle);
   void  SetCurrentCamera(ECameraType camera);
   void  SetOrthoCamera(ECameraType camera, Double_t left, Double_t right, Double_t top, Double_t bottom);
   void  SetPerspectiveCamera(ECameraType camera, Double_t fov, Double_t dolly, 
                              Double_t center[3], Double_t hRotate, Double_t vRotate);
   void  ToggleLight(ELight light);
   void  SetLight(ELight light, Bool_t on);
   void  GetGuideState(EAxesType & axesType, Bool_t & referenceOn, Double_t referencePos[3]) const;
   void  SetGuideState(EAxesType axesType, Bool_t referenceOn, const Double_t referencePos[3]);
   void  GetClipState(EClipType type, Double_t data[6]) const;
   void  SetClipState(EClipType type, const Double_t data[6]);
   EClipType GetCurrentClip() const;
   void  SetCurrentClip(EClipType type, Bool_t edit);
   void  SetSelectedColor(const Float_t rgba[17]);
   void  SetColorOnSelectedFamily(const Float_t rgba[17]);
   void  SetSelectedGeom(const TGLVertex3 & trans, const TGLVector3 & scale);
   const TGLPhysicalShape * GetSelected() const { return fScene.GetSelected(); }
   
   // Overloadable 
   virtual void PostSceneBuildSetup();
   virtual void SelectionChanged(); // *SIGNAL*
   virtual void ClipChanged();      // *SIGNAL*

   // Draw and selection - unpleasant as we need to send via cross thread
   // gVirtualGL objects to ensure GL context is correct. To be removed when
   // TGLManager is
   void RequestDraw(UInt_t redrawLOD = kMed); // Cross thread draw request
   void DoDraw();
   void RequestSelect(UInt_t x, UInt_t y); // Cross thread select request
   Bool_t DoSelect(const TGLRect & rect); // Window coords origin top left
   void RequestSelectManip(const TGLRect & rect); // Cross thread manipulator select request
   void DoSelectManip(const TGLRect & rect);

   // Interaction - events to ExecuteEvent are passed on to these
   Bool_t HandleEvent(Event_t *ev);
   Bool_t HandleButton(Event_t *ev);
   Bool_t HandleDoubleClick(Event_t *ev);
   Bool_t HandleConfigureNotify(Event_t *ev);
   Bool_t HandleKey(Event_t *ev);
   Bool_t HandleMotion(Event_t *ev);
   Bool_t HandleExpose(Event_t *ev);

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
      Bool_t Notify() { TurnOff(); fViewer.RequestDraw(kHigh); return kTRUE; }
};

#endif // ROOT_TGLViewer
