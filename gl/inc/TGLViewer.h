// @(#)root/gl:$Name:  $:$Id: TGLViewer.h,v 1.11 2005/09/06 09:26:40 brun Exp $
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
class TGLClipShape;
class TGLManip;
class TGLTransManip;
class TGLScaleManip;

/*************************************************************************
 * TGLViewer - TODO
 *
 *
 *
 *************************************************************************/
class TGLViewer : public TVirtualViewer3D
{
   RQ_OBJECT("TGLViewer")
   friend class TGLOutput;
public:

   enum ECameraType { kCameraPerspective, kCameraXOY, kCameraYOZ, kCameraXOZ };
   enum ELight      { kLightFront =  0x00000001, 
                      kLightTop   =  0x00000002, 
                      kLightBottom = 0x00000004,
                      kLightLeft   = 0x00000008,
                      kLightRight  = 0x00000010, 
                      kLightMask   = 0x0000001f }; 
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
   TGLPerspectiveCamera fPerspectiveCamera;  //!
   TGLOrthoCamera       fOrthoXOYCamera;     //!
   TGLOrthoCamera       fOrthoYOZCamera;     //!
   TGLOrthoCamera       fOrthoXOZCamera;     //!
   TGLCamera          * fCurrentCamera;      //!

   // Scene management - to TGLScene or helper object?
   Bool_t            fInternalRebuild;       //! internal scene rebuild in progress?
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
   Bool_t         fDrawAxes;       //! draw scene axes
   Bool_t         fInitGL;         //! has GL been initialised?

   // Clipping
   TGLClipPlane   * fClipPlane;
   TGLClipShape   * fClipBox;
   TGLClip        * fCurrentClip;  //! the current clipping shape
   Bool_t           fClipEdit;

   // Object manipulators - physical + clipping shapes
   TGLTransManip * fTransManip;    //! translation manipulator
   TGLScaleManip * fScaleManip;    //! scaling manipulator
   TGLManip      * fCurrentManip;  //! current manipulator
    
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
   Int_t              ValidateObjectBuffer(const TBuffer3D & buffer, Bool_t logical) const;
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

   // Coordinate conversion
   void WindowToGL(TGLRect & rect)      const { rect.Y() = fViewport.Height() - rect.Y(); }
   void WindowToGL(TGLVertex3 & vertex) const { vertex.Y() = fViewport.Height() - vertex.Y(); }

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

   // Manipulation interface - e.g. external GUI component bindings
   void  SetCurrentCamera(ECameraType camera);
   void  ToggleLight(ELight light);
   void  SetAxes(Bool_t on);
   virtual void  SetDefaultClips();
   void  GetClipState(EClipType type, std::vector<Double_t> & data) const;
   void  SetClipState(EClipType type, const std::vector<Double_t> & data);
   EClipType GetCurrentClip() const;
   void  SetCurrentClip(EClipType type, Bool_t edit);
   void  SetSelectedColor(const Float_t rgba[4]);
   void  SetColorOnSelectedFamily(const Float_t rgba[4]);
   void  SetSelectedGeom(const TGLVertex3 & trans, const TGLVector3 & scale);
   const TGLPhysicalShape * GetSelected() const { return fScene.GetSelected(); }
   
   virtual void SelectionChanged(); // *SIGNAL*
   virtual void ClipChanged();      // *SIGNAL*

   // Draw and selection - unpleasant as we need to send via cross thread
   // gVirtualGL objects to ensure GL context is correct. To be replaced with
   // TGLManager
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
