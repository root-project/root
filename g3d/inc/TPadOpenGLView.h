// @(#)root/g3d:$Name$:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   08/05/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
//*

#ifndef ROOT_TPadOpenGLView
#define ROOT_TPadOpenGLView


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPadOpenGLView                                                       //
//                                                                      //
// TPadOpenGLView is a window in which an OpenGL representation of a    //
// pad is viewed.                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TPadView3D
#include "TPadView3D.h"
#endif
#ifndef ROOT_Buttons
#include "Buttons.h"
#endif


class TGLViewerImp;
class TNode;

class TPadOpenGLView : public TPadView3D
{
 private:
   enum EDrawMode { kHiddenLine=-1, kSolidView=0, kWireFrame=1 };
   UInt_t        fGLList;      // OpenGL lis to implement PROJECTION
   UInt_t        fGLLastList;  // OpenGL free list free to implement PROJECTION
   Bool_t        fMouseInit;   // Turn mouse activities on/off
   Int_t         fMouseX;      // Current X mouse position
   Int_t         fMouseY;      // Current Y mouse position
   Float_t       fSpeedMove;
   Float_t       fStep[3];      // the steps to move object with keyboard interactions
   Bool_t        fResetView;    // Flag whether we need to reset OpenGL view from the TPad::GetView();
   Bool_t        fPerspective;  // Flag to switch between the perspectibe and orthographic projection view
   Double_t      fExtraRotMatrix[16]; // The current GL projection rotation matrix defined via "mouse"
   EDrawMode     fCurrentMode;

   TGLViewerImp *fGLViewerImp;  // Pointer to the OpenGL viewer

 public:
    virtual void MapOpenGL();
    void         UpdateModelView();
    void         UpdateObjectView();
    void         MoveModelView(const Char_t option,Int_t count=1);
    void         MoveModelView(const Char_t *commands, Int_t display_time=0);
    void         RotateView(Int_t x, Int_t y);



    enum { kScene=0,       // defines the common object to resize and re-paint operations
           kProject,       // defines the "Projection" transformation
           kUpdateView,    // changes the "Viewing"    transformation (after the interactive acts for example)
           kView,          // defines the "Viewing"    transformation
           kModel,         // defines the "Modeling"   transformation
           kGLListSize };  // The size of this list

   TPadOpenGLView(){ }
   TPadOpenGLView(TVirtualPad *pad);
   virtual ~TPadOpenGLView();
   virtual void ExecuteEvent(Int_t event, Int_t px, Int_t py);
   Int_t        GetGLList() {return fGLList ? fGLList+1 : 0;}
   UInt_t       GetNextGLList() {return fGLList ? ++fGLLastList : 0;}
   Float_t      GetSpeedMove() { return fSpeedMove;}
   virtual void GetSteps(Float_t *steps) {if (steps) {steps[0] = fStep[0]; steps[1] = fStep[1]; steps[2] = fStep[2];}; }
   virtual void Paint(Option_t *option="");
   virtual void Size(Int_t width, Int_t height);
   UInt_t       ReleaseLastGLList() {return fGLList ? --fGLLastList : 0;}

   virtual void PaintBeginModel(Option_t *opt="");
   virtual void PaintEnd(Option_t *opt="");
   virtual void PaintScene(Option_t *opt="");
   virtual void PaintPolyMarker(TPolyMarker3D *marker,Option_t *opt="");
   virtual void PaintPolyLine(TPolyLine3D *line,Option_t *opt="");
   virtual void PaintPoints3D(const TPoints3DABC *line,Option_t *opt="");
   virtual void PushMatrix();
   virtual void PopMatrix();
   virtual void ResetView(Bool_t flag=kTRUE){fResetView=flag;}
   virtual void SetAttNode(TNode *node,Option_t *opt="");
   virtual void SetLineAttr(Color_t color,Int_t  width, Option_t *opt="");
   virtual void SetSpeedMove(Float_t speed) { fSpeedMove = speed;}
   virtual void SetSteps(Float_t *steps) {if (steps) {fStep[0] = steps[0]; fStep[1] = steps[1]; fStep[2] = steps[2];}; }
   virtual void UpdateNodeMatrix(TNode *node,Option_t *opt="");
   virtual void UpdatePosition(Double_t x,Double_t y,Double_t z,TRotMatrix *matrix,Option_t *opt="");

   virtual void UpdateProjectView();

//   ClassDef(TPadOpenGLView,0)   //3D OpenGL Viewer
};


#endif
