// @(#)root/gl:$Name:  $:$Id: TGLRender.cxx,v 1.7 2004/10/04 07:38:37 brun Exp $
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifdef GDK_WIN32
#include "Windows4Root.h"
#endif

#include <algorithm>
#include <utility>
#include <vector>

#include <GL/gl.h>
#include <GL/glu.h>

#include "TError.h"

#include "TGLSceneObject.h"
#include "TGLRender.h"
#include "TGLCamera.h"

//______________________________________________________________________________
TGLRender::TGLRender()
{
   fGLObjects.SetOwner(kTRUE);
   fGLCameras.SetOwner(kTRUE);

   fAllActive = kTRUE;
   fIsPicking = kFALSE;
   fBoxInList = kFALSE;
   fActiveCam = 0;
   fDList = 0;
   fPlaneEqn[0] = 1.;
   fPlaneEqn[1] = fPlaneEqn[2] = fPlaneEqn[3] = 0.;
   fClipping = kFALSE;
   fSelected = 0;

   fFirstT = 0;
   fSelectedObj = 0;
   fSelectionBox = 0;
}

//______________________________________________________________________________
TGLRender::~TGLRender()
{
   if (fDList) 
      glDeleteLists(fDList, 1);
}

//______________________________________________________________________________
void TGLRender::Traverse()
{
   if (!fDList) {
      if (!(fDList = glGenLists(1))) {
         Error("TGLRender::Travesre", "Could not create gl list\n");
         return;
      }
      BuildGLList();
   }

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   Int_t start = 0, end = fGLCameras.GetEntriesFast();

   if (!fAllActive) {
      start = fActiveCam;
      end = start + 1;
   }

   for (; start < end; ++start) {
      TGLCamera *currCam = (TGLCamera *)fGLCameras.At(start);
      currCam->TurnOn();

      if (fClipping) {
         glClipPlane(GL_CLIP_PLANE0, fPlaneEqn);
      }

      if (fSelectionBox) {
         fSelectionBox->DrawBox();
      }
      RunGLList();
   }
}

//______________________________________________________________________________
void TGLRender::SetActive(UInt_t ncam)
{
   fActiveCam = ncam;
   fAllActive = kFALSE;
}

//______________________________________________________________________________
void TGLRender::AddNewObject(TGLSceneObject *newobject)
{
   fGLObjects.AddLast(newobject);
}

//______________________________________________________________________________
void TGLRender::AddNewCamera(TGLCamera *newcamera)
{
   fGLCameras.AddLast(newcamera);
}

//______________________________________________________________________________
TGLSceneObject *TGLRender::SelectObject(Int_t x, Int_t y, Int_t cam)
{
   TGLCamera *actCam = (TGLCamera *)fGLCameras.At(cam);
   static std::vector<UInt_t>selectBuff(fGLObjects.GetEntriesFast() * 4);
   std::vector<std::pair<UInt_t, Int_t> >objNames;

   glSelectBuffer(selectBuff.size(), &selectBuff[0]);
   glRenderMode(GL_SELECT);
   glInitNames();
   glPushName(0);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   actCam->TurnOn(x, y);
   RunGLList();
   Int_t hits = glRenderMode(GL_RENDER);

   if (hits < 0) {
      Error("TGLRender::SelectObject", "selection buffer overflow\n");
   } else if (hits > 0) {
      objNames.resize(hits);
      for (Int_t i = 0; i < hits; ++i) {
         //object's "depth"
         objNames[i].first = selectBuff[i * 4 + 1];
         //object's name
         objNames[i].second = selectBuff[i * 4 + 3];
      }
      std::sort(objNames.begin(), objNames.end());
      UInt_t chosen = 0;
      TGLSceneObject *hitObject = 0;
      for (Int_t j = 0; j < hits; ++j) {
         chosen = objNames[j].second;
         hitObject = (TGLSceneObject *)fGLObjects.At(chosen - 1);
         if (!hitObject->IsTransparent())
            break;
      }
      if (hitObject->IsTransparent()) {
         chosen = objNames[0].second;
         hitObject = (TGLSceneObject *)fGLObjects.At(chosen - 1);
      }
      if (fSelected != chosen) {
         fSelected = chosen;
         fSelectedObj = hitObject;
         fSelectionBox = fSelectedObj->GetBox();
         Traverse();
      }
   } else if (fSelected) {
      fSelected = 0;
      fSelectedObj = 0;
      fSelectionBox = 0;
      Traverse();
   }

   return fSelectedObj;
}

//______________________________________________________________________________
void TGLRender::MoveSelected(Double_t x, Double_t y, Double_t z)
{
   if (!fIsPicking) {
      fIsPicking = kTRUE;
   }
   fSelectedObj->Shift(x, y, z);
   fSelectionBox->Shift(x, y, z);
}

//______________________________________________________________________________
void TGLRender::SetPlane(const Double_t *n)
{
   fPlaneEqn[0] = n[0];
   fPlaneEqn[1] = n[1];
   fPlaneEqn[2] = n[2];
   fPlaneEqn[3] = n[3];   
}

//______________________________________________________________________________
void TGLRender::EndMovement()
{
   if (fIsPicking) {
      fIsPicking = kFALSE;
      glDeleteLists(fDList, 1);
      if (!(fDList = glGenLists(1))) {
         Error("TGLSceneGraph::EndMovement", " Could not create display list\n");
         return;
      }
      fFirstT = 0;
      BuildGLList();
   }
}

//______________________________________________________________________________
void TGLRender::BuildGLList(Bool_t exec)
{
   glNewList(fDList, exec ? GL_COMPILE_AND_EXECUTE : GL_COMPILE);
   Bool_t isTr = kFALSE;
   if (fSelectedObj && !(isTr = fSelectedObj->IsTransparent())) {
      fSelectedObj->GLDraw();
   }
   
   for (Int_t i = 0, e = fGLObjects.GetEntriesFast(); i < e; ++i) {
      TGLSceneObject *currObj = (TGLSceneObject *)fGLObjects.At(i);
      if (currObj->IsTransparent() && currObj != fSelectedObj) {
         currObj->SetNextT(fFirstT);
         fFirstT = currObj;
      } else if (currObj != fSelectedObj) {
         currObj->GLDraw();
      }
   }

   if (isTr)
      fSelectedObj->GLDraw();

   while (fFirstT) {
      fFirstT->GLDraw();
      fFirstT = fFirstT->GetNextT();
   }

   glEndList();
}

//______________________________________________________________________________
void TGLRender::RunGLList()
{
   glCallList(fDList);
}

//______________________________________________________________________________
void TGLRender::Invalidate()
{
   if(fDList)
      glDeleteLists(fDList, 1);
   if (!(fDList = glGenLists(1))) {
      Error("TGLSceneGraph::EndMovement", " Could not create display list\n");
      return;
   }
   fFirstT = 0;
   BuildGLList();
}
