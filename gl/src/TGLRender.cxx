// @(#)root/gl:$Name:  $:$Id: TArcBall.cxx,v 1.4 2004/09/03 12:52:42 brun Exp $
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

TGLRender::TGLRender()
{
   fGLObjects.SetOwner(kTRUE);
   fGLCameras.SetOwner(kTRUE);
   fGLBoxes.SetOwner(kTRUE);

   fAllActive = kTRUE;
   fIsPicking = kFALSE;
   fBoxInList = kFALSE;
   fActiveCam = 0;
   fDList = 0;
   fPlane = 0;
   fSelected = 0;

   fFirstT = 0;
   fSelectedObj = 0;
   fSelectionBox = 0;
}

TGLRender::~TGLRender()
{
   if(fDList)
      glDeleteLists(fDList, 1);
}

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

   for (;start < end; ++start) {
      TGLCamera *currCam = (TGLCamera *)fGLCameras.At(start);
      currCam->TurnOn();

      if (fSelectionBox) {
         fSelectionBox->GLDraw();
      }

      RunGLList();
   }
}

void TGLRender::SetActive(UInt_t ncam)
{
   fActiveCam = ncam;
   fAllActive = kFALSE;
}

void TGLRender::AddNewObject(TGLSceneObject *newobject, TGLSelection *box)
{
   fGLObjects.AddLast(newobject);
   fGLBoxes.AddLast(box);
}

void TGLRender::AddNewCamera(TGLCamera *newcamera)
{
   fGLCameras.AddLast(newcamera);
}

TGLSceneObject *TGLRender::SelectObject(Int_t x, Int_t y, Int_t cam)
{
   TGLCamera *actCam = (TGLCamera *)fGLCameras.At(cam);
   static std::vector<UInt_t>selectBuff(fGLObjects.GetEntriesFast() * 4);
   std::vector<std::pair<Int_t, Int_t> >objNames;

   glSelectBuffer(selectBuff.size(), &selectBuff[0]);
   glRenderMode(GL_SELECT);
   glInitNames();
   glPushName(0);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   actCam->TurnOn(x, y);
   RunGLList();
   Int_t hits = glRenderMode(GL_RENDER);

   if (hits > 0) {
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
         fSelectionBox = (TGLSelection *)fGLBoxes.At(fSelected - 1);
         Traverse();
      }
   } else if (fSelected) {
      fSelected = 0;
      fSelectedObj = 0;
      fSelectionBox = 0;
      Traverse();
   } else {
      Error("TGLRender::SelectObject", "selection buffer overflow\n");
   }

   return fSelectedObj;
}

void TGLRender::MoveSelected(Double_t x, Double_t y, Double_t z)
{
   if (!fIsPicking) {
      fIsPicking = kTRUE;
   }
   fSelectedObj->Shift(x, y, z);
   fSelectionBox->Shift(x, y, z);
}

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

void TGLRender::BuildGLList(Bool_t exec)
{
   glNewList(fDList, exec ? GL_COMPILE_AND_EXECUTE : GL_COMPILE);

   for (Int_t i = 0, e = fGLObjects.GetEntriesFast(); i < e; ++i) {
      TGLSceneObject *currObj = (TGLSceneObject *)fGLObjects.At(i);
      if (currObj->IsTransparent() && currObj != fSelectedObj) {
         currObj->SetNextT(fFirstT);
         fFirstT = currObj;
      } else if (currObj != fSelectedObj) {
         currObj->GLDraw();
      }
   }

   if (fSelectedObj)
      fSelectedObj->GLDraw();

   while (fFirstT) {
      fFirstT->GLDraw();
      fFirstT = fFirstT->GetNextT();
   }

   glEndList();
}

void TGLRender::RunGLList()
{
   glCallList(fDList);
}

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
