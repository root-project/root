// @(#)root/gl:$Name:  $:$Id: TGLRender.cxx,v 1.12 2004/10/21 15:11:50 rdm Exp $
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

#include "Rstrstream.h"
#include "TError.h"
#include "TGLSceneObject.h"
#include "TGLRender.h"
#include "TGLCamera.h"

namespace std {} using namespace std;

const UChar_t gXyz[][8] = {{0x44, 0x44, 0x28, 0x10, 0x10, 0x28, 0x44, 0x44},
                           {0x10, 0x10, 0x10, 0x10, 0x10, 0x28, 0x44, 0x44},
                           {0x7c, 0x20, 0x10, 0x10, 0x08, 0x08, 0x04, 0x7c}};

const UChar_t gDigits[][8] = {{0x38, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x38},//0
                              {0x10, 0x10, 0x10, 0x10, 0x10, 0x70, 0x10, 0x10},//1
                              {0x7c, 0x44, 0x20, 0x18, 0x04, 0x04, 0x44, 0x38},//2
                              {0x38, 0x44, 0x04, 0x04, 0x18, 0x04, 0x44, 0x38},//3
                              {0x04, 0x04, 0x04, 0x04, 0x7c, 0x44, 0x44, 0x44},//4
                              {0x7c, 0x44, 0x04, 0x04, 0x7c, 0x40, 0x40, 0x7c},//5
                              {0x7c, 0x44, 0x44, 0x44, 0x7c, 0x40, 0x40, 0x7c},//6
                              {0x20, 0x20, 0x20, 0x10, 0x08, 0x04, 0x44, 0x7c},//7
                              {0x38, 0x44, 0x44, 0x44, 0x38, 0x44, 0x44, 0x38},//8
                              {0x7c, 0x44, 0x04, 0x04, 0x7c, 0x44, 0x44, 0x7c},//9
                              {0x18, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},//.
                              {0x00, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x00, 0x00}};//-

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
   fPxs = kFALSE;
   fAxes = kFALSE;
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

      RunGLList();
      if (fSelectionBox) {
         fSelectionBox->DrawBox();
      }
      if(fAxes) DrawAxes();
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

//______________________________________________________________________________
void TGLRender::SetAxes(const PDD_t &x, const PDD_t &y, const PDD_t &z)
{
   fAxeD[0] = x;
   fAxeD[1] = y;
   fAxeD[2] = z;
}

//______________________________________________________________________________
void PrintNumber(Double_t x, Double_t y, Double_t z, Double_t num, Double_t ys)
{
#ifdef R__SSTREAM
   ostringstream ss;
#else
   ostrstream ss;
#endif
   ss<<num;
   std::string str(ss.str());
   glRasterPos3d(x, y, z);
   for (UInt_t i = 0, e = str.length(); i < e; ++i) {
      if (str[i] == '.') {
         glBitmap(8, 8, 0., ys, 7., 0., gDigits[10]);
         if (i + 1 < e)
            glBitmap(8, 8, 0., ys, 7., 0., gDigits[str[i + 1] - '0']);
         break;
      } else if (str[i] == '-') {
         glBitmap(8, 8, 0., ys, 7., 0., gDigits[11]);
      } else {
         glBitmap(8, 8, 0., ys, 7., 0., gDigits[str[i] - '0']);
      }
   }
}

//______________________________________________________________________________
void TGLRender::DrawAxes()
{
   if (!fPxs) {
      fPxs = kTRUE;
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   }
   glPushAttrib(GL_DEPTH_BUFFER_BIT);
   glDisable(GL_DEPTH_TEST);

   const Double_t axeColors[][3] = {{1., 0., 0.},
                                    {0., 1., 0.},
                                    {0., 0., 1.}};
   //white axes
   glDisable(GL_LIGHTING);
   glColor3dv(axeColors[3]);

   glBegin(GL_LINES);
   glColor3dv(axeColors[0]);
   glVertex3d(fAxeD[0].first, fAxeD[1].first, fAxeD[2].first);
   glVertex3d(fAxeD[0].second, fAxeD[1].first, fAxeD[2].first);
   glColor3dv(axeColors[1]);
   glVertex3d(fAxeD[0].first, fAxeD[1].first, fAxeD[2].first);
   glVertex3d(fAxeD[0].first, fAxeD[1].second, fAxeD[2].first);
   glColor3dv(axeColors[2]);
   glVertex3d(fAxeD[0].first, fAxeD[1].first, fAxeD[2].first);
   glVertex3d(fAxeD[0].first, fAxeD[1].first, fAxeD[2].second);
   glEnd();

   glColor3dv(axeColors[0]);
   glRasterPos3d(fAxeD[0].second, fAxeD[1].first + 12, fAxeD[2].first);
   glBitmap(8, 8, 0., 0., 0., 0., gXyz[0]);
   PrintNumber(fAxeD[0].second, fAxeD[1].first, fAxeD[2].first, fAxeD[0].second, 9.);
   PrintNumber(fAxeD[0].first, fAxeD[1].first, fAxeD[2].first, fAxeD[0].first, 0.);

   glColor3dv(axeColors[1]);
   glRasterPos3d(fAxeD[0].first, fAxeD[1].second + 12, fAxeD[2].first);
   glBitmap(8, 8, 0, 0, 12., 0, gXyz[1]);
   PrintNumber(fAxeD[0].first, fAxeD[1].second, fAxeD[2].first, fAxeD[1].second, 9.);
   PrintNumber(fAxeD[0].first, fAxeD[1].first, fAxeD[2].first, fAxeD[1].first, 9.);

   glColor3dv(axeColors[2]);
   glRasterPos3d(fAxeD[0].first, fAxeD[1].first, fAxeD[2].second);
   glBitmap(8, 8, 0, 0, 0., 0, gXyz[2]);
   PrintNumber(fAxeD[0].first, fAxeD[1].first, fAxeD[2].second, fAxeD[2].second, 9.);
   PrintNumber(fAxeD[0].first, fAxeD[1].first, fAxeD[2].first, fAxeD[2].first, -9.);

   glEnable(GL_LIGHTING);
   glPopAttrib();
}

void TGLRender::SetFamilyColor(const Float_t *newColor)
{
   if (TObject *ro = fSelectedObj->GetRealObject()) {
      TString famName = ro->GetName();
      for (Int_t i = 0, e = fGLObjects.GetEntriesFast(); i < e; ++i) {
         TObject *obj = ((TGLSceneObject *)fGLObjects[i])->GetRealObject();
         if (obj && obj->GetName() == famName)
            ((TGLSceneObject *)fGLObjects[i])->SetColor(newColor);
      }
   } else fSelectedObj->SetColor(newColor);
}
