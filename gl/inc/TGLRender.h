// @(#)root/gl:$Name:  $:$Id: TGLRender.h,v 1.3 2004/09/14 15:37:34 rdm Exp $
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLRender
#define ROOT_TGLRender

#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

class TGLSceneObject;
class TGLSelection;
class TGLCamera;

class TGLRender {
private:
   TObjArray      fGLObjects;
   TObjArray      fGLCameras;

   Bool_t         fAllActive;
   Bool_t         fIsPicking;
   Bool_t         fBoxInList;
   Int_t          fActiveCam;
   Int_t          fDList;
   Int_t          fPlane;
   UInt_t         fSelected;

   TGLSceneObject *fFirstT;
   TGLSceneObject *fSelectedObj;
   TGLSelection   *fSelectionBox;

public:
   TGLRender();
   ~TGLRender();
   void Traverse();
   void SetAllActive()
   {
      fAllActive = kTRUE;
   }
   void SetActive(UInt_t cam);
   void AddNewObject(TGLSceneObject *newObject);
   void AddNewCamera(TGLCamera *newCamera);
   TGLSceneObject *SelectObject(Int_t x, Int_t y, Int_t);
   void MoveSelected(Double_t x, Double_t y, Double_t z);
   void SetPlain(Int_t p)
   {
      fPlane = p;
   }
   Int_t GetSize()const
   {
      return fGLObjects.GetEntriesFast();
   }
   void EndMovement();
   void Invalidate();

private:
   void BuildGLList(Bool_t execute = kFALSE);
   void RunGLList();

   TGLRender(const TGLRender &);
   TGLRender & operator = (const TGLRender &);
};

#endif
