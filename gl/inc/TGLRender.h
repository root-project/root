#ifndef ROOT_TGLRender
#define ROOT_TGLRender

#include <TObjArray.h>

class TGLSceneObject;
class TGLSelection;
class TGLCamera;

class TGLRender {
private:
   TObjArray fGLObjects;
   TObjArray fGLCameras;
   TObjArray fGLBoxes;

   Bool_t fAllActive;
   Bool_t fIsPicking;
   Bool_t fBoxInList;
   Int_t fActiveCam;
   Int_t fDList;
   Int_t fPlane;
   UInt_t fSelected;

   TGLSceneObject *fFirstT;
   TGLSceneObject *fSelectedObj;
   TGLSceneObject *fSelectionBox;

public:
   TGLRender();
   ~TGLRender();
   void Traverse();
   void SetAllActive()
   {
      fAllActive = kTRUE;
   }
   void SetActive(UInt_t cam);
   void AddNewObject(TGLSceneObject *newobject, TGLSelection *box);
   void AddNewCamera(TGLCamera *newcamera);
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
