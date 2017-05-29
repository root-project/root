// @(#)root/gl:$Id$
// Author: Matevz Tadel   25/09/2006

#ifndef ROOT_TGLPShapeObjEditor
#define ROOT_TGLPShapeObjEditor

#include <memory>

#include "TGedFrame.h"

#include "TGLUtil.h"

#include "TGLPShapeRef.h"

class TGLPShapeObj;
class TGLayoutHints;
class TGCheckButton;
class TGNumberEntry;
class TGButtonGroup;
class TGroupFrame;
class TGHSlider;
class TGRadioButton;
class TGTabElement;
class TGButton;
class TGLViewer;
class TGTab;

class TGLWidget;

class TGLPShapeObjEditor : public TGedFrame,
                           public TGLPShapeRef
{

private:
   enum ELightMode   { kDiffuse, kAmbient, kSpecular, kEmission };
   ELightMode        fLMode;

   TGLayoutHints     fLb;  //button layout
   TGLayoutHints     fLe;  //num entry layout
   TGLayoutHints     fLl;  //label layout
   TGLayoutHints     fLs;  //slider layout

   TGCompositeFrame *fGeoFrame;          //orientation, clipping

   // "Geometry" tab's controls
   TGNumberEntry    *fGeomData[6];       //position and clipping control
   TGButton         *fGeoApplyButton;    //action button

   // "Color" tab's controls
   TGCompositeFrame *fColorFrame;        //top frame for color componet control
   TGLWidget        *fMatView;           //inner structure to handle sphere GL window

   TGButton         *fLightTypes[4];     //light type

   TGHSlider        *fRedSlider;         //red component of selected material
   TGHSlider        *fGreenSlider;       //green component of selected material
   TGHSlider        *fBlueSlider;        //blue component of selected material
   TGHSlider        *fAlphaSlider;       //alpha component of selected material lider;
   TGHSlider        *fShineSlider;       //specular refelction of selected material

   TGButton         *fColorApplyButton;  //apply to selected
   TGButton         *fColorApplyFamily;  //apply to selected and family
   Float_t           fRGBA[17];          //color multiplet

   Window_t          fGLWin;             //GL window with sphere
   ULong_t           fCtx;               //GL context

   TGLPShapeObj     *fPShapeObj;         //model

   void CreateGeoControls();
   void CreateColorControls();

   virtual void DoRedraw();

public:
   TGLPShapeObjEditor(const TGWindow *p = 0,
                      Int_t width = 140, Int_t height = 30,
                      UInt_t options = kChildFrame,
                      Pixel_t back = GetDefaultFrameBackground());
   ~TGLPShapeObjEditor();

   // Virtuals from TGLPShapeRef
   virtual void SetPShape(TGLPhysicalShape * shape);
   virtual void PShapeModified();

   virtual void SetModel(TObject* obj);

   // geometry
   void SetCenter(const Double_t *center);
   void SetScale(const Double_t *scale);
   void DoGeoButton();
   void GetObjectData(Double_t *shift, Double_t *scale);
   void GeoValueSet(Long_t unusedVal);
   //colors
   void CreateColorRadioButtons();
   void CreateColorSliders();
   void SetColorSlidersPos();

   void DrawSphere()const;

   void SetRGBA(const Float_t *rgba);
   const Float_t *GetRGBA()const{return fRGBA;}
   //color slots
   void DoColorSlider(Int_t val);
   void DoColorButton();

   ClassDef(TGLPShapeObjEditor, 0); //GUI for editing attributes of a physical-shape.
};

#endif
