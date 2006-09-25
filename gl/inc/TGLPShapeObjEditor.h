#ifndef ROOT_TGLPShapeObjEditor
#define ROOT_TGLPShapeObjEditor

#include <memory>
class TGLPShapeObj;

#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif

#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif

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
class TGLMatView;


class TGLPShapeObjEditor : public TGedFrame {
private:   

   enum ELightMode{kDiffuse, kAmbient, kSpecular, kEmission, kLTot};
   ELightMode     fLMode;

   TGLayoutHints     fLb;  //button layout
   TGLayoutHints     fLe;  //num entry layout
   TGLayoutHints     fLl;  //label layout
   TGLayoutHints     fLs;  //slider layout

   Bool_t            fIsActive;          //editor active

   TGCompositeFrame *fGeoFrame;          //orientation, clipping

   // "Geometry" tab's controls
   TGNumberEntry    *fGeomData[6];       //position and clipping control
   TGButton         *fGeoApplyButton;    //action button
   
   // "Color" tab's controls
   TGCompositeFrame *fColorFrame;        //top frame for color componet control 
   TGLMatView       *fMatView;           //inner structure to handle sphere GL window

   TGButton         *fLightTypes[4];     //light type

   TGHSlider        *fRedSlider;         //red component of selected material    
   TGHSlider        *fGreenSlider;       //green component of selected material 
   TGHSlider        *fBlueSlider;        //blue component of selected material     
   TGHSlider        *fAlphaSlider;       //alpha component of selected material lider;
   TGHSlider        *fShineSlider;       //specular refelction of selected material 

   TGButton         *fColorApplyButton;  //apply to selected
   TGButton         *fColorApplyFamily;  //apply to selected and family
   Bool_t            fIsLight;           //does object emit light
   Float_t           fRGBA[17];          //color multiplet

   Window_t          fGLWin;             //GL window with sphere
   ULong_t           fCtx;               //GL context

   TGLPShapeObj     *fPShapeObj;         //model

   void CreateGeoControls();
   void CreateColorControls();

public:
   TGLPShapeObjEditor(const TGWindow *p = 0,
                      Int_t width = 140, Int_t height = 30,
                      UInt_t options = kChildFrame,
                      Pixel_t back = GetDefaultFrameBackground());
   ~TGLPShapeObjEditor();

   virtual void SetModel(TObject* obj);

   // geometry
   void SetCenter(const Double_t *center);
   void SetScale(const Double_t *scale);
   void GeoDisable();
   void DoGeoButton();
   void GetObjectData(Double_t *shift, Double_t *scale);
   void GeoValueSet(Long_t unusedVal);
   //colors
   void CreateMaterialView();
   void CreateColorRadioButtons();
   void CreateColorSliders();
   void SetColorSlidersPos();
   Bool_t HandleContainerNotify(Event_t *event);
   Bool_t HandleContainerExpose(Event_t *event);
   void DrawSphere()const;
   
   void SetRGBA(const Float_t *rgba);
   const Float_t *GetRGBA()const{return fRGBA;}
   //color slots
   void DoColorSlider(Int_t val);
   void DoColorButton();

   ClassDef(TGLPShapeObjEditor, 0) //GUI for editing TGLViewer attributes
};

#endif
