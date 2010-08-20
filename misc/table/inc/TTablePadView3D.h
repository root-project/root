// @(#)root/table:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   30/05/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTablePadView3D
#define ROOT_TTablePadView3D


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTablePadView3D                                                      //
//                                                                      //
// TTablePadView3D is a generic 3D viewer.                              //
// For a concrete viewer see TGLViewer.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif



class TVirtualPad;
class TPolyMarker3D;
class TPolyLine3D;
class TPoints3DABC;
class TNode;
class TRotMatrix;

class TTablePadView3D {
protected:
   TVirtualPad  *fParent;            // Pointer to the original TPad object

   Double_t      fViewBoxMin[3];     // Minimum of clip box
   Double_t      fViewBoxMax[3];     // Maximum of clip box
   Double_t      fTranslate[3];      // The vector to move object into the center of the scene
   Double_t      fExtraTranslate[3]; // The vector to move object with a mouse
   Double_t      fAngles[3];         // Latitude, Longitude, Psi
   Double_t      fExtraAngles[3];    // Latitude, Longitude, Psi
   Double_t      fAnglFactor[3];     // Latitude, Longitude, Psi
   Float_t       fScale;             // The scale factor to control the border of the clip box

   void Init(TVirtualPad *pad) {
      fParent = pad;
      for(UInt_t i=0; i<3; ++i) {
         fViewBoxMin[i] = 0;
         fViewBoxMax[i] = 0;
         fTranslate[i] = 0;
         fExtraTranslate[i] = 0;
         fAngles[i] = 0;
         fExtraAngles[i] = 0;
         fAnglFactor[i] = 0;
      }
      fScale = 1;
   }
public:
   TTablePadView3D() { Init(0); }  //default ctor
   TTablePadView3D(TVirtualPad *pad) { Init(pad); }
   virtual ~TTablePadView3D();
   virtual void ExecuteEvent(Int_t event, Int_t px, Int_t py);
   TVirtualPad *GetPad() const { return fParent; }
   virtual void Paint(Option_t *option="");
   virtual void Size(Int_t width, Int_t height);
   virtual void PaintBeginModel(Option_t *opt="");
   virtual void PaintEnd(Option_t *opt="");
   virtual void PaintScene(Option_t *opt="");
   virtual void PaintPolyMarker(TPolyMarker3D *marker, Option_t *opt="");
   virtual void PaintPolyLine(TPolyLine3D *line, Option_t *opt="");
   virtual void PaintPoints3D(const TPoints3DABC *points,Option_t *opt="");
   virtual void PushMatrix() { }
   virtual void PopMatrix() { }
   virtual void SetAttNode(TNode *node, Option_t *opt="");
   virtual void SetLineAttr(Color_t color, Int_t width, Option_t *opt="");
           void SetPad(TVirtualPad *pad=0) { fParent = pad; }
   virtual void UpdateNodeMatrix(TNode *node, Option_t *opt="");
   virtual void UpdatePosition(Double_t x,Double_t y,Double_t z,TRotMatrix *matrix, Option_t *opt="");
   virtual void UpdateView() { }

//  Getter's / Setter's methods for the data-members

   virtual void  GetRange(Double_t min[3], Double_t max[3]) const;
   virtual void  SetRange(Double_t min[3], Double_t max[3]);

   virtual void  GetShift(Double_t main_shift[3], Double_t extra_shift[3]) const;
   virtual void  SetShift(Double_t main_shift[3], Double_t extra_shift[3]);

   virtual void  GetAngles(Double_t main_angles[3], Double_t extra_angles[3]) const;
   virtual void  SetAngles(Double_t main_angles[3], Double_t extra_angles[3]);

   virtual void  GetAnglesFactors(Double_t factors[3]) const;
   virtual void  SetAnglesFactors(Double_t factors[3]);

   virtual Float_t GetScale(){return  fScale;}
   virtual void    SetScale(Float_t scale);


//   ClassDef(TTablePadView3D,0);   //Generic 3D viewer
};

inline void TTablePadView3D::ExecuteEvent(Int_t, Int_t, Int_t) { }
inline void TTablePadView3D::Paint(Option_t *) { }
inline void TTablePadView3D::Size(Int_t, Int_t) { }
inline void TTablePadView3D::PaintBeginModel(Option_t *) { }
inline void TTablePadView3D::PaintEnd(Option_t *) { }
inline void TTablePadView3D::PaintScene(Option_t *) { }
inline void TTablePadView3D::PaintPolyMarker(TPolyMarker3D *, Option_t *) { }
inline void TTablePadView3D::PaintPolyLine(TPolyLine3D *, Option_t *) { }
inline void TTablePadView3D::PaintPoints3D(const TPoints3DABC *,Option_t *){ }
inline void TTablePadView3D::SetAttNode(TNode *, Option_t *) { }
inline void TTablePadView3D::SetLineAttr(Color_t ,Int_t ,Option_t *) { }
inline void TTablePadView3D::UpdateNodeMatrix(TNode *, Option_t *) { }
inline void TTablePadView3D::UpdatePosition(Double_t ,Double_t ,Double_t ,TRotMatrix *, Option_t *){ }

#endif
