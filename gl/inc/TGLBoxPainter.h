#ifndef ROOT_TGLBoxPainter
#define ROOT_TGLBoxPainter

#ifndef ROOT_TGLPlotPainter
#include "TGLPlotPainter.h"
#endif
#ifndef ROOT_TGLQuadric
#include "TGLQuadric.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif
#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TGLOrthoCamera;
class TAxis;
class TH3;
class TH1;

class TGLTH3Slice : public TNamed {
public:
   enum ESliceAxis {kXOZ, kYOZ, kXOY};
   
private:
   ESliceAxis                fAxisType;
   TAxis                    *fAxis;
   mutable TGLLevelPalette   fPalette;

   const TGLPlotCoordinates *fCoord;
   const TGLPlotBox         *fBox;
   Int_t                     fSliceWidth;

   const TH3                *fHist;

   mutable TGL2DArray<Double_t> fTexCoords;

public:
   TGLTH3Slice(const TString &sliceName, 
               const TH3 *hist, 
               const TGLPlotCoordinates *coord, 
               const TGLPlotBox * box,
               ESliceAxis axis);

   void DrawSlice(Double_t pos)const;
   //SetSliceWidth must have "menu" comment.
   void SetSliceWidth(Int_t width = 1); // *MENU*

private:
   void   PrepareTexCoords()const;
   void   FindMinMax(Double_t &minVal, Double_t &maxVal, Int_t low, Int_t max)const;
   Bool_t PreparePalette(Double_t minVal, Double_t maxVal)const;
   void   DrawSliceTextured(Double_t pos)const;
   void   DrawSliceFrame(Int_t low, Int_t up)const;

   ClassDef(TGLTH3Slice, 0) // TH3 slice
};

class TGLBoxPainter : public TGLPlotPainter {
private:
   TGLTH3Slice fXOZSlice;
   TGLTH3Slice fYOZSlice;
   TGLTH3Slice fXOYSlice;

   enum EBoxType {
      kBox, //boxes, sizes are proportional to bin content
      kBox1 //spheres, not boxes
   };

   mutable EBoxType        fType;

   TString                 fPlotInfo;
   Rgl::Range_t            fMinMaxVal;   

   mutable TGLQuadric      fQuadric;

   TGLBoxPainter(const TGLBoxPainter &);
   TGLBoxPainter &operator = (const TGLBoxPainter &);

public:
   TGLBoxPainter(TH1 *hist, TGLOrthoCamera *camera, TGLPlotCoordinates *coord, Int_t ctx = -1);
   
   char   *GetPlotInfo(Int_t px, Int_t py);
   Bool_t  InitGeometry();
   void    StartPan(Int_t px, Int_t py);
   void    Pan(Int_t px, Int_t py);
   void    AddOption(const TString &stringOption);
   void    ProcessEvent(Int_t event, Int_t px, Int_t py);

private:
   //Overriders
   void    InitGL()const;
   void    DrawPlot()const;
   void    ClearBuffers()const;

   void    SetPlotColor()const;

   void    DrawSectionXOZ()const;
   void    DrawSectionYOZ()const;
   void    DrawSectionXOY()const;

   Bool_t  HasSections()const;

   ClassDef(TGLBoxPainter, 0)//Box painter
};

#endif
