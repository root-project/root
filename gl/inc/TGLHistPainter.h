#ifndef ROOT_TGLHistPainter
#define ROOT_TGLHistPainter

#include <utility>
#include <vector>

#ifndef ROOT_TVirtualHistPainter
#include "TVirtualHistPainter.h"
#endif

#ifndef ROOT_TVirtualGL
#include "TVirtualGL.h"
#endif

#ifndef ROOT_TArcBall
#include "TArcBall.h"
#endif

#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif

#ifndef ROOT_TPoint
#include "TPoint.h"
#endif

class TString;
class TAxis;
class TF3;

namespace RootGL {

   template<class T>
   class T2DArray : public std::vector<T> {
   private:
      Int_t fRowLen;
      Int_t fMaxRow;
      typedef typename std::vector<T>::size_type size_type;

   public:
      T2DArray() : fRowLen(0), fMaxRow(0){}
      void SetMaxRow(Int_t max)
      {
         fMaxRow = max;
      }
      void SetRowLen(Int_t len)
      {
         fRowLen = len;
      }
      const T *operator [] (size_type ind)const
      {
         return &std::vector<T>::operator [](ind * fRowLen);
      }
      T *operator [] (size_type ind)
      {
         return &std::vector<T>::operator [] (ind * fRowLen);
      }
   };
   
   struct TGLTriFace_t {
      TGLVertex3 fXYZ[3];
      TGLVector3 fNormals[3];
   };
}

class TGLHistPainter : public TVirtualHistPainter, private TVirtualGLPainter{
private:
   //If gl hist painter does not support Paint option
   //it has to delegate Paint call to default hist painter
   TVirtualHistPainter *fDefaultPainter;
   TH1                 *fHist;
   mutable TF3         *fF3;

   enum EGLPaintOption {
      kLego,//for lego and lego1
      kLego2,
      kSurface,
      kSurface1,
      kSurface2,
      kSurface4,
      kTF3,
      kUnsupported
   };

   EGLPaintOption       fLastOption;

   enum EGLTF3Style {
      kDefault,
      kMaple0,
      kMaple1,
      kMaple2
   };

   EGLTF3Style         fTF3Style;

   TAxis               *fAxisX;
   TAxis               *fAxisY;
   TAxis               *fAxisZ;

   Double_t             fMinX;
   Double_t             fMaxX;
   Double_t             fScaleX;
   Double_t             fMinXScaled;
   Double_t             fMaxXScaled;
   Double_t             fMinY;
   Double_t             fMaxY;
   Double_t             fScaleY;
   Double_t             fMinYScaled;
   Double_t             fMaxYScaled;
   Double_t             fMinZ;
   Double_t             fMaxZ;
   Double_t             fScaleZ;
   Double_t             fMinZScaled;
   Double_t             fMaxZScaled;
   Double_t             fFactor;

   TArcBall             fRotation;
   Double_t             fFrustum[4];
   Double_t             fCenter[3];
   Double_t             fShift;
   Int_t                fViewport[4];

   Int_t                fFirstBinX;
   Int_t                fLastBinX;
   Int_t                fFirstBinY;
   Int_t                fLastBinY;
   Int_t                fFirstBinZ;
   Int_t                fLastBinZ;

   Bool_t               fLogX;
   Bool_t               fLogY;
   Bool_t               fLogZ;

   std::vector<Double_t> fX;
   std::vector<Double_t> fY;
   std::vector<Double_t> fZ;

   RootGL::T2DArray<TGLVertex3> fMesh;
   RootGL::T2DArray<std::pair<TGLVector3, TGLVector3> > fFaceNormals;
   RootGL::T2DArray<TGLVector3> fAverageNormals;

   std::vector<RootGL::TGLTriFace_t> fF3Mesh;

   std::vector<Double_t> fZLevels;

   Int_t                fGLDevice;
   mutable TGLVertex3   f2DAxes[8];

   mutable Bool_t       f2DPass;
   mutable UInt_t       fTextureName;

   std::vector<UChar_t> fTexture;
   void (TGLHistPainter::*fCurrentPainter)()const;
   mutable Int_t        fFrontPoint;
   
   Double_t             fZoom;
   TGLVertex3           fPan;
   TPoint               fCurrPos;
   
public:
   TGLHistPainter(TH1 *hist);
   ~TGLHistPainter();
   
   //TVirtualHistPainter final overriders
   Int_t          DistancetoPrimitive(Int_t px, Int_t py);
   void           DrawPanel();
   void           ExecuteEvent(Int_t event, Int_t px, Int_t py);
   void           FitPanel();
   TList         *GetContourList(Double_t contour)const;
   char          *GetObjectInfo(Int_t px, Int_t py)const;
   TList         *GetStack()const;
   Bool_t         IsInside(Int_t x, Int_t y);
   Bool_t         IsInside(Double_t x, Double_t y);

   void           Paint(Option_t *option);

   void           PaintStat(Int_t dostat, TF1 *fit);
   void           ProcessMessage(const char *mess, const TObject *obj);
   void           SetHistogram(TH1 *hist);
   void           SetStack(TList *stack);
   Int_t          MakeCuts(char *cutsOpt);
   void           SetShowProjection(const char * /*option*/) {;}

private:
   EGLPaintOption SetPaintFunction(TString &option);
   //TVirtualGLPainter's final overrider
   void           Paint();
   //Texture for lego2/surf1/surf2 options
   void           SetTexture();
   void           EnableTexture()const;
   void           DisableTexture()const;

   void           SetZLevels();
   void           AdjustScales();

   Bool_t         SetVertices();
   Bool_t         SetAxes();
   void           SetTable();
   void           SetMesh();
   void           SetNormals();
   void           SetAverageNormals();
   void           SetTF3Mesh();

   void           InitGL()const;
   Bool_t         MakeCurrent()const;

   void           CalculateTransformation();

   void           DrawLego()const;
   void           DrawSurface()const;
   void           DrawTF3()const;

   void           DrawFrame()const;
   void           DrawBackPlane(Int_t plane)const;
   void           DrawProfile(Int_t plane)const;
   void           DrawLegoProfileX(Int_t plane)const;
   void           DrawLegoProfileY(Int_t plane)const;
   void           DrawSurfaceProfileX(Int_t plane)const;
   void           DrawSurfaceProfileY(Int_t plane)const;
   void           DrawGrid(Int_t plane)const;
   void           DrawZeroPlane()const;
   void           DrawAxes()const;

   Int_t          FrontPoint()const;

   void           PrintPlot();

   void           SetPlotColor()const;
   void           SetCamera()const;
   void           SetTransformation()const;
   void           ClearBuffers()const;

   
   void           Pan(Int_t newX, Int_t newY);
   Bool_t         Select(Int_t x, Int_t y)const;
   void           SelectAxes(Int_t front, Int_t x, Int_t y)const;

   TGLHistPainter(const TGLHistPainter &);
   TGLHistPainter &operator = (const TGLHistPainter &);

   ClassDef(TGLHistPainter, 0) //GL hist painter
};

#endif
