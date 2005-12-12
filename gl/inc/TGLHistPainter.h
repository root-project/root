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

namespace Root_GL {

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
   
}

class TString;
class TAxis;

class TGLHistPainter : public TVirtualHistPainter, public TVirtualGLPainter {
public:
   enum {kTexLength = 16};

private:
   //If gl hist painter does not support Paint option
   //it has to delegate Paint call to default hist painter
   TVirtualHistPainter *fDefaultPainter;
   TH1                 *fHist;

   enum EGLPaintOption {
      kLego,//for lego and lego1
      kSurface,
      kSurface1,
      kSurface2,
      kSurface4,
      kUnsupported
   };
   
   EGLPaintOption       fLastOption;
   
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

   Bool_t               fLogX;
   Bool_t               fLogY;
   Bool_t               fLogZ;
   
   std::vector<Double_t> fX;
   std::vector<Double_t> fY;

   Root_GL::T2DArray<TGLVertex3> fMesh;
   Root_GL::T2DArray<std::pair<TGLVector3, TGLVector3> > fFaceNormals;
   Root_GL::T2DArray<TGLVector3> fAverageNormals;
   
   std::vector<Double_t> fZLevels;

   Int_t               fGLDevice;
   mutable TGLVertex3  f2DAxes[8];

   mutable Bool_t      f2DPass;
   mutable UInt_t      fTextureName;

   UChar_t             fTexture[kTexLength * 4];

public:
   TGLHistPainter(TH1 *hist);
   ~TGLHistPainter();
   
   //TVirtualHistPainter final overriders
   Int_t    DistancetoPrimitive(Int_t px, Int_t py);
   void     DrawPanel();
   void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
   void     FitPanel();
   TList   *GetContourList(Double_t contour)const;
   char    *GetObjectInfo(Int_t px, Int_t py)const;
   TList   *GetStack()const;
   Bool_t   IsInside(Int_t x, Int_t y);
   Bool_t   IsInside(Double_t x, Double_t y);
   
   void     Paint(Option_t *option);
   void     Paint();
   
   void     PaintStat(Int_t dostat, TF1 *fit);
   void     ProcessMessage(const char *mess, const TObject *obj);
   void     SetHistogram(TH1 *hist);
   void     SetStack(TList *stack);
   Int_t    MakeCuts(char *cutsOpt);
   
private:

   TGLHistPainter(const TGLHistPainter &);
   TGLHistPainter &operator = (const TGLHistPainter &);

   void InitDefaultPainter();
   static EGLPaintOption GetPaintOption(const TString &option);

   //inner painting stuff
   Bool_t         InitPainter();
   void           InitTexture();
   Bool_t         SetSizes();
   static Bool_t  SetAxisRange(const TAxis *axis, Bool_t log, Int_t &first, Int_t &last,
                               Double_t &min, Double_t &max);
   void           SetZLevels();
   void           AdjustScales();

   void           FillVertices();
   void           SetNormals();
   void           SetAverageNormals();

   void           InitGL()const;
   
   void           SetGLParameters();

   void           PaintLego()const;
   void           PaintSurface()const;
   void           PaintSurface4()const;
   void           PaintSurface1()const;

   void           DrawFrame(Int_t frontPoint)const;
   void           DrawBackPlane(Int_t plane)const;
   void           DrawProfile(Int_t plane)const;
   void           DrawLegoProfileX(Int_t plane)const;
   void           DrawSurfaceProfileX(Int_t plane)const;
   void           DrawSurfaceProfileY(Int_t plane)const;
   void           DrawLegoProfileY(Int_t plane)const;
   void           DrawGrid(Int_t plane)const;
   void           DrawZeroPlane()const;
   void           DrawAxes(Int_t frontPoint)const;
   Int_t          FrontPoint()const;

   typedef std::pair<Double_t, Double_t> PD_t;
   
   PD_t           GetMaxRowContent(Int_t row)const;
   PD_t           GetMaxColumnContent(Int_t column)const;

   void           SetCamera()const;
   void           SetTransformation()const;

   void           ClearBuffer()const;

   Bool_t         Select(Int_t x, Int_t y)const;
   void           SelectAxes(Int_t front, Int_t x, Int_t y)const;           
   //
   static void    DrawBox(Double_t xmin, Double_t xmax, Double_t ymin, 
                          Double_t ymax, Double_t zmin, Double_t zmax, Int_t frontPoint);
   static void    DrawFlatFace(const TGLVertex3 &v1, const TGLVertex3 &v2, 
                               const TGLVertex3 &v3, const TGLVector3 &normal);
   static void    DrawFace(const TGLVertex3 &v1, const TGLVertex3 &v2, const TGLVertex3 &v3,
                           const TGLVector3 &norm1, const TGLVector3 &norm2, const TGLVector3 &norm3);
   static void    DrawFaceTextured(const TGLVertex3 &v1, const TGLVertex3 &v2, 
                                   const TGLVertex3 &v3, const TGLVector3 &norm1,
                                   const TGLVector3 &norm2, const TGLVector3 &norm3,
                                   Double_t zMin, Double_t zMax);
   static void    DrawQuadOutline(const TGLVertex3 &v1, const TGLVertex3 &v2,
                                  const TGLVertex3 &v3, const TGLVertex3 &v4);

   //
   ClassDef(TGLHistPainter, 0) //GL hist painter
};

#endif
