#ifndef ROOT_TGLCamera
#define ROOT_TGLCamera

#include <TObject.h>

class TGLCamera : public TObject{
protected:
   const Double_t *fViewVolume;
   const Int_t *fViewPort;
   Double_t fZoom;
   Bool_t fDrawFrame;

public:
   TGLCamera(const Double_t *viewvolume, const Int_t *viewport);
   const Int_t *GetViewport()const
   {
      return fViewPort;
   }
   virtual void TurnOn()const = 0;
   virtual void TurnOn(Int_t x, Int_t y)const = 0;
   void Zoom(Double_t zoom)
   {
      fZoom = zoom;
   }
   void Select()
   {
      fDrawFrame = kTRUE;
   }
private:
   TGLCamera(const TGLCamera &);
   TGLCamera & operator = (const TGLCamera &);
};

class TGLTransformation {
public:
   virtual ~TGLTransformation();
   virtual void Apply()const = 0;
};

class TGLSimpleTransform : public TGLTransformation {
private:
   const Double_t *fRotMatrix;
   Double_t fShift;
   Double_t fX;
   Double_t fY;
   Double_t fZ;
public:
   TGLSimpleTransform(const Double_t *rm, Double_t s, Double_t x,
                      Double_t y, Double_t z);
   void Apply()const;
};

class TGLPerspectiveCamera : public TGLCamera {
private:
   TGLSimpleTransform fTransformation;
public:
   TGLPerspectiveCamera(const Double_t *vv, const Int_t *vp,
                        const TGLSimpleTransform &tr);
   void TurnOn()const;
   void TurnOn(Int_t x, Int_t y)const;
};

class TGLOrthoCamera : public TGLCamera {
private:
   TGLSimpleTransform fTransformation;
public:
   TGLOrthoCamera(const Double_t *viewvolume, const Int_t *viewport,
                  const TGLSimpleTransform &tr);
   void TurnOn()const;
   void TurnOn(Int_t x, Int_t y)const;
};

#endif
