// $Header: /user/cvs/root/gl/inc/TPointSet3DGL.h,v 1.1 2006/04/07 08:43:59 brun Exp $

#ifndef ROOT_TPointSet3DGL
#define ROOT_TPointSet3DGL

#include <TGLObject.h>

class TPointSet3DGL : public TGLObject
{
protected:
  virtual void DirectDraw(const TGLDrawFlags & flags) const;

public:
   TPointSet3DGL();

   virtual Bool_t SetModel(TObject* obj);
   virtual void   SetBBox();

   virtual Bool_t ShouldCache(const TGLDrawFlags & /*flags*/) const { return false; }

  ClassDef(TPointSet3DGL,1)
}; // endclass TPointSet3DGL

#endif
