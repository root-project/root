// $Header$

#ifndef ROOT_TAttBBox
#define ROOT_TAttBBox

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

class TAttBBox
{
protected:
   Float_t*  fBBox;   //! Dynamic Float_t[6] X(min,max), Y(min,max), Z(min,max)

   void bbox_init(Float_t infinity=1e6);
   void bbox_zero(Float_t epsilon=0, Float_t x=0, Float_t y=0, Float_t z=0);
   void bbox_clear();

   void bbox_check_point(Float_t x, Float_t y, Float_t z);
   void bbox_check_point(const Float_t* p);

public:
   TAttBBox()          { fBBox = 0; }
   virtual ~TAttBBox() { bbox_clear(); }

   Bool_t   GetBBoxOK() const { return fBBox != 0; }
   Float_t* GetBBox()         { return fBBox; }
   Float_t* AssertBBox()      { if(fBBox == 0) ComputeBBox(); return fBBox; }
   void     ResetBBox()       { if(fBBox != 0) bbox_clear(); }

   virtual void ComputeBBox() = 0;

   ClassDef(TAttBBox,1); // Helper for management of bounding-box information
};


// Inline methods:

inline void TAttBBox::bbox_check_point(Float_t x, Float_t y, Float_t z)
{
   if(x < fBBox[0]) fBBox[0] = x;   if(x > fBBox[1]) fBBox[1] = x;
   if(y < fBBox[2]) fBBox[2] = y;   if(y > fBBox[3]) fBBox[3] = y;
   if(z < fBBox[4]) fBBox[4] = z;   if(z > fBBox[5]) fBBox[5] = z;	 
}

inline void TAttBBox::bbox_check_point(const Float_t* p)
{
   bbox_check_point(p[0], p[1], p[2]);
}

#endif
