// @(#)root/graf:$Name$:$Id$
// Author: Rene Brun   16/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEllipse
#define ROOT_TEllipse


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEllipse                                                             //
//                                                                      //
// An ellipse.                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif
#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif


class TEllipse : public TObject, public TAttLine, public TAttFill {

protected:
        Float_t    fX1;        //X coordinate of centre
        Float_t    fY1;        //Y coordinate of centre
        Float_t    fR1;        //first radius
        Float_t    fR2;        //second radius
        Float_t    fPhimin;    //Minimum angle (degrees)
        Float_t    fPhimax;    //Maximum angle (degrees)
        Float_t    fTheta;     //Rotation angle (degrees)

public:
        TEllipse();
        TEllipse(Float_t x1, Float_t y1,Float_t r1,Float_t r2=0,Float_t phimin=0, Float_t phimax=360,Float_t theta=0);
        TEllipse(const TEllipse &ellipse);
        virtual ~TEllipse();
                void   Copy(TObject &ellipse);
        virtual Int_t  DistancetoPrimitive(Int_t px, Int_t py);
        virtual void   Draw(Option_t *option="");
        virtual void   DrawEllipse(Float_t x1, Float_t y1, Float_t r1,Float_t r2,Float_t phimin, Float_t phimax,Float_t theta);
        virtual void   ExecuteEvent(Int_t event, Int_t px, Int_t py);
        Float_t        GetX1() {return fX1;}
        Float_t        GetY1() {return fY1;}
        Float_t        GetR1() {return fR1;}
        Float_t        GetR2() {return fR2;}
        Float_t        GetPhimin() {return fPhimin;}
        Float_t        GetPhimax() {return fPhimax;}
        Float_t        GetTheta()  {return fTheta;}
        virtual void   ls(Option_t *option="");
        virtual void   Paint(Option_t *option="");
        virtual void   PaintEllipse(Float_t x1, Float_t y1, Float_t r1,Float_t r2,Float_t phimin, Float_t phimax,Float_t theta);
        virtual void   Print(Option_t *option="");
        virtual void   SavePrimitive(ofstream &out, Option_t *option);
        virtual void   SetPhimin(Float_t phi=0)   {fPhimin=phi;} // *MENU*
        virtual void   SetPhimax(Float_t phi=360) {fPhimax=phi;} // *MENU*
        virtual void   SetR1(Float_t r1) {fR1=r1;} // *MENU*
        virtual void   SetR2(Float_t r2) {fR2=r2;} // *MENU*
        virtual void   SetTheta(Float_t theta=0) {fTheta=theta;} // *MENU*
        virtual void   SetX1(Float_t x1) {fX1=x1;} // *MENU*
        virtual void   SetY1(Float_t y1) {fY1=y1;} // *MENU*

        ClassDef(TEllipse,1)  //An ellipse
};

#endif
