// @(#)root/g3d:$Name:  $:$Id: TSPHE.h,v 1.3 2000/12/13 15:13:46 brun Exp $
// Author: Rene Brun   13/06/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSPHE
#define ROOT_TSPHE


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TSPHE                                                                  //
//                                                                        //
// SPHE is Sphere. Not implemented yet.                                   //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TBRIK
#include "TShape.h"
#endif

// const Int_t kDiv = 30;               //default number of z segments for semi-sphere

class TSPHE : public TShape {

    private:
       Double_t  *fSiTab;       //! Table of sin(fPhimin) .... sin(Phi)
       Double_t  *fCoTab;       //! Table of cos(fPhimin) .... cos(Phi)
       Double_t  *fCoThetaTab;  //! Table of sin(gThemin) .... cos(Theta)
       Int_t      fNdiv;        // number of divisions
       Int_t      fNz;          //! number of sections
       Float_t    fAspectRatio; // Relation between asumth and grid size (by default 1.0)


    protected:
        Float_t fRmin;    // minimum radius
        Float_t fRmax;    // maximum radius
        Float_t fThemin;  // minimum theta
        Float_t fThemax;  // maximum theta
        Float_t fPhimin;  // minimum phi
        Float_t fPhimax;  // maximum phi
        Float_t faX;      // Coeff along Ox
        Float_t faY;      // Coeff along Oy
        Float_t faZ;      // Coeff along Oz
        
        virtual void    MakeTableOfCoSin();  // Create the table of the fSiTab; fCoTab
        virtual void    PaintGLPoints(Float_t *vertex);



    public:
        TSPHE();
        TSPHE(const char *name, const char *title, const char *material, Float_t rmin, Float_t rmax, Float_t themin,
              Float_t themax, Float_t phimin, Float_t phimax);
        TSPHE(const char *name, const char *title, const char *material, Float_t rmax);
        virtual ~TSPHE();
        virtual Int_t   DistancetoPrimitive(Int_t px, Int_t py);
        virtual Float_t GetRmin() const {return fRmin;}
        virtual Float_t GetRmax() const {return fRmax;}
        virtual Float_t GetThemin() const {return fThemin;}
        virtual Float_t GetThemax() const {return fThemax;}
        virtual Float_t GetPhimin() const {return fPhimin;}
        virtual Float_t GetPhimax() const {return fPhimax;}
        virtual Float_t GetAspectRatio() const { return fAspectRatio;}
        virtual Int_t   GetNumberOfDivisions () const {if (fNdiv) return fNdiv; else return 0; /*kDiv;*/}
        virtual void    Paint(Option_t *option);
        virtual void    SetAspectRatio(Float_t factor=1.0){ fAspectRatio = factor; MakeTableOfCoSin();}
        virtual void    SetEllipse(const Float_t *factors);
        virtual void    SetNumberOfDivisions (Int_t p);
        virtual void    SetPoints  (Float_t *buff);
        virtual void    Sizeof3D() const;

        ClassDef(TSPHE,3)  //SPHE shape
};

#endif
