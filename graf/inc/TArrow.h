// @(#)root/graf:$Name$:$Id$
// Author: Rene Brun   17/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TArrow
#define ROOT_TArrow


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TArrow                                                               //
//                                                                      //
// One arrow --->.                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TLine
#include "TLine.h"
#endif
#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif


class TArrow : public TLine, public TAttFill {
protected:
        Float_t      fAngle;        //Arrow opening angle (degrees)
        Float_t      fArrowSize;    //Arrow Size
        TString      fOption;       //Arrow shapes

public:
        TArrow();
        TArrow(Float_t x1, Float_t y1,Float_t x2 ,Float_t y2
                               ,Float_t arrowsize=0.05 ,Option_t *option=">");
        TArrow(const TArrow &arrow);
        virtual ~TArrow();
                void   Copy(TObject &arrow);
        virtual void   Draw(Option_t *option="");
        virtual void   DrawArrow(Float_t x1, Float_t y1,Float_t x2 ,Float_t y2
                               ,Float_t arrowsize=0.05 ,Option_t *option=">");
        Float_t        GetAngle() {return fAngle;}
        Float_t        GetArrowSize() {return fArrowSize;}
        Option_t      *GetOption() const { return fOption.Data();}
        virtual void   Paint(Option_t *option="");
        virtual void   PaintArrow(Float_t x1, Float_t y1,Float_t x2 ,Float_t y2
                                 ,Float_t arrowsize=0.05 ,Option_t *option=">");
        virtual void   SavePrimitive(ofstream &out, Option_t *option);
        virtual void   SetAngle(Float_t angle=60) {fAngle=angle;} // *MENU*
        virtual void   SetArrowSize(Float_t arrowsize=0.05) {fArrowSize=arrowsize;} // *MENU*
        virtual void   SetOption(Option_t *option=">"){ fOption = option;}

        ClassDef(TArrow,1)  //one arrow --->
};

#endif
