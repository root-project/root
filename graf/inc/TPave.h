// @(#)root/graf:$Name:  $:$Id: TPave.h,v 1.1.1.1 2000/05/16 17:00:50 rdm Exp $
// Author: Rene Brun   16/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPave
#define ROOT_TPave


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPave                                                                //
//                                                                      //
// Pave class.                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TBox
#include "TBox.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TPave : public TBox {

protected:
        Double_t     fX1NDC;         //X1 point in NDC coordinates
        Double_t     fY1NDC;         //Y1 point in NDC coordinates
        Double_t     fX2NDC;         //X2 point in NDC coordinates
        Double_t     fY2NDC;         //Y2 point in NDC coordinates
        Int_t        fBorderSize;    //window box bordersize in pixels
        Int_t        fInit;          //(=0 if transformation to NDC not yet done)
        Double_t     fCornerRadius;  //Corner radius in case of option arc
        TString      fOption;        //Pave style
        TString      fName;          //Pave name

public:
        // TPave status bits
        enum {
           kNameIsAction = BIT(11)   // double clicking on TPave will execute action
        };

        TPave();
        TPave(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
              Int_t bordersize=4 ,Option_t *option="br");
        TPave(const TPave &pave);
        virtual ~TPave();
                void  Copy(TObject &pave);
        virtual void  ConvertNDCtoPad();
        virtual void  Draw(Option_t *option="");
        virtual void  DrawPave(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
                      Int_t bordersize=4 ,Option_t *option="br");
        virtual void  ExecuteEvent(Int_t event, Int_t px, Int_t py);
          Int_t       GetBorderSize() { return fBorderSize;}
          Double_t    GetCornerRadius() {return fCornerRadius;}
          Option_t   *GetName() const {return fName.Data();}
          Option_t   *GetOption() const {return fOption.Data();}
          Double_t    GetX1NDC() {return fX1NDC;}
          Double_t    GetX2NDC() {return fX2NDC;}
          Double_t    GetY1NDC() {return fY1NDC;}
          Double_t    GetY2NDC() {return fY2NDC;}
        virtual void  ls(Option_t *option="");
        virtual void  Paint(Option_t *option="");
        virtual void  PaintPave(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
                      Int_t bordersize=4 ,Option_t *option="br");
        virtual void  PaintPaveArc(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
                      Int_t bordersize=4 ,Option_t *option="br");
        virtual void  Print(Option_t *option="");
        virtual void  SavePrimitive(ofstream &out, Option_t *option);
        virtual void  SetBorderSize(Int_t bordersize=4) {fBorderSize = bordersize;} // *MENU*
        virtual void  SetCornerRadius(Double_t rad = 0.2) {fCornerRadius = rad;} // *MENU*
        virtual void  SetName(const char *name="") {fName = name;} // *MENU*
        virtual void  SetOption(Option_t *option="br") {fOption = option;}
        virtual void  SetX1NDC(Double_t x1) {fX1NDC=x1;}
        virtual void  SetX2NDC(Double_t x2) {fX2NDC=x2;}
        virtual void  SetY1NDC(Double_t y1) {fY1NDC=y1;}
        virtual void  SetY2NDC(Double_t y2) {fY2NDC=y2;}

        ClassDef(TPave,2)  //Pave. A box with shadowing
};

#endif

