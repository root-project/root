// @(#)root/graf:$Name$:$Id$
// Author: Rene Brun   20/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPaveText
#define ROOT_TPaveText


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPaveText                                                            //
//                                                                      //
// PaveText   A Pave with several lines of text.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TPave
#include "TPave.h"
#endif
#ifndef ROOT_TText
#include "TText.h"
#endif
#ifndef ROOT_TLine
#include "TLine.h"
#endif


class TPaveText : public TPave, public TAttText {

protected:
        TString    fLabel;          //Label written at the top of the pavetext
        Int_t      fLongest;        //Length of the longest line
        Float_t    fMargin;         //Text margin
        TList      *fLines;         //List of labels

public:
        TPaveText();
        TPaveText(Coord_t x1, Coord_t y1,Coord_t x2 ,Coord_t y2, Option_t *option="br");
        TPaveText(const TPaveText &pavetext);
        virtual ~TPaveText();
        virtual TBox    *AddBox(Float_t x1, Float_t y1, Float_t x2, Float_t y2);
        virtual TLine   *AddLine(Float_t x1=0, Float_t y1=0, Float_t x2=0, Float_t y2=0);
        virtual TText   *AddText(Float_t x1, Float_t y1, const char *label);
        virtual TText   *AddText(const char *label);
        virtual void     Clear(Option_t *option="");  // *MENU*
                void     Copy(TObject &pavetext);
        virtual void     DeleteText(); // *MENU*
        virtual void     Draw(Option_t *option="");
        virtual void     DrawFile(const char *filename, Option_t *option="");
        virtual void     EditText(); // *MENU*
        const  char     *GetLabel() const {return fLabel.Data();}
        virtual TText   *GetLine(Int_t number);
        virtual TText   *GetLineWith(const char *text);
        virtual TList   *GetListOfLines() {return fLines;}
              Float_t    GetMargin() {return fMargin;}
        virtual TObject *GetObject(Float_t &ymouse, Float_t &yobj);
        virtual Int_t    GetSize();
        virtual void     InsertLine(); // *MENU*
        virtual void     InsertText(const char *label); // *MENU*
        virtual void     Paint(Option_t *option="");
        virtual void     PaintPrimitives(Int_t mode);
        virtual void     Print(Option_t *option="");
        virtual void     ReadFile(const char *filename, Option_t *option="", Int_t nlines=50, Int_t fromline=0); // *MENU*
        virtual void     SaveLines(ofstream &out, const char *name);
        virtual void     SavePrimitive(ofstream &out, Option_t *option);
        virtual void     SetAllWith(const char *text, Option_t *option, Float_t value); // *MENU*
        virtual void     SetLabel(const char *label) {fLabel = label;} // *MENU*
        virtual void     SetMargin(Float_t margin=0.05) {fMargin=margin;} // *MENU*

        ClassDef(TPaveText,2)  //PaveText. A Pave with several lines of text.
};

#endif

