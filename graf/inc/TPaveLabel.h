// @(#)root/graf:$Name$:$Id$
// Author: Rene Brun   17/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPaveLabel
#define ROOT_TPaveLabel


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPaveLabel                                                           //
//                                                                      //
// PaveLabel  A Pave with a label.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TPave
#include "TPave.h"
#endif
#ifndef ROOT_TAttText
#include "TAttText.h"
#endif


class TPaveLabel : public TPave, public TAttText {

protected:
        TString      fLabel;         //Label written at the center of Pave

public:
        TPaveLabel();
        TPaveLabel(Coord_t x1, Coord_t y1,Coord_t x2 ,Coord_t y2, const char *label, Option_t *option="br");
        TPaveLabel(const TPaveLabel &pavelabel);
        virtual ~TPaveLabel();
                void  Copy(TObject &pavelabel);
        virtual void  Draw(Option_t *option="");
        virtual void  DrawPaveLabel(Coord_t x1, Coord_t y1,Coord_t x2 ,Coord_t y2,
                      const char *label, Option_t *option="");
        const  char  *GetLabel() const {return fLabel.Data();}
        const  char  *GetTitle() const {return fLabel.Data();}
        virtual void  Paint(Option_t *option="");
        virtual void  PaintPaveLabel(Coord_t x1, Coord_t y1,Coord_t x2 ,Coord_t y2,
                      const char *label, Option_t *option="");
        virtual void  Print(Option_t *option="");
        virtual void  SavePrimitive(ofstream &out, Option_t *option);
        virtual void  SetLabel(const char *label) {fLabel = label;} // *MENU*

        ClassDef(TPaveLabel,1)  //PaveLabel. A Pave with a label
};

#endif

