// @(#)root/base:$Id$
// Author: Rene Brun   16/08/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMacro
#define ROOT_TMacro

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMacro                                                               //
//                                                                      //
// Class supporting a collection of lines with C++ code.                //
// A TMacro can be executed, saved to a ROOT file, edited, etc.         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"

class TList;
class TObjString;
class TMD5;


class TMacro : public TNamed {

protected:
   TList         *fLines;      //collection of lines
   TString        fParams;     //default string of macro parameters

   void           SaveSource(FILE *fp);

public:
   TMacro();
   TMacro(const TMacro&);
   TMacro(const char *name, const char *title="");
   virtual ~TMacro();
   TMacro& operator=(const TMacro&);
   virtual TObjString  *AddLine(const char *text);
   void                 Browse(TBrowser *b) override;
   virtual TMD5        *Checksum();
   virtual TObjString  *GetLineWith(const char *text) const;
   virtual Bool_t       Load() const; //*MENU*
   virtual Longptr_t    Exec(const char *params = nullptr, Int_t *error = nullptr); //*MENU*
   TList               *GetListOfLines() const {return fLines;}
   void                 Paint(Option_t *option="") override;
   void                 Print(Option_t *option="") const override;  //*MENU*
   virtual Int_t        ReadFile(const char *filename);
   virtual void         SaveSource(const char *filename);  //*MENU*
   void                 SavePrimitive(std::ostream &out, Option_t *option = "") override;
   virtual void         SetParams(const char *params = nullptr); //*MENU*

   ClassDefOverride(TMacro,1)  // Class supporting a collection of lines with C++ code.
};

#endif
