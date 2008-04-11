// @(#)root/gpad:$Id$
// Author: Rene Brun   06/08/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TPaveClass
#define ROOT_TPaveClass


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPaveClass                                                           //
//                                                                      //
// A TPaveLabel specialized for TClassTree objects                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TPaveLabel
#include "TPaveLabel.h"
#endif
#ifndef ROOT_TTreeViewer
#include "TClassTree.h"
#endif


class TPaveClass : public TPaveLabel{

protected:
   TClassTree   *fClassTree;       //Pointer to the TClassTree referencing this object

public:
   TPaveClass();
   TPaveClass(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2, const char *label, TClassTree *classtree);
   TPaveClass(const TPaveClass &PaveVar);
   virtual      ~TPaveClass();

   void          Copy(TObject &PaveVar) const;
   virtual void  DrawClasses(const char *classes="this");   // *MENU*
   TClassTree   *GetClassTree() const {return fClassTree;}
   virtual void  SaveAs(const char *filename="",Option_t *option="") const; // *MENU*
   virtual void  SavePrimitive(ostream &out, Option_t *option = "");
   virtual void  SetClasses(const char *classes="this", Option_t *option="ID");   // *MENU*
   virtual void  ShowClassesUsedBy(const char *classes="this");  // *MENU*
   virtual void  ShowClassesUsing(const char *classes="this");   // *MENU*
   virtual void  SetClassTree(TClassTree *classtree) {fClassTree = classtree;}
   virtual void  ShowLinks(Option_t *option="HMR"); // *MENU*

   ClassDef(TPaveClass,1)  //A TPaveLabel specialized for TClassTree objects
};

#endif
