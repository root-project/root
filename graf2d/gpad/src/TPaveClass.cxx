// @(#)root/gpad:$Id$
// Author: Rene Brun   06/08/99
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TROOT.h"
#include "Buttons.h"
#include "TPaveClass.h"

ClassImp(TPaveClass)


//______________________________________________________________________________
//  A PaveClass is a TPaveLabel  specialized to process classes
//  inside a TClassTree.
//   A TPaveClass object is used by the TClassTree to represent a class.
//   A TPaveClass has the same graphical representation as a TPaveLabel.
//   Using the context menu on can select additional options in the ClassTree:
//     - Show classes using this class
//     - Show all classes used by this class


//______________________________________________________________________________
TPaveClass::TPaveClass(): TPaveLabel()
{
   // PaveClass default constructor.

   fClassTree  = 0;
}


//______________________________________________________________________________
TPaveClass::TPaveClass(Double_t x1, Double_t y1,Double_t x2, Double_t  y2, const char *label, TClassTree *classtree)
           :TPaveLabel(x1,y1,x2,y2,label,"br")
{
   // PaveClass normal constructor.

   fClassTree  = classtree;
   SetName(label);
   SetTextFont(61);
}


//______________________________________________________________________________
TPaveClass::~TPaveClass()
{
   // PaveClass default destructor.
}


//______________________________________________________________________________
TPaveClass::TPaveClass(const TPaveClass &PaveClass) : TPaveLabel(PaveClass)
{
   // PaveClass copy constructor.

   ((TPaveClass&)PaveClass).Copy(*this);
}


//______________________________________________________________________________
void TPaveClass::Copy(TObject &obj) const
{
   // Copy this PaveClass to PaveClass.

   TPaveLabel::Copy(obj);
   ((TPaveClass&)obj).fClassTree      = fClassTree;
}


//______________________________________________________________________________
void TPaveClass::DrawClasses(const char *classes)
{
   // Draw classes.

   if (!fClassTree) return;
   if (!strcmp(classes,"this")) fClassTree->Draw(GetName());
   else                         fClassTree->Draw(classes);
}


//______________________________________________________________________________
void TPaveClass::SaveAs(const char *filename, Option_t *option) const
{
   // Save as.

   if (!fClassTree) return;
   fClassTree->SaveAs(filename,option);
}


//______________________________________________________________________________
void TPaveClass::SetClasses(const char *classes, Option_t *option)
{
   // Set classes.

   if (!fClassTree) return;
   if (!strcmp(classes,"this")) fClassTree->SetClasses(GetName(),option);
   else                         fClassTree->SetClasses(classes,option);
}


//______________________________________________________________________________
void TPaveClass::ShowLinks(Option_t *option)
{
   // Set link options in the ClassTree object.
   //
   //   "C"  show References from code
   //   "H"  show "Has a" relations
   //   "M"  show Multiple Inheritance
   //   "R"  show References from data members

   if (!fClassTree) return;
   fClassTree->ShowLinks(option);
}


//______________________________________________________________________________
void TPaveClass::ShowClassesUsedBy(const char *classes)
{
   // Show classes used by.

   if (!fClassTree) return;
   if (!strcmp(classes,"this")) fClassTree->ShowClassesUsedBy(GetName());
   else                         fClassTree->ShowClassesUsedBy(classes);
}


//______________________________________________________________________________
void TPaveClass::ShowClassesUsing(const char *classes)
{
   // Show classes using.

   if (!fClassTree) return;
   if (!strcmp(classes,"this")) fClassTree->ShowClassesUsing(GetName());
   else                         fClassTree->ShowClassesUsing(classes);
}


//______________________________________________________________________________
void TPaveClass::SavePrimitive(ostream &out, Option_t * /*= ""*/)
{
   // Save primitive as a C++ statement(s) on output stream out

   char quote = '"';
   out<<"   "<<endl;
   if (gROOT->ClassSaved(TPaveClass::Class())) {
      out<<"   ";
   } else {
      out<<"   TPaveClass *";
   }
   out<<"pclass = new TPaveClass("<<fX1<<","<<fY1<<","<<fX2<<","<<fY2
      <<","<<quote<<fLabel<<quote<<","<<quote<<fOption<<quote<<");"<<endl;

   SaveFillAttributes(out,"pclass",0,1001);
   SaveLineAttributes(out,"pclass",1,1,1);
   SaveTextAttributes(out,"pclass",22,0,1,62,0);

   out<<"   pclass->Draw();"<<endl;
}
