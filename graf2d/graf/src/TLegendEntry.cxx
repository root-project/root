// @(#)root/graf:$Id$
// Author: Matthew.Adam.Dobbs   06/09/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdio.h>

#include "TLegendEntry.h"
#include "TVirtualPad.h"
#include "TROOT.h"
#include "Riostream.h"


ClassImp(TLegendEntry)


//____________________________________________________________________________
// TLegendEntry   Matthew.Adam.Dobbs@Cern.CH, September 1999
// Storage class for one entry of a TLegend
//


//____________________________________________________________________________
TLegendEntry::TLegendEntry(): TAttText(), TAttLine(), TAttFill(), TAttMarker()
{
   // TLegendEntry do-nothing default constructor

   fObject = 0;
}


//____________________________________________________________________________
TLegendEntry::TLegendEntry(const TObject* obj, const char* label, Option_t* option )
             :TAttText(0,0,0,0,0), TAttLine(1,1,1), TAttFill(0,0), TAttMarker(1,21,1)
{
   // TLegendEntry normal constructor for one entry in a TLegend
   //   obj is the object this entry will represent. If obj has
   //   line/fill/marker attributes, then the TLegendEntry will display
   //   these attributes.
   //   label is the text that will describe the entry, it is displayed using
   //   TLatex, so may have a complex format.
   //   option may have values
   //    L draw line associated w/ TAttLine if obj inherits from TAttLine
   //    P draw polymarker assoc. w/ TAttMarker if obj inherits from TAttMarker
   //    F draw a box with fill associated w/ TAttFill if obj inherits TAttFill
   //   default is object = "LPF"

   fObject = 0;
   if ( !label && obj ) fLabel = obj->GetTitle();
   else                 fLabel = label;
   fOption = option;
   if (obj) SetObject((TObject*)obj);
}


//____________________________________________________________________________
TLegendEntry::TLegendEntry( const TLegendEntry &entry ) : TObject(entry), TAttText(entry), TAttLine(entry), TAttFill(entry), TAttMarker(entry)
{
   // TLegendEntry copy constructor

   ((TLegendEntry&)entry).Copy(*this);
}


//____________________________________________________________________________
TLegendEntry::~TLegendEntry()
{
   // TLegendEntry default destructor

   fObject = 0;
}


//____________________________________________________________________________
void TLegendEntry::Copy( TObject &obj ) const
{
   // copy this TLegendEntry into obj

   TObject::Copy(obj);
   TAttText::Copy((TLegendEntry&)obj);
   TAttLine::Copy((TLegendEntry&)obj);
   TAttFill::Copy((TLegendEntry&)obj);
   TAttMarker::Copy((TLegendEntry&)obj);
   ((TLegendEntry&)obj).fObject = fObject;
   ((TLegendEntry&)obj).fLabel = fLabel;
   ((TLegendEntry&)obj).fOption = fOption;
}


//____________________________________________________________________________
void TLegendEntry::Print( Option_t *) const
{
   // dump this TLegendEntry to std::cout

   TString output;
   std::cout << "TLegendEntry: Object ";
   if ( fObject ) output = fObject->GetName();
   else output = "NULL";
   std::cout << output << " Label ";
   if ( fLabel ) output = fLabel.Data();
   else output = "NULL";
   std::cout << output << " Option ";
   if (fOption ) output = fOption.Data();
   else output = "NULL";
   std::cout << output << std::endl;
}


//____________________________________________________________________________
void TLegendEntry::SaveEntry(std::ostream &out, const char* name )
{
   // Save this TLegendEntry as C++ statements on output stream out
   //  to be used with the SaveAs .C option

   char quote = '"';
   if ( gROOT->ClassSaved( TLegendEntry::Class() ) ) {
      out << "   entry=";
   } else {
      out << "   TLegendEntry *entry=";
   }
   TString objname = "NULL";
   if ( fObject ) objname = fObject->GetName();
   out << name << "->AddEntry("<<quote<<objname<<quote<<","<<quote<<
      fLabel.Data()<<quote<<","<<quote<<fOption.Data()<<quote<<");"<<std::endl;
   SaveFillAttributes(out,"entry",0,0);
   SaveLineAttributes(out,"entry",0,0,0);
   SaveMarkerAttributes(out,"entry",0,0,0);
   SaveTextAttributes(out,"entry",0,0,0,0,0);
}


//____________________________________________________________________________
void TLegendEntry::SetObject(TObject* obj )
{
   // (re)set the obj pointed to by this entry

   if ( ( fObject && fLabel == fObject->GetTitle() ) || !fLabel ) {
      if (obj) fLabel = obj->GetTitle();
   }
   fObject = obj;
}


//____________________________________________________________________________
void TLegendEntry::SetObject( const char* objectName)
{
   // (re)set the obj pointed to by this entry

   TObject* obj = 0;
   TList* padprimitives = gPad->GetListOfPrimitives();
   if (padprimitives) obj = padprimitives->FindObject( objectName );
   if (obj) SetObject( obj );
}
