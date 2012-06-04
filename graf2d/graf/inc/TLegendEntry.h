// @(#)root/graf:$Id$
// Author: Matthew.Adam.Dobbs   06/09/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
//--------------------------------------------------------------------------

#ifndef ROOT_TLegendEntry
#define ROOT_TLegendEntry


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLegendEntry                                                         //
// Matthew.Adam.Dobbs@Cern.CH, September 1999                           //
// Storage class for one entry of a TLegend                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TAttText
#include "TAttText.h"
#endif
#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif
#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif
#ifndef ROOT_TAttMarker
#include "TAttMarker.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TLegendEntry : public TObject, public TAttText, public TAttLine,
                     public TAttFill, public TAttMarker {
public:
   TLegendEntry();
   TLegendEntry(const TObject *obj, const char *label = 0, Option_t *option="lpf" );
   TLegendEntry( const TLegendEntry &entry );
   virtual ~TLegendEntry();
   virtual void          Copy( TObject &obj ) const;
   virtual const char   *GetLabel() const { return fLabel.Data(); }
   virtual TObject      *GetObject() const { return fObject; }
   virtual Option_t     *GetOption() const { return fOption.Data(); }
   virtual void          Print( Option_t *option = "" ) const;
   virtual void          SaveEntry( std::ostream &out, const char *name );
   virtual void          SetLabel( const char *label = "" ) { fLabel = label; } // *MENU*
   virtual void          SetObject(TObject* obj );
   virtual void          SetObject( const char *objectName );  // *MENU*
   virtual void          SetOption( Option_t *option="lpf" ) { fOption = option; } // *MENU*

protected:
   TObject      *fObject;   // pointer to object being represented by this entry
   TString       fLabel;    // Text associated with the entry, will become latex
   TString       fOption;   // Options associated with this entry

private:
   TLegendEntry& operator=(const TLegendEntry&); // Not implemented

   ClassDef(TLegendEntry,1) // Storage class for one entry of a TLegend
};

#endif
