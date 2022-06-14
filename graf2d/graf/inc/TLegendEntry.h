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


#include "TObject.h"
#include "TAttText.h"
#include "TAttLine.h"
#include "TAttFill.h"
#include "TAttMarker.h"
#include "TString.h"

class TLegendEntry : public TObject, public TAttText, public TAttLine,
                     public TAttFill, public TAttMarker {
public:
   TLegendEntry();
   TLegendEntry(const TObject *obj, const char *label = nullptr, Option_t *option="lpf" );
   TLegendEntry( const TLegendEntry &entry );
   virtual ~TLegendEntry();
   void                  Copy( TObject &obj ) const override;
   virtual const char   *GetLabel() const { return fLabel.Data(); }
   virtual TObject      *GetObject() const { return fObject; }
   Option_t             *GetOption() const override { return fOption.Data(); }
   void                  Print( Option_t *option = "" ) const override;
   virtual void          SaveEntry( std::ostream &out, const char *name );
   virtual void          SetLabel( const char *label = "" ) { fLabel = label; } // *MENU*
   virtual void          SetObject(TObject* obj );
   virtual void          SetObject( const char *objectName );  // *MENU*
   virtual void          SetOption( Option_t *option="lpf" ) { fOption = option; } // *MENU*

protected:
   TObject      *fObject;   ///< pointer to object being represented by this entry
   TString       fLabel;    ///< Text associated with the entry, will become latex
   TString       fOption;   ///< Options associated with this entry

private:
   TLegendEntry& operator=(const TLegendEntry&) = delete;

   ClassDefOverride(TLegendEntry,1) // Storage class for one entry of a TLegend
};

#endif
