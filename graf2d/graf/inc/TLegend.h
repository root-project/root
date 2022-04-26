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
#ifndef ROOT_TLegend
#define ROOT_TLegend


#include "TPave.h"
#include "TAttText.h"

class TObject;
class TList;
class TLegendEntry;

class TLegend : public TPave , public TAttText {

protected:
   TLegend& operator=(const TLegend&);

public:
   TLegend();
   TLegend( Double_t x1, Double_t y1, Double_t x2, Double_t y2,
            const char* header = "", Option_t* option="brNDC" );
   TLegend( Double_t w, Double_t h, const char* header = "", Option_t* option="brNDC" );
   virtual ~TLegend();
   TLegend(const TLegend &legend);

   TLegendEntry   *AddEntry(const TObject* obj, const char* label = "", Option_t* option = "lpf" );
   TLegendEntry   *AddEntry(const char *name, const char* label = "", Option_t* option = "lpf" );
   void            Clear( Option_t* option = "" ) override; // *MENU*
   void            Copy( TObject &obj ) const override;
   virtual void    DeleteEntry(); // *MENU*
   void            Draw( Option_t* option = "" ) override;
   virtual void    EditEntryAttFill();
   virtual void    EditEntryAttLine();
   virtual void    EditEntryAttMarker();
   virtual void    EditEntryAttText();
   Float_t         GetColumnSeparation() const { return fColumnSeparation; }
   TLegendEntry   *GetEntry() const;
   Float_t         GetEntrySeparation() const { return fEntrySeparation; }
   virtual const char *GetHeader() const;
   TList          *GetListOfPrimitives() const {return fPrimitives;}
   Float_t         GetMargin() const { return fMargin; }
   Int_t           GetNColumns() const { return fNColumns; }
   Int_t           GetNRows() const;
   virtual void    InsertEntry( const char* objectName = "",const char* label = "",
                             Option_t* option = "lpf" ); // *MENU*
   void            Paint( Option_t* option = "" ) override;
   virtual void    PaintPrimitives();
   void            Print( Option_t* option = "" ) const override;
   void            RecursiveRemove(TObject *obj) override;
   void            SavePrimitive(std::ostream &out, Option_t *option  = "") override;
   void            SetDefaults() { fEntrySeparation = 0.1f; fMargin = 0.25f; fNColumns = 1; fColumnSeparation = 0.0f; }
   void            SetColumnSeparation( Float_t columnSeparation )
                     { fColumnSeparation = columnSeparation; } // *MENU*
   virtual void    SetEntryLabel( const char* label ); // *MENU*
   virtual void    SetEntryOption( Option_t* option ); // *MENU*
   void            SetEntrySeparation( Float_t entryseparation )
                     { fEntrySeparation = entryseparation; } // *MENU*
   virtual void    SetHeader( const char *header = "", Option_t *option = "" );  // *MENU*
   void            SetMargin( Float_t margin ) { fMargin = margin; } // *MENU*
   void            SetNColumns( Int_t nColumns ); // *MENU*

protected:
   TList     *fPrimitives;       ///< List of TLegendEntries
   Float_t    fEntrySeparation;  ///< Separation between entries, as a fraction of
                                 ///< The space allocated to one entry.
                                 ///< Typical value is 0.1.
   Float_t    fMargin;           ///< Fraction of total width used for symbol
   Int_t      fNColumns;         ///< Number of columns in the legend
   Float_t    fColumnSeparation; ///< Separation between columns, as a fraction of
                                 ///< The space allowed to one column

   ClassDefOverride(TLegend,3) // Legend of markers/lines/boxes to represent obj's
};

#endif
