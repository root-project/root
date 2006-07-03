// @(#)root/graf:$Name:  $:$Id: TLegend.h,v 1.10 2006/05/23 04:47:37 brun Exp $
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


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLegend        (a second attempt- the first was TPadLegend           //
// Matthew.Adam.Dobbs@Cern.CH, September 1999                           //
// Legend of markers/lines/boxes for histos & graphs                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TPave
#include "TPave.h"
#endif
#ifndef ROOT_TAttText
#include "TAttText.h"
#endif

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
   virtual ~TLegend();
   TLegend( const TLegend &legend );

   TLegendEntry   *AddEntry(TObject* obj, const char* label = "", Option_t* option = "lpf" );
   TLegendEntry   *AddEntry(const char *name, const char* label = "", Option_t* option = "lpf" );
   virtual void    Clear( Option_t* option = "" ); // *MENU*
   virtual void    Copy( TObject &obj ) const;
   virtual void    DeleteEntry(); // *MENU*
   virtual void    Draw( Option_t* option = "" );
   virtual void    EditEntryAttFill();
   virtual void    EditEntryAttLine();
   virtual void    EditEntryAttMarker();
   virtual void    EditEntryAttText();
   TLegendEntry   *GetEntry() const;
   Float_t         GetEntrySeparation() const { return fEntrySeparation; }
   virtual const char *GetHeader() const;
   TList          *GetListOfPrimitives() const {return fPrimitives;}
   Float_t         GetMargin() const { return fMargin; }
   virtual void    InsertEntry( const char* objectName = "",const char* label = "",
                             Option_t* option = "lpf" ); // *MENU*
   virtual void    Paint( Option_t* option = "" );
   virtual void    PaintPrimitives();
   virtual void    Print( Option_t* option = "" ) const;
   virtual void    RecursiveRemove(TObject *obj);
   virtual void    SavePrimitive(ostream &out, Option_t *option  = "");
   void            SetDefaults() { fEntrySeparation = 0.1f; fMargin = 0.25f; }
   virtual void    SetEntryLabel( const char* label ); // *MENU*
   virtual void    SetEntryOption( Option_t* option ); // *MENU*
   void            SetEntrySeparation( Float_t entryseparation )
                     { fEntrySeparation = entryseparation; } // *MENU*
   virtual void    SetHeader( const char *header = "" );  // *MENU*
   void            SetMargin( Float_t margin ) { fMargin = margin; } // *MENU*

protected:
   TList     *fPrimitives;       // list of TLegendEntries
   Float_t    fEntrySeparation;  // separation between entries, as a fraction of
                                 // the space allocated to one entry.
                                 // Typical value is 0.1.
   Float_t    fMargin;           // fraction of total width used for symbol

   ClassDef(TLegend,1) // Legend of markers/lines/boxes to represent obj's
};

#endif
