// @(#)root/graf:$Name$:$Id$
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
 public:
  TLegend();
  TLegend( Coord_t x1, Coord_t y1, Coord_t x2, Coord_t y2,
           const char* header = "", Option_t* option="brNDC" );
  virtual ~TLegend();
  TLegend( const TLegend &legend );
  TLegendEntry   *AddEntry(TObject* obj, const char* label = "", Option_t* option = "lpf" );
  TLegendEntry   *AddEntry(const char *name, const char* label = "", Option_t* option = "lpf" );
  virtual void    Clear( Option_t* option = "" ); // *MENU*
  virtual void    Copy( TObject &obj );
  virtual void    DeleteEntry(); // *MENU*
  virtual void    Draw( Option_t* option = "" );
  virtual void    EditEntryAttFill(); // *MENU*
  virtual void    EditEntryAttLine(); // *MENU*
  virtual void    EditEntryAttMarker(); // *MENU*
  virtual void    EditEntryAttText(); // *MENU*
  TLegendEntry   *GetEntry();
  Float_t         GetEntrySeparation() { return fEntrySeparation; }
  virtual const char *GetHeader();
  TList          *GetListOfPrimitives() {return fPrimitives;}
  Float_t         GetMargin() { return fMargin; }
  virtual void    InsertEntry( const char* objectName = "",const char* label = "",
                            Option_t* option = "lpf" ); // *MENU*
  virtual void    Paint( Option_t* option = "" );
  virtual void    PaintPrimitives();
  virtual void    Print( Option_t* option = "" );
  virtual void    SavePrimitive(ofstream &out, Option_t *option );
  void            SetDefaults() { fEntrySeparation = 0.1; fMargin = 0.25; }
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
