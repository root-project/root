// Author: Roel Aaij   21/07/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTable
#define ROOT_TGTable

#include "TGCanvas.h"
#include "TGWidget.h"
#include "TGTableHeader.h"

class TGWindow;
class TGString;
class TGToolTip;
class TGPicture;
class TVirtualTableInterface;
class TGTableCell;
class TGTableHeader;
class TGToolTip;
class TGTableFrame;
class TGTableHeaderFrame;
class TGTextButton;
class TGNumberEntryField;
class TGLabel;
class TGTextEntry;
class TTableRange;

class TGTable : public TGCompositeFrame, public TGWidget {

protected:
   TObjArray     *fRows;          ///< Array of rows
   TObjArray     *fRowHeaders;    ///< Array of row headers
   TObjArray     *fColumnHeaders; ///< Array of column headers
   TGTableHeader *fTableHeader;   ///< Top left element of the table
   Bool_t         fReadOnly;      ///< Table readonly state
   Pixel_t        fSelectColor;   ///< Select Color
   Int_t          fTMode;         ///< Text justify mode
   Bool_t         fAllData;       ///< Is the data bigger than the table
   TTableRange   *fCurrentRange;  ///< Range of data currently loaded
   TTableRange   *fDataRange;     ///< Full range of the data set
   TTableRange   *fGotoRange;     ///< Range used by Goto frame
   TGTableFrame  *fTableFrame;    ///< Container for the frames
   TGCanvas      *fCanvas;        ///< Canvas that will contains the cells
   UInt_t         fCellWidth;     ///< Default cell width
   UInt_t         fCellHeight;    ///< Default cell width

   ///@{
   ///@name Frames used for layout
   TGTableHeaderFrame *fCHdrFrame;     ///< Frame that contains the row headers
   TGTableHeaderFrame *fRHdrFrame;     ///< Frame that contains the row headers
   TGHorizontalFrame  *fRangeFrame;    ///< Frame that contains the top part
   TGHorizontalFrame  *fTopFrame;      ///< Frame that contains the top part
   TGHorizontalFrame  *fTopExtraFrame; ///< Dev idea
   TGHorizontalFrame  *fBottomFrame;   ///< Frame that contains the bottom part
   TGHorizontalFrame  *fButtonFrame;   ///< Contains the buttons
   ///@}

   ///@{
   ///@name Buttons for interaction
   TGTextButton *fNextButton;     ///< Button to view next chunk
   TGTextButton *fPrevButton;     ///< Button to view previous chunk
   TGTextButton *fUpdateButton;   ///< Button to update current view
   TGTextButton *fGotoButton;     ///< Button to goto a new range
   ///@}

   ///@{
   ///@name Labels and text entries for range information and input
   TGLabel     *fFirstCellLabel;  ///< Label for the range frame
   TGLabel     *fRangeLabel;      ///< Label for the range frame
   TGTextEntry *fFirstCellEntry;  ///< TextEntry for the range frame
   TGTextEntry *fRangeEntry;      ///< TextEntry for the range frame

   Pixel_t fOddRowBackground;     ///< Background color for odd numbered rows
   Pixel_t fEvenRowBackground;    ///< Background color for even numbered rows
   Pixel_t fHeaderBackground;     ///< Background color for headers
   ///@}

   // Those are neither used nor even initialized:
   // static const TGGC *fgDefaultSelectGC; // Default select GC
   // static const TGGC *fgDefaultBckgndGC; // Default cell background GC
   // static const Int_t fgDefaultTMode;    // Default text justify mode

   ///@{
   ///@name Data members to keep track of LayoutHints that can't be automatically cleaned
   TList *fCellHintsList;
   TList *fRHdrHintsList;
   TList *fCHdrHintsList;
   TList *fMainHintsList;   ///< List for all hints used in the main table frame
   ///@}

   // Add rows and/or columns to the edge of the table.

   virtual void Init();

   // Remove rows and/or columns from the edge of the table.
protected:
   TVirtualTableInterface *fInterface; // Interface to the data source

   void DoRedraw() override;

   virtual void Expand(UInt_t nrows, UInt_t ncolumns);
   virtual void ExpandColumns(UInt_t ncolumns);
   virtual void ExpandRows(UInt_t nrows);

   virtual UInt_t GetRHdrHeight() const;
   virtual UInt_t GetCHdrWidth() const;

   virtual void Shrink(UInt_t nrows, UInt_t ncolumns);
   virtual void ShrinkColumns(UInt_t ncolumns);
   virtual void ShrinkRows(UInt_t nrows);

   virtual void UpdateHeaders(EHeaderType type);
   virtual void SetInterface(TVirtualTableInterface *interface,
                             UInt_t nrows = 50, UInt_t ncolumns = 20);
   virtual void ResizeTable(UInt_t nrows, UInt_t ncolumns);

   virtual void UpdateRangeFrame();

public:
   TGTable(const TGWindow *p = nullptr, Int_t id = 0,
           TVirtualTableInterface *interface = 0, UInt_t nrows = 50,
           UInt_t ncolumns = 20);
   virtual ~TGTable();

   virtual TObjArray *GetRow(UInt_t row);
   virtual TObjArray *GetColumn(UInt_t columns);

//    // Selection
//    virtual void Select(TGTableCell *celltl, TGTableCell *cellbr);
//    virtual void Select(UInt_t xcell1, UInt_t ycell1, UInt_t xcell2, UInt_t ycell2);
//    virtual void SelectAll();
//    virtual void SelectRow(TGTableCell *cell);
//    virtual void SelectRow(UInt_t row);
//    virtual void SelectRows(UInt_t row, UInt_t nrows);
//    virtual void SelectColumn(TGTableCell *cell);
//    virtual void SelectColumn(UInt_t column);
//    virtual void SelectColumns(UInt_t column, UInt_t ncolumns);

//    virtual void SetSelectGC(TGGC *gc);
//    virtual void SetTextJustify(Int_t tmode);

   // Cells
   virtual const TGTableCell* GetCell(UInt_t i, UInt_t j) const;
   virtual TGTableCell* GetCell(UInt_t i, UInt_t j);

   virtual const TGTableCell* FindCell(TGString label) const;
   virtual TGTableCell* FindCell(TGString label);

   virtual void Show();

   // Because insertion and removal of columns in the middle of a data
   // set is not yet supported in this design iteration, these methods
   // have been commented out.

//    // Insert a range of columns or rows, if the label is empty, a
//    // default scheme will be used.
//    virtual void InsertRowBefore(UInt_t row, UInt_t nrows);
//    virtual void InsertRowBefore(TGString label, UInt_t nrows);
//    virtual void InsertRowAfter(UInt_t row, UInt_t nrows);
//    virtual void InsertRowAfter(TGString label, UInt_t nrows);
//    virtual void InsertRowAt(UInt_t row, UInt_t nrows = 1);
//    virtual void InsertRowAt(TGString label, UInt_t nrows);

//    virtual void InsertColumnBefore(UInt_t column, UInt_t ncolumns);
//    virtual void InsertColumnBefore(TGString label, UInt_t ncolumns);
//    virtual void InsertColumnAfter(UInt_t column, UInt_t ncolumns);
//    virtual void InsertColumnAfter(TGString label, UInt_t ncolumns);
//    virtual void InsertColumnAt(UInt_t column, UInt_t ncolumns = 1);
//    virtual void InsertColumnAt(TGString label, UInt_t ncolumns);

//    // Remove rows or columns.
//    virtual void RemoveRows(UInt_t row, UInt_t nrows = 1);
//    virtual void RemoveColumns(UInt_t column, UInt_t ncolumns = 1);

   // Update view
   virtual void UpdateView();

   // Getters
   virtual UInt_t       GetNTableRows() const;
   virtual UInt_t       GetNDataRows() const;
   virtual UInt_t       GetNTableColumns() const;
   virtual UInt_t       GetNDataColumns() const;
   virtual UInt_t       GetNTableCells() const;
   virtual UInt_t       GetNDataCells() const;
   virtual const  TTableRange *GetCurrentRange() const;

   virtual TVirtualTableInterface *GetInterface() { return fInterface; }

   virtual TGCanvas                 *GetCanvas() { return fCanvas; }
   virtual const TGTableHeaderFrame *GetRHdrFrame() { return fRHdrFrame; }
   virtual const TGTableHeaderFrame *GetCHdrFrame() { return fCHdrFrame; }
   virtual const TGTableHeader      *GetRowHeader(const UInt_t row) const;
   virtual TGTableHeader            *GetRowHeader(const UInt_t row);
   virtual const TGTableHeader      *GetColumnHeader(const UInt_t column) const;
   virtual TGTableHeader            *GetColumnHeader(const UInt_t column);
   virtual TGTableHeader            *GetTableHeader();

//    virtual const TGGC*  GetSelectGC() const;
//    virtual const TGGC*  GetCellBckgndGC(TGTableCell *cell) const;
//    virtual const TGGC*  GetCellBckgndGC(UInt_t row, UInt_t column) const;

   virtual Pixel_t GetRowBackground(UInt_t row) const;
   virtual Pixel_t GetHeaderBackground() const ;

   virtual void SetOddRowBackground(Pixel_t pixel);
   virtual void SetEvenRowBackground(Pixel_t pixel);
   virtual void SetHeaderBackground(Pixel_t pixel);
   virtual void SetDefaultColors();

   // Range manipulators
   virtual void MoveTable(Int_t rows, Int_t columns);
   virtual void GotoTableRange(Int_t xtl, Int_t ytl,
                               Int_t xbr, Int_t ybr);
   // Operators
   virtual TGTableCell* operator() (UInt_t row, UInt_t column);

   // Internal slots
   virtual void ScrollCHeaders(Int_t xpos);
   virtual void ScrollRHeaders(Int_t ypos);
   virtual void NextChunk();
   virtual void PreviousChunk();
   virtual void UserRangeChange();
   virtual void Goto();
   virtual void Update();

   ClassDefOverride(TGTable, 0) // A table used to visualize data from different sources.
};

class TTableRange {
public:
   UInt_t fXtl; ///< Top left X coordinate
   UInt_t fYtl; ///< Top left Y coordinate
   UInt_t fXbr; ///< Bottom right X coordinate
   UInt_t fYbr; ///< Bottom right Y coordinate

   TTableRange();
   virtual ~TTableRange() {}
   virtual void Print();

   Bool_t operator==(TTableRange &other);

   ClassDef(TTableRange, 0) // Range used in TGTable.
};

#endif

