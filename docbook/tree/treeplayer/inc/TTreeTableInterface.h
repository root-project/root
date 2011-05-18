// Author: Roel Aaij   21/07/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeTableInterface
#define ROOT_TTreeTableInterface

#ifndef ROOT_TVirtualTableInterface
#include "TVirtualTableInterface.h"
#endif

class TTree;
class TObjArray;
class TTreeFormula;
class TTreeFormulaManager;
class TSelectorDraw;
class TEntryList;
class TList;


class TTreeTableInterface : public TVirtualTableInterface {

private:
   TTree               *fTree;       // Data in a TTree
   TList               *fFormulas;   // Array of TTreeFormulas to display values
   Long64_t             fEntry;      // Present entry number in fTree.
   Long64_t             fNEntries;   // Number of entries in the tree.
   Long64_t             fFirstEntry; // First entry.
   TTreeFormulaManager *fManager;    // Coordinator for the formulas.
   TTreeFormula        *fSelect;     // Selection condition
   TSelectorDraw       *fSelector;   // Selector
   TList               *fInput;      // Used for fSelector.
   Bool_t               fForceDim;   // Force dimension.
   TEntryList          *fEntries;    // Currently active entries
   UInt_t               fNRows;      // Amount of rows in the data
   UInt_t               fNColumns;   // Amount of columns in the data

   void SetVariablesExpression(const char *varexp);
   void SyncFormulas();
   void InitEntries();

protected:

public:
   TTreeTableInterface(TTree *tree = 0, const char *varexp = 0,
                       const char *selection = 0, Option_t *option = 0,
                       Long64_t nentries = 0, Long64_t firstentry = 0);
   virtual ~TTreeTableInterface();

   virtual Double_t    GetValue(UInt_t row, UInt_t column);
   virtual const char *GetValueAsString(UInt_t row, UInt_t column);
   virtual const char *GetRowHeader(UInt_t row);
   virtual const char *GetColumnHeader(UInt_t column);
   virtual UInt_t      GetNRows();
   virtual UInt_t      GetNColumns();
   virtual TEntryList *GetEntryList() { return fEntries; }

   virtual void AddColumn(const char *expression, UInt_t position);
   virtual void AddColumn(TTreeFormula *formula, UInt_t position);
   virtual void RemoveColumn(UInt_t position);
   virtual void SetFormula(TTreeFormula *formula, UInt_t position);
   virtual void SetSelection(const char *selection);
   virtual void SetEntryList(TEntryList *entrylist = 0);

   ClassDef(TTreeTableInterface, 0) // Interface to data in a TTree
};

#endif
