// Author: Roel Aaij 15/08/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TClass.h"
#include "Riostream.h"
#include "TSystem.h"
#include "TEnv.h"
#include "TGResourcePool.h"
#include "TTreeTableInterface.h"
#include "TTreeFormula.h"
#include "TError.h"
#include "TTree.h"
#include "TEntryList.h"
#include "TSelectorDraw.h"
#include "TTreeFormulaManager.h"
#include "TTreeFormula.h"

ClassImp(TTreeTableInterface)

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeTableInterface.                                                 //
//                                                                      //
// TTreeTableInterface is used to interface to data that is stored in a //
// TTree. When the interface is created, an expression can be           //
// specified. This expression will define the columns to be shown.      //
//                                                                      //
// A selection criterium can also be specified. A TEntryList will be    //
// created and applied to the TTree using this criterium.               //
// a new TEntryList to use can be specified using SetEntryList.         //
// TGTable->Update() will need to be called to show the effects.        //
//                                                                      //
// WARNING: Do not apply an entrylist to the tree in any other way than //
// through the interface, this will have undefined consequences.        //
//                                                                      //
// Columns can be manipulated using the appropriate methods. A          //
// TGTable->Update is always needed afterwards to make the table aware  //
// of the changes.                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TTreeTableInterface::TTreeTableInterface (TTree *tree, const char *varexp,
   const char *selection, Option_t *option, Long64_t nentries,
   Long64_t firstentry)
   : TVirtualTableInterface(), fTree(tree), fFormulas(0), fEntry(0),
     fNEntries(nentries), fFirstEntry(firstentry), fManager(0), fSelect(0), fSelector(0), fInput(0),
     fForceDim(kFALSE), fEntries(0), fNRows(0), fNColumns(0)
{
   // TTreeTableInterface constructor.

   if (fTree == 0) {
      Error("TTreeTableInterface", "No tree supplied");
      return;
   }

   fFormulas= new TList();
   fSelector = new TSelectorDraw();
   fInput = new TList();
   fInput->Add(new TNamed("varexp",""));
   fInput->Add(new TNamed("selection",""));
   fSelector->SetInputList(fInput);
   fEntry=fFirstEntry;

   TString opt = option;

   if (nentries == 0) {
      fNEntries = fTree->GetEntries();
      Info("TTreeTableInterface", "nentries was 0, setting to maximum number"
           " available in the tree");
   }

   // Do stuff with opt.Contains() and options
   SetVariablesExpression(varexp);
   SetSelection(selection);

   if (fNRows == 0) {
      Warning ("TTreeTableInterface::TTreeTableInterface", "nrows = 0");
   }
   if (fNColumns == 0) {
      Warning ("TTreeTableInterface::TTreeTableInterface", "ncolumns = 0");
   }
}

//______________________________________________________________________________
TTreeTableInterface::~TTreeTableInterface()
{
   // TTreeTableInterface destructor.

   fFormulas->Delete();
   delete fFormulas;
   delete fInput;
   delete fSelector;

   if (fTree) fTree->SetEntryList(0);
   delete fEntries;
}

//______________________________________________________________________________
void TTreeTableInterface::SetVariablesExpression(const char *varexp)
{
   // Compile the variables expression from the given varexp.

   // FIXME check if enough protection against wrong expressions is in place

   Bool_t allvar = kFALSE;

   if (varexp) {
      if (!strcmp(varexp, "*")) { allvar = kTRUE; }
   } else {
      // if varexp is empty, take all available leaves as a column
      allvar = kTRUE;
   }

   if (allvar) {
      TObjArray *leaves = fTree->GetListOfLeaves();
      UInt_t nleaves = leaves->GetEntries();
      if (!nleaves) {
         Error("TTreeTableInterface", "No leaves in Tree");
         return;
      }
      fNColumns = nleaves;
      for (UInt_t ui = 0; ui < fNColumns; ui++) {
         TLeaf *lf = (TLeaf*)leaves->At(ui);
         fFormulas->Add(new TTreeFormula("Var1", lf->GetName(), fTree));
      }
      // otherwise select only the specified columns
   } else {
      std::vector<TString> cnames;
      fNColumns = fSelector->SplitNames(varexp,cnames);

      // Create the TreeFormula objects corresponding to each column
      for (UInt_t ui = 0; ui < fNColumns; ui++) {
         fFormulas->Add(new TTreeFormula("Var1", cnames[ui].Data(), fTree));
      }
   }
}

//______________________________________________________________________________
void TTreeTableInterface::SetSelection(const char *selection)
{
   // Set the selection expression.

   // FIXME verify functionality
   if (fSelect) {
      fFormulas->Remove(fSelect);
      delete fSelect;
      fSelect = 0;
   }
   if (selection && strlen(selection)) {
      fSelect = new TTreeFormula("Selection", selection, fTree);
      fFormulas->Add(fSelect);
   }

   if (fManager) {
      for (Int_t i = 0; i <= fFormulas->LastIndex(); i++) {
         fManager->Remove((TTreeFormula*)fFormulas->At(i));
      }
   }

   // SyncFormulas() will update the formula manager if needed
   SyncFormulas();
   InitEntries();
}

//______________________________________________________________________________
void TTreeTableInterface::SyncFormulas()
{
   // Sync the formulas using the TTreeFormulaManager.

   // FIXME verify functionality

   Int_t i = 0;
   if (fFormulas->LastIndex() >= 0) {
      if (fSelect) {
         if (fSelect->GetManager()->GetMultiplicity() > 0 ) {
            if (!fManager) fManager = new TTreeFormulaManager;
            for (i = 0; i <= fFormulas->LastIndex(); i++) {
               fManager->Add((TTreeFormula*)fFormulas->At(i));
            }
            fManager->Sync();
         }
      }
      for (i = 0; i < fFormulas->LastIndex(); i++) {
         TTreeFormula *form = ((TTreeFormula*)fFormulas->At(i));
         switch (form->GetManager()->GetMultiplicity()) {
            case  1:
            case  2:
            case -1:
               fForceDim = kTRUE;
               break;
            case  0:
               break;
         }
      }
   }
}

//______________________________________________________________________________
void TTreeTableInterface::InitEntries()
{
   // Initialise the TEntryList with the entries that match the
   // selection criterium.

   TEntryList *entrylist = new TEntryList(fTree);

   UInt_t ui = 0;
   Int_t i = 0;

   Long64_t notSkipped = 0;
   Int_t tnumber = -1;
   Long64_t entry = fFirstEntry;
   Int_t entriesToDisplay = fNEntries;

   while (entriesToDisplay != 0){
//       entryNumber = fTree->GetEntryNumber(entry);
//       if(entryNumber < 0) break;
      Long64_t localEntry = fTree->LoadTree(entry);
      if (localEntry < 0) break;
      if (tnumber != fTree->GetTreeNumber()) {
         tnumber = fTree->GetTreeNumber();
         if (fManager) fManager->UpdateFormulaLeaves();
         else {
            for(i = 0; i < fFormulas->LastIndex(); i++)
               ((TTreeFormula*)fFormulas->At(ui))->UpdateFormulaLeaves();
         }
      }
      Int_t ndata = 1;
      if (fForceDim){
         if (fManager)
            ndata = fManager->GetNdata(kTRUE);
         else {
            for (ui = 0; ui < fNColumns; ui++){
               if (ndata < ((TTreeFormula*)fFormulas->At(ui))->GetNdata())
                   ndata = ((TTreeFormula*)fFormulas->At(ui))->GetNdata();
            }
            if (fSelect && fSelect->GetNdata() == 0)
               ndata = 0;
         }
      }
      Bool_t skip = kFALSE;

      // Loop over the instances of the selection condition
      for (Int_t inst = 0; inst < ndata; inst++){
         if (fSelect){
            if (fSelect->EvalInstance(inst) == 0){
               skip = kTRUE;
               entry++;
            }
         }
      }
      if (!skip){
         entrylist->Enter(entry);
         notSkipped++;
         entriesToDisplay--;
         entry++;
      }
   }
   SetEntryList(entrylist);
}

//______________________________________________________________________________
Double_t TTreeTableInterface::GetValue(UInt_t row, UInt_t column)
{
   // Return the value of row,column. If the position does not exist
   // or does not contain a number, 0 is returned.

   static UInt_t prow = 0;

   if (row < fNRows) {
      Long64_t entry = 0;
      if (row == prow + 1) {
         entry = fEntries->Next();
      } else {
         entry = fEntries->GetEntry(row);
      }
      prow = row;
      fTree->LoadTree(entry);
   } else {
      Error("TTreeTableInterface", "Row requested does not exist");
      return 0;
   }
   if (column < fNColumns) {
      TTreeFormula *formula = (TTreeFormula *)fFormulas->At(column);
      if (!formula->IsString()) {
         return (Double_t)formula->EvalInstance();
      } else {
         Warning("TTreeTableInterface::GetValue", "Value requested is a "
                 "string, returning 0.");
         return 0;
      }
   } else {
      Error("TTreeTableInterface", "Column requested does not exist");
      return 0;
   }
}

//______________________________________________________________________________
const char *TTreeTableInterface::GetValueAsString(UInt_t row, UInt_t column)
{
   // Return the content of row,column as string to use in a
   // TGTableCell label.

   static UInt_t prow = 0;

   if (row < fNRows) {
      Long64_t entry = 0;
      if (row == prow + 1) {
         entry = fEntries->Next();
      } else {
         entry = fEntries->GetEntry(row);
      }
      prow = row;
      fTree->LoadTree(entry);
   } else {
      Error("TTreeTableInterface", "Row requested does not exist");
      return 0;
   }
   if (column < fNColumns) {
      TTreeFormula *formula = (TTreeFormula *)fFormulas->At(column);
      if(formula->IsString()) {
         return Form("%s", formula->EvalStringInstance());
      } else {
         return Form("%5.2f", (Double_t)formula->EvalInstance());
      }
   } else {
      Error("TTreeTableInterface", "Column requested does not exist");
      return 0;
   }
}

//______________________________________________________________________________
const char *TTreeTableInterface::GetRowHeader(UInt_t row)
{
   // Return a string to use as a label for rowheader at column.

   if (row < fNRows) {
      return Form("%lld", fEntries->GetEntry(row));
   } else {
      Error("TTreeTableInterface", "Row requested does not exist");
      return "";
   }
}

//______________________________________________________________________________
const char *TTreeTableInterface::GetColumnHeader(UInt_t column)
{
   // Return a string to use as a label for columnheader at column.

   TTreeFormula *formula = (TTreeFormula *)fFormulas->At(column);
   if (column < fNColumns) {
      return formula->GetTitle();
   } else {
      Error("TTreeTableInterface", "Column requested does not exist");
      return "";
   }
}

//______________________________________________________________________________
UInt_t TTreeTableInterface::GetNColumns()
{
   // Return the amount of column available.

   return fNColumns;
}

//______________________________________________________________________________
UInt_t TTreeTableInterface::GetNRows()
{
   // Return the amount of rows in the Tree.

   return fNRows;
}

//______________________________________________________________________________
void TTreeTableInterface::AddColumn(const char *expression, UInt_t position)
{
   // Add column according ot expression at position,
   // TGTable->Update() is needed afterwards to apply the change to
   // the TGTable.

   TString onerow = expression;

   if (onerow.Contains(':')) {
      Error("TTreeTableInterface::AddColumn", "Only 1 expression allowed.");
      return;
   }

   // Create the TreeFormula objects corresponding to the new expression
   TTreeFormula *formula = new TTreeFormula("Var1", expression, fTree);
   fFormulas->AddAt(formula, position);

   if (fManager) {
      fManager->Add(formula);
      fManager->Sync();
   }
   fNColumns++;
}

//______________________________________________________________________________
void TTreeTableInterface::AddColumn(TTreeFormula *formula, UInt_t position)
{
   // Add column with formula at position, TGTable->Update() is needed
   // afterwards to apply the change to the TGTable.

   if (position > fNColumns) {
      Error("TTreeTableInterface::AddColumn", "Please specify a "
            "valid position.");
      return;
   }
   fFormulas->AddAt(formula, position);
   if (fManager) {
      fManager->Add(formula);
      fManager->Sync();
   }
   fNColumns++;
}

//______________________________________________________________________________
void TTreeTableInterface::RemoveColumn(UInt_t position)
{
   // Remove column at position, TGTable->Update() is needed
   // afterwards to apply the change to the TGTable.

   if (position >= fNColumns) {
      Error("TTreeTableInterface::RemoveColumn", "Please specify a "
            "valid column.");
      return;
   } else if (fNColumns == 1) {
      Error("TTreeTableInterface::RemoveColumn", "Can't remove last column");
      return;
   }

   TTreeFormula *formula = (TTreeFormula *)fFormulas->RemoveAt(position);
   if (fManager) {
      fManager->Remove(formula);
      fManager->Sync();
   }

   if (formula) delete formula;
   fNColumns--;
}

//______________________________________________________________________________
void TTreeTableInterface::SetFormula(TTreeFormula *formula, UInt_t position)
{
   // Set the TTreeFormula of position to formula.

   if (position >= fNColumns) {
      Error("TTreeTableInterface::SetFormula", "Please specify a "
            "valid position.");
      return;
   }
   TTreeFormula *form = (TTreeFormula *)fFormulas->RemoveAt(position);
   if (fSelect) {
      fManager->Remove(form);
   }
   if (form) delete form;
   fFormulas->AddAt(formula, position);
   if (fManager) {
      fManager->Add(formula);
      fManager->Sync();
   }

}

//______________________________________________________________________________
void TTreeTableInterface::SetEntryList(TEntryList *entrylist)
{
   // Set the currently active entrylist.

   // Untested
   if (fEntries) delete fEntries;
   fEntries = entrylist;
   fNRows = fEntries->GetN();
   fTree->SetEntryList(entrylist);
}
