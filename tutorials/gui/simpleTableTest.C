/// \file
/// \ingroup tutorial_gui
/// This TableTest class is a simple example of how to use a TGSimpleTable that creates and owns it's own TGSimpleTableInterface.
/// TableTest inherits from TGMainFrame to create a top level frame to embed the TGTable in.
/// First the data needed is created. Then the TGSimpleTable is created using this data.
/// In the end, the table is added to the TGMainFrame that is the TableTest and the necessary calls to correctly draw the window are made.
/// For more information about the use of TGSimpleTable see it's documentation.
///
/// \macro_code
///
/// \author Roel Aaij 13/07/2007

#include <iostream>
#include <TApplication.h>
#include <TGClient.h>
#include <TGButton.h>
#include <TGFrame.h>
#include <TGLayout.h>
#include <TGWindow.h>
#include <TGLabel.h>
#include <TGNumberEntry.h>
#include <TString.h>
#include <TGButtonGroup.h>
#include <TGMenu.h>
#include <TGSimpleTable.h>

// A little class to automatically handle the generation of unique
// widget ids.
class IDList {
private:
   Int_t nID ;               // Generates unique widget IDs.
public:
   IDList() : nID(0) {}
   ~IDList() {}
   Int_t GetUnID(void) { return ++nID ; }
} ;

class TableTest : public TGMainFrame {

private:
   IDList         IDs ;      // Generator for unique widget IDs.
   Double_t     **fData;
   UInt_t         fNDataRows;
   UInt_t         fNDataColumns;
   UInt_t         fNTableRows;
   UInt_t         fNTableColumns;
   TGSimpleTable *fSimpleTable;

public:
   TableTest(const TGWindow *p, UInt_t ndrows, UInt_t ndcols,
             UInt_t ntrows, UInt_t ntcols, UInt_t w = 100, UInt_t h = 100) ;
   virtual ~TableTest() ;

   void DoExit() ;

   TGSimpleTable *GetTable() { return fSimpleTable; }

   ClassDef(TableTest, 0)
};

TableTest::TableTest(const TGWindow *p,  UInt_t ndrows, UInt_t ndcols,
                     UInt_t ntrows, UInt_t ntcols, UInt_t w, UInt_t h)
   : TGMainFrame(p, w, h), fData(0), fNDataRows(ndrows), fNDataColumns(ndcols),
     fNTableRows(ntrows), fNTableColumns(ntcols), fSimpleTable(0)
{
   SetCleanup(kDeepCleanup) ;
   Connect("CloseWindow()", "TableTest", this, "DoExit()") ;
   DontCallClose() ;

   // Create the needed data.
   Int_t i = 0, j = 0;
   fData = new Double_t*[fNDataRows];
   for (i = 0; i < (Int_t)fNDataRows; i++) {
      fData[i] = new Double_t[fNDataColumns];
      for (j = 0; j < (Int_t)fNDataColumns; j++) {
         fData[i][j] = 10 * i + j;
      }
   }

   // Create the table and add it to the TableTest that is a TGMainFrame.
   fSimpleTable = new TGSimpleTable(this, IDs.GetUnID(), fData, fNTableRows,
                                    fNTableColumns);
   AddFrame(fSimpleTable, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   // Calls to layout and draw the TableTest that is a TGMainFrame.
   SetWindowName("TGSimpleTable Test") ;
   MapSubwindows() ;
   Layout();
   Resize(GetDefaultWidth()+20, 600) ;
   MapWindow() ;

} ;

TableTest::~TableTest()
{
   // Destructor
   UInt_t i = 0;
   for (i = 0; i < fNDataRows; i++) {
      delete[] fData[i];
   }
   delete[] fData;
   Cleanup() ;
 }

 void TableTest::DoExit()
{
   // Exit this application via the Exit button or Window Manager.
   // Use one of the both lines according to your needs.
   // Please note to re-run this macro in the same ROOT session,
   // you have to compile it to get signals/slots 'on place'.

   DeleteWindow();            // to stay in the ROOT session
   //   gApplication->Terminate();   // to exit and close the ROOT session
}

TGSimpleTable *simpleTableTest(UInt_t ndrows = 500, UInt_t ndcols = 20,
                   UInt_t ntrows = 50, UInt_t ntcols = 10) {
   TableTest *test = new TableTest(0, ndrows, ndcols, ntrows, ntcols, 500, 200);
   return test->GetTable();
}
