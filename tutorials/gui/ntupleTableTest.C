// This TableTest class is a simple example of how to use a TGTable
// with a TTreeTableInterface. TableTest inherits from TGMainFrame to
// create a top level frame to embed the TGTable in. First, the
// hsimple.root file is opened to obtain an ntuple. Then a
// TTreeTableInterface is created using this ntuple. A table is then
// created using the interface. In the end, the table is added to the
// TGMainFrame that is the TableTest and the necessary calls to
// correctly draw the window are made. For more information about the
// use of TTreeTableInterface and TGTable, see their documentation.

// author: Roel Aaij 13/07/2007

#include <iostream>
#include <TApplication.h>
#include <TGClient.h>
#include <TGButton.h>
#include <TGFrame.h>
#include <TGWindow.h>
#include <TString.h>
#include <TGTable.h>
#include <TTreeTableInterface.h>
#include <TFile.h>
#include <TNtuple.h>
#include <TSelectorDraw.h>


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
   IDList   fIDs ;      // Generator for unique widget IDs.
   UInt_t   fNTableRows;
   UInt_t   fNTableColumns;
   TGTable *fTable;
   TFile   *fFile;

   TTreeTableInterface *fInterface;

public:
   TableTest(const TGWindow *p, UInt_t ntrows, UInt_t ntcols,
             UInt_t w = 100, UInt_t h = 100) ;
   virtual ~TableTest() ;

   void DoExit() ;

   TGTable *GetTable() { return fTable; }
   TTreeTableInterface *GetInterface() { return fInterface; }

   ClassDef(TableTest, 0)
};

TableTest::TableTest(const TGWindow *p, UInt_t ntrows, UInt_t ntcols,
                     UInt_t w, UInt_t h)
   : TGMainFrame(p, w, h),  fNTableRows(ntrows), fNTableColumns(ntcols),
     fTable(0)
{
   SetCleanup(kDeepCleanup) ;
   Connect("CloseWindow()", "TableTest", this, "DoExit()") ;
   DontCallClose() ;

   // Open root file for the tree
   fFile = new TFile("$ROOTSYS/tutorials/hsimple.root");

   if (!fFile || fFile->IsZombie()) {
      printf("Please run <ROOT location>/tutorials/hsimple.C first.");
      return;
   }

   // Get the ntuple from the file.
   TNtuple *ntuple = (TNtuple *)fFile->Get("ntuple");

   // Setup the expressions for the columns and the selection of the
   // interface.
   TString varexp = "px:py:pz:random:sin(px):log(px/py):log(pz)";
   TString select = "";
   TString options = "";

   // Create the interface.
   fInterface = new TTreeTableInterface(ntuple, varexp.Data(), select.Data(),
                                        options.Data());

   // Create the table and add it to the TableTest that is a TGMainFrame.
   fTable = new TGTable(this, fIDs.GetUnID(), fInterface, fNTableRows,
                                fNTableColumns);
   AddFrame(fTable, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   // Calls to layout and draw the TableTest that is a TGMainFrame.
   SetWindowName("Tree Table Test") ;
   MapSubwindows() ;
   Layout();
   Resize(GetDefaultWidth()+20, 600) ;
   MapWindow() ;

} ;

TableTest::~TableTest()
{
   // Destructor
   delete fInterface;
   fFile->Close();
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

TGTable *ntupleTableTest(UInt_t ntrows = 50, UInt_t ntcols = 10) {
   TableTest *test = new TableTest(0, ntrows, ntcols, 500, 200);
   return test->GetTable();
}
