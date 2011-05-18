// @(#)macros:$Id$
// Author: Axel Naumann, 2008-05-22
//
// This script gets executed when double-clicking a ROOT file (currently only on Windows).
// The file that got double clicked and opened is accessible as _file0.

void onBrowserClose() {
   gApplication->Terminate(0);
}

void fileopen() 
{
   TBrowser *b = new TBrowser;
   // or, to only browse the file:
   // new TBrowser(_file0);

   // Quit ROOT when the browser gets closed:
   b->GetBrowserImp()->GetMainFrame()->Connect("CloseWindow()", 0, 0, "onBrowserClose()");
} 
