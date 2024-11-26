/// \file
/// \ingroup tutorial_gui
/// Example showing how to customize a context menu for a class
///
/// \macro_code
///
/// \author Ilka antcheva

{
   cl = gROOT->GetClass("TH1F");

   cl->MakeCustomMenuList();
   ml = cl->GetMenuList();

   ((TClassMenuItem*)ml->At(1))->SetTitle("Add histos...");
   ((TClassMenuItem*)ml->At(2))->SetTitle("Divide histos...");
   ((TClassMenuItem*)ml->At(3))->SetTitle("Draw panel...");
   ((TClassMenuItem*)ml->At(4))->SetTitle("Fit one function...");
   ((TClassMenuItem*)ml->At(5))->SetTitle("Fit panel...");
   ((TClassMenuItem*)ml->At(6))->SetTitle("Multiply histos...");
   ((TClassMenuItem*)ml->At(7))->SetTitle("Rebin...");
   ((TClassMenuItem*)ml->At(8))->SetTitle("Set maximum scale...");
   ((TClassMenuItem*)ml->At(9))->SetTitle("Set minimum scale...");
   ((TClassMenuItem*)ml->At(10))->SetTitle("Smooth histogram");
   ((TClassMenuItem*)ml->At(12))->SetTitle("Set name...");
   ((TClassMenuItem*)ml->At(13))->SetTitle("Set title...");
   ((TClassMenuItem*)ml->At(15))->SetTitle("Delete histogram");
   ((TClassMenuItem*)ml->At(16))->SetTitle("Draw class info");
   ((TClassMenuItem*)ml->At(17))->SetTitle("Draw clone");
   ((TClassMenuItem*)ml->At(18))->SetTitle("Dump information");
   ((TClassMenuItem*)ml->At(19))->SetTitle("Inspect");
   ((TClassMenuItem*)ml->At(20))->SetTitle("Set drawing option...");
   ((TClassMenuItem*)ml->At(22))->SetTitle("Set line attributes...");
   ((TClassMenuItem*)ml->At(24))->SetTitle("Set fill attributes...");
   ((TClassMenuItem*)ml->At(26))->SetTitle("Set marker attributes...");

// Remove separators at the end, between attributes
   mi = (TClassMenuItem*)ml->At(23);
   delete mi;
   mi = (TClassMenuItem*)ml->At(24);
   delete mi;
}
