 {
   TControlBar *ed = new TControlBar("vertical");
   ed->AddButton("Arc",      "gROOT->SetEditorMode(\"Arc\")",       "Create an arc of circle");
   ed->AddButton("Arrow",    "gROOT->SetEditorMode(\"Arrow\")",     "Create an Arrow");
   ed->AddButton("Diamond",  "gROOT->SetEditorMode(\"Diamond\")",   "Create a diamond");
   ed->AddButton("Ellipse",  "gROOT->SetEditorMode(\"Ellipse\")",   "Create an Ellipse");
   ed->AddButton("Pad",      "gROOT->SetEditorMode(\"Pad\")",       "Create a pad");
   ed->AddButton("Pave",     "gROOT->SetEditorMode(\"Pave\")",      "Create a Pave");
   ed->AddButton("PaveLabel","gROOT->SetEditorMode(\"PaveLabel\")", "Create a PaveLabel (prompt for label)");
   ed->AddButton("PaveText", "gROOT->SetEditorMode(\"PaveText\")",  "Create a PaveText");
   ed->AddButton("PavesText","gROOT->SetEditorMode(\"PavesText\")", "Create a PavesText");
   ed->AddButton("PolyLine", "gROOT->SetEditorMode(\"PolyLine\")",  "Create a PolyLine (TGraph)");
   ed->AddButton("Text",     "gROOT->SetEditorMode(\"Text\")",      "Create a Text string");
   ed->Show();
}
