// Macro used to prepare the environment before running the pythia_display.C macro

void run_pythia_display()
{
   gROOT->ProcessLine("#define __RUN_PYTHIA_DISPLAY__ 1");
   gROOT->LoadMacro("MultiView.C+");
   gInterpreter->ExecuteMacro("pythia_display.C");
   gROOT->ProcessLine("#undef __RUN_PYTHIA_DISPLAY__");
}

