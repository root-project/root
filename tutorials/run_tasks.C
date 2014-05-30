// Macro used to prepare the environment before running the tasks.C macro

void run_tasks()
{
   TString dir = gSystem->UnixPathName(gInterpreter->GetCurrentMacroName());
   dir.ReplaceAll("run_tasks.C","");
   dir.ReplaceAll("/./","/");
   gROOT->LoadMacro(dir +"MyTasks.cxx+");
   
   gROOT->ProcessLine("#define __RUN_TASKS__ 1");
   gInterpreter->ExecuteMacro("tasks.C");
   gROOT->ProcessLine("#undef __RUN_TASKS__");
}
