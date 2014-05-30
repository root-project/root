// Macro used to prepare the environment before running the games.C macro

void run_games()
{
   gSystem->Load("libGui");
   Bool_t UNIX = strcmp(gSystem->GetName(), "Unix") == 0;
   Int_t st1 = gSystem->Load("$(ROOTSYS)/test/Aclock");
   if (st1 == -1) {
      printf("===>The macro games will try to build the Aclock library\n");
      if (UNIX)
         gSystem->Exec("(cd $ROOTSYS/test; make Aclock)");
      else
         gSystem->Exec("(cd %ROOTSYS%\\test && nmake -f Makefile.win32 Aclock.dll)");

      st1 = gSystem->Load("$(ROOTSYS)/test/Aclock");
   }
   Int_t st2 = gSystem->Load("$(ROOTSYS)/test/Hello");
   if (st2 == -1) {
      printf("===>The macro games will try to build the Hello library\n");
      if (UNIX)
         gSystem->Exec("(cd $ROOTSYS/test; make Hello)");
      else
         gSystem->Exec("(cd %ROOTSYS%\\test && nmake -f Makefile.win32 Hello.dll)");

      st2 = gSystem->Load("$(ROOTSYS)/test/Hello");
   }
   Int_t st3 = gSystem->Load("$(ROOTSYS)/test/Tetris");
   if (st3 == -1) {
      if (UNIX) {
         printf("===>The macro games will try to build the Tetris library\n");
         gSystem->Exec("(cd $ROOTSYS/test; make Tetris)");
      } else {
         gSystem->Exec("(cd %ROOTSYS%\\test && nmake -f Makefile.win32 Tetris.dll)");
      }
      st3 = gSystem->Load("$(ROOTSYS)/test/Tetris");
   }
   if (st1 || st2 || st3) {
      printf("ERROR: one of the shared libs in $ROOTSYS/test didn't load properly\n");
      return;
   }
   gROOT->ProcessLine("#define __RUN_GAMES__ 1");
   gInterpreter->ExecuteMacro("games.C");
   gROOT->ProcessLine("#undef __RUN_GAMES__");
}
