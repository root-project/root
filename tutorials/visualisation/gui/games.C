/// \file
/// \ingroup tutorial_gui
/// This macro runs three "games" that each nicely illustrate the graphics capabilities of ROOT.
/// Thanks to the clever usage of TTimer objects it looks like they are all executing in parallel (emulation of
/// multi-threading). It uses the small classes generated in $ROOTSYS/test/Hello, Aclock, Tetris
///
/// \macro_code
///
/// \author Valeriy Onuchin

#ifndef __RUN_GAMES__

void games()
{
   gSystem->Load("libGui");
   Bool_t UNIX = strcmp(gSystem->GetName(), "Unix") == 0;
   Int_t st1 = gSystem->Load("$(ROOTSYS)/test/Aclock");
   if (st1 == -1) {
      printf("===>The $(ROOTSYS)/test/Aclock library failed to load\n");
   }
   Int_t st2 = gSystem->Load("$(ROOTSYS)/test/Hello");
   if (st2 == -1) {
      printf("===>The $(ROOTSYS)/test/Hello library failed to load\n");
   }
   Int_t st3 = gSystem->Load("$(ROOTSYS)/test/Tetris");
   if (st3 == -1) {
      printf("===>The $(ROOTSYS)/test/Tetris library failed to load\n");
   }
   if (st1 || st2 || st3) {
      printf("ERROR: at least one of the shared libs in $ROOTSYS/test didn't load properly\n");
      return;
   }
   gROOT->ProcessLine("#define __RUN_GAMES__ 1");

   gInterpreter->AddIncludePath("${ROOTSYS}/test");

   // Add this for CLING not to complain
   gROOT->ProcessLine("#include \"Hello.h\"");
   gROOT->ProcessLine("#include \"Aclock.h\"");
   gROOT->ProcessLine("#include \"Tetris.h\"");

   gROOT->ProcessLine("#include \"games.C\"");
   gROOT->ProcessLine("rungames()");
   gROOT->ProcessLine("#undef __RUN_GAMES__");
}

#else

class Hello;
class Aclock;
class Tetris;

void rungames()
{
   // run the dancing Hello World
   Hello *hello = new Hello();

   // run the analog clock
   Aclock *clock = new Aclock();

   // run the Tetris game
   Tetris *tetris = new Tetris();
}

#endif
