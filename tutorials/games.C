void games()
{
   // This macro runs three "games" that each nicely illustrate
   // some of the graphics capabilities of ROOT. Thanks to the
   // clever usage of TTimer objects it looks like they are all
   // executing in parallel (emulation of multi-threading).
   // It uses the small classes generated in $ROOTSYS/test/Hello,
   // Aclock, Tetris

   Bool_t UNIX = strcmp(gSystem->GetName(), "Unix") == 0;
   Int_t st1 = gSystem->Load("$(ROOTSYS)/test/Aclock");
   if (st1 == -1) {
      printf("===>The macro games will try to build the Aclock library\n");
      if (UNIX)
         gSystem->Exec("(cd $ROOTSYS/test; make Aclock.so)");
      else
         gSystem->Exec("(cd %ROOTSYS%\\test && nmake Aclock.dll)");

      st1 = gSystem->Load("$(ROOTSYS)/test/Aclock");
   }
   Int_t st2 = gSystem->Load("$(ROOTSYS)/test/Hello");
   if (st2 == -1) {
      printf("===>The macro games will try to build the Hello library\n");
      if (UNIX)
         gSystem->Exec("(cd $ROOTSYS/test; make Hello.so)");
      else
         gSystem->Exec("(cd %ROOTSYS%\\test && nmake Hello.dll)");

      st2 = gSystem->Load("$(ROOTSYS)/test/Hello");
   }
   Int_t st3 = gSystem->Load("$(ROOTSYS)/test/Tetris");
   if (st3 == -1) {
      if (UNIX) {
         printf("===>The macro games will try to build the Tetris library\n");
         gSystem->Exec("(cd $ROOTSYS/test; make Tetris.so)");
      } else {
         gSystem->Exec("(cd $ROOTSYS/test; nmake Tetris.dll)");
      }
      st3 = gSystem->Load("$(ROOTSYS)/test/Tetris");
   }

   if (st1 || st2 || st3) {
      printf("ERROR: one of the shared libs in $ROOTSYS/test didn't load properly\n");
      return;
   }

   // run the dancing Hello World
   Hello *hello = new Hello();

   // run the analog clock
   Aclock *clock = new Aclock();

   // run the Tetris game
   if (UNIX) Tetris *tetris = new Tetris();
}
