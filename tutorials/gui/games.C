#ifndef __RUN_GAMES__

void games()
{
   Error("games", "Must be called from run_games.C...");
}

#else

class Hello;
class Aclock;
class Tetris;

void games()
{
   // This macro runs three "games" that each nicely illustrate the graphics capabilities of ROOT. 
   // Thanks to the clever usage of TTimer objects it looks like they are all
   // executing in parallel (emulation of multi-threading).
   // It uses the small classes generated in $ROOTSYS/test/Hello,
   // Aclock, Tetris
   //Author: Valeriy Onuchin

   // run the dancing Hello World
   Hello *hello = new Hello();

   // run the analog clock
   Aclock *clock = new Aclock();

   // run the Tetris game
   Tetris *tetris = new Tetris();
}

#endif
