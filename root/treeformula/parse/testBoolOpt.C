bool right() {
   fprintf(stderr,"running right side\n");
   return true;
}

bool left() {
   fprintf(stderr,"running left side\n");
   return true;
}

void testBoolOpt() {
   // .L testBoolOpt.C

   fprintf(stderr,"regular C++: left() || right()\n");
   if ( left() || right() ) {}

   TFormula * for = new TFormula("for","left() || right()");
   for->Print();
   fprintf(stderr,"TFormula: left() || right()\n");
   for->Eval(0);

   fprintf(stderr,"regular C++: !left() && !right()\n");
   if ( !left() && !right() ) {}

   TFormula * fand = new TFormula("fand","!left() && !right()");
   fand->Print();
   fprintf(stderr,"TFormula: !left() && !right()\n");
   fand->Eval(0);


   fprintf(stderr,"regular C++: left()+ (!left() && !right()) + right()\n");
   bool h = left() + ( !left() && !right() ) + right();

   TFormula * fcomp = new TFormula("fcomp","left()+ (!left() && !right()) + right()");
   fcomp->Print();
   fprintf(stderr,"TFormula: left()+ (!left() && !right()) + right()\n");
   fcomp->Eval(0);

}
