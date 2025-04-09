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

   TFormula * form = new TFormula("for","left() || right()");
   form->Print();
   TFormula * fand = new TFormula("fand","!left() && !right()");
   fand->Print();

   fprintf(stderr,"regular C++: left() || right()\n");
   if ( left() || right() ) {}

   fprintf(stderr,"TFormula: left() || right()\n");
   form->Eval(0);

   fprintf(stderr,"regular C++: !left() && !right()\n");
   if ( !left() && !right() ) {}

   fprintf(stderr,"TFormula: !left() && !right()\n");
   fand->Eval(0);

   TFormula * for2 = new TFormula("for2","!left() || right()");

   fprintf(stderr,"regular C++: !left() || right()\n");
   if ( !left() || right() ) {}

   fprintf(stderr,"TFormula: !left() || right()\n");
   for2->Eval(0);
   

   TFormula * fand2 = new TFormula("fand2","left() && !right()");

   fprintf(stderr,"regular C++: left() && !right()\n");
   if ( left() && !right() ) {}

   fprintf(stderr,"TFormula: left() && !right()\n");
   fand2->Eval(0);

   TFormula * fcomp = new TFormula("fcomp","left()+ (!left() && !right()) + right()");

   fprintf(stderr,"regular C++: left()+ (!left() && !right()) + right()\n");
   bool h = left() + ( !left() && !right() ) + right();

   fprintf(stderr,"TFormula: left()+ (!left() && !right()) + right()\n");
   fcomp->Eval(0);

}
