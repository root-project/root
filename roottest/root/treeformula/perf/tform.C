#include "TFormula.h"
#include "stdlib.h"

void tform(int formula, int iter) {

   TFormula *form;
   switch(formula) {
      case 0: form = new TFormula("form","[0]"); break;
      case 1: form = new TFormula("form","xylandau"); break;
      default: return;
   }
   form->Print();

   Double_t x[2];
   Double_t param[10];
   x[0] = 1.0;
   x[1] = 2.0;
   for(int i=0;i<10;i++) param[i] = (i+2.5)*3;

   for(int n=0; n<iter; ++n) {
      form->EvalPar(x,param);
   }
      
}

#ifndef __CINT__
int main(int argc,char**argv) {

   if (argc!=3) {
      fprintf(stderr,"tform requires 1 argument:\n");
      fprintf(stderr,"tform <formula#> <samplesize>\n");
      return 1;
   }
   
   int formula = atoi(argv[1]);
   int size = atoi(argv[2]);
   tform(formula,size);
}
#endif


