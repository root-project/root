#include <TFile.h>
#include "set_problem.h"


void run() {
   // create a ROOT file with object SetProblem
   TFile * file = new TFile("set_problem.root", "RECREATE");

   SetProblem * pr = new SetProblem("test");

   pr->Write();

   file->Write();
   delete file;
   delete pr;


   // read the SetProblem object
   file = new TFile("set_problem.root", "READ");

   pr = (SetProblem*)file->Get("test");

   delete file;
   delete pr;
}

#ifndef __CINT__
main()
{
run();
}
#endif
