#include<TRootMpi.h>
#include<iostream>

Int_t main(Int_t argc, Char_t *argv[])
{
   ROOT::Mpi::TRootMpi rootmpi(argc, argv);
   return rootmpi.Launch();
}


