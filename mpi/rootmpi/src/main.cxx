#include <TRootMpi.h>
#include <iostream>

Int_t main(Int_t argc, Char_t *argv[])
{
   ROOT::Mpi::TRootMpi rootmpi(argc, argv);
   auto status = rootmpi.Launch();
   return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
