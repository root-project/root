#include<TROOTMpi.h>

int main(int argc, char *argv[])
{
   ROOT::Mpi::TROOTMpi rootmpi(argc, argv);
   return rootmpi.Launch();
}


