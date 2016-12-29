#include<TRun.h>

int main(int argc, char *argv[])
{
   ROOT::Mpi::TRun rootmpi(argc, argv);
   return rootmpi.Launch();
}


