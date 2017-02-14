#include<Mpi/TTimer.h>
#include<iostream>
#include<Mpi/TIntraCommunicator.h>


//______________________________________________________________________________
void ROOT::Mpi::TTimer::Start()
{
   fStarTime = MPI_Wtime();
}

//______________________________________________________________________________
void ROOT::Mpi::TTimer::ReStart()
{
   Start();
}

//______________________________________________________________________________
Double_t ROOT::Mpi::TTimer::GetElapsed() const
{
   return MPI_Wtime() - fStarTime;
}

//______________________________________________________________________________
void ROOT::Mpi::TTimer::Sleep(Double_t msec)
{
   gSystem->Sleep(msec);
}

//______________________________________________________________________________
Bool_t ROOT::Mpi::TTimer::IsGlobal() const
{
   Int_t *global;
   Int_t flag = 0;

   MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_WTIME_IS_GLOBAL, &global, &flag);
   return flag == false ? kFALSE : *global != 0;
}

//______________________________________________________________________________
void ROOT::Mpi::TTimer::Print() const
{
}

