#include<Mpi/TEnvironment.h>

using namespace ROOT::Mpi;
//TODO: enable thread level and thread-safe for ROOT

//______________________________________________________________________________
TEnvironment::TEnvironment()
{
   MPI::Init();
}

//______________________________________________________________________________
TEnvironment::TEnvironment(Int_t &argc, Char_t ** &argv)
{
   MPI::Init(argc, argv);
}

//______________________________________________________________________________
TEnvironment::~TEnvironment()
{
   //if mpi's environment is initialized then finalize it
   if (!IsFinalized()) {
      Finalize();
   }
}

//______________________________________________________________________________
Bool_t TEnvironment::IsFinalized()
{
   return (Bool_t)MPI::Is_finalized();
}

//______________________________________________________________________________
Bool_t TEnvironment::IsInitialized()
{
   return (Bool_t)MPI::Is_initialized();
}


//______________________________________________________________________________
void TEnvironment::Finalize()
{
   //Finalize the mpi's environment
   MPI::Finalize();
}

