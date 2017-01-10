#include<Mpi/TEnvironment.h>

using namespace ROOT::Mpi;
//TODO: enable thread level and thread-safe for ROOT

//______________________________________________________________________________
TEnvironment::TEnvironment()
{
   MPI_Init(NULL, NULL);
}

//______________________________________________________________________________
TEnvironment::TEnvironment(Int_t &argc, Char_t ** &argv)
{
   MPI_Init(&argc, &argv);
}

//______________________________________________________________________________
TEnvironment::~TEnvironment()
{
   //if mpi's environment is initialized then finalize it
   if (!IsFinalized()) {
      Finalize();
   }
}

void TEnvironment::Init()
{
   MPI_Init(NULL, NULL);
}

//______________________________________________________________________________
Bool_t TEnvironment::IsFinalized()
{
   Int_t t;
   MPI_Finalized(&t);
   return Bool_t(t);
}

//______________________________________________________________________________
Bool_t TEnvironment::IsInitialized()
{
   Int_t t;
   MPI_Initialized(&t);
   return (Bool_t)(t);
}


//______________________________________________________________________________
void TEnvironment::Finalize()
{
   //Finalize the mpi's environment
   MPI_Finalize();
}
