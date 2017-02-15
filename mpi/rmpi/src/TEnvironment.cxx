#include<Mpi/TEnvironment.h>
#include<Mpi/TIntraCommunicator.h>
using namespace ROOT::Mpi;
//TODO: enable thread level and thread-safe for ROOT

//______________________________________________________________________________
TEnvironment::TEnvironment()
{
   MPI_Init(NULL, NULL);

   if (IsInitialized()) {
      Int_t result;
      MPI_Comm_compare((MPI_Comm)COMM_WORLD, MPI_COMM_WORLD, &result);
      if (result == IDENT) COMM_WORLD.SetCommName("ROOT::Mpi::COMM_WORLD");
   } else {
      //TODO: added error handling here
   }
}

//______________________________________________________________________________
TEnvironment::TEnvironment(Int_t &argc, Char_t ** &argv)
{
   MPI_Init(&argc, &argv);
   if (IsInitialized()) {
      Int_t result;
      MPI_Comm_compare((MPI_Comm)COMM_WORLD, MPI_COMM_WORLD, &result);
      if (result == IDENT) COMM_WORLD.SetCommName("ROOT::Mpi::COMM_WORLD");
   } else {
      //TODO: added error handling here
   }
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

//______________________________________________________________________________
TString TEnvironment::GetProcessorName()
{
   Char_t name[MAX_PROCESSOR_NAME];
   Int_t size;
   MPI_Get_processor_name(name, &size);
   return TString(name, size);
}
