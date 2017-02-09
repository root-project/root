#include<Mpi/TIntraCommunicator.h>
using namespace ROOT::Mpi;

//______________________________________________________________________________
TIntraCommunicator::TIntraCommunicator(const MPI_Comm &comm): TCommunicator(comm)
{
}

//______________________________________________________________________________
TIntraCommunicator::TIntraCommunicator(const TIntraCommunicator &comm): TCommunicator(comm.fComm)
{
}

//______________________________________________________________________________
TIntraCommunicator TIntraCommunicator::Dup() const
{
   MPI_Comm dupcomm;
   MPI_Comm_dup(fComm, &dupcomm);
   const TIntraCommunicator icomm(dupcomm);
   return  icomm;
}

//______________________________________________________________________________
TIntraCommunicator &TIntraCommunicator::Clone() const
{
   MPI_Comm dupcomm;
   MPI_Comm_dup(fComm, &dupcomm);
   TIntraCommunicator *icomm = new TIntraCommunicator(dupcomm);
   return  *icomm;
}

//______________________________________________________________________________
TIntraCommunicator TIntraCommunicator::Create(const TGroup &group) const
{
   MPI_Comm ncomm;
   ROOT_MPI_CHECK_GROUP(group);
   MPI_Comm_create(fComm, group, &ncomm);
   ROOT_MPI_CHECK_COMM(ncomm);
   return  ncomm;
}

//______________________________________________________________________________
TIntraCommunicator TIntraCommunicator::Split(int color, int key) const
{
   MPI_Comm ncomm;
   MPI_Comm_split(fComm, color, key, &ncomm);
   ROOT_MPI_CHECK_COMM(ncomm);
   const TIntraCommunicator icomm(ncomm);
   return icomm;
}

