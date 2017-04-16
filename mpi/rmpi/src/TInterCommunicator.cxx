#include<Mpi/TIntraCommunicator.h>
#include<Mpi/TInterCommunicator.h>
using namespace ROOT::Mpi;

//______________________________________________________________________________
TInterCommunicator::TInterCommunicator(): TCommunicator() {}

//______________________________________________________________________________
TInterCommunicator::TInterCommunicator(const TInterCommunicator &comm): TCommunicator(comm) {}

//______________________________________________________________________________
TInterCommunicator::TInterCommunicator(const MPI_Comm &comm): TCommunicator(comm) {}

//______________________________________________________________________________
TInterCommunicator &TInterCommunicator::Clone() const
{
   MPI_Comm dupcomm;
   ROOT_MPI_CHECK_CALL(MPI_Comm_dup, (fComm, &dupcomm), this);
   TInterCommunicator *icomm = new TInterCommunicator(dupcomm);
   return  *icomm;
}

//______________________________________________________________________________
TIntraCommunicator TInterCommunicator::Merge(Int_t high)
{
   MPI_Comm ncomm;
   ROOT_MPI_CHECK_CALL(MPI_Intercomm_merge, (fComm, high, &ncomm), this);
   return ncomm;
}

//______________________________________________________________________________
Int_t TInterCommunicator::GetRemoteSize() const
{
   Int_t size;
   ROOT_MPI_CHECK_CALL(MPI_Comm_remote_size, (fComm, &size), this);
   return size;
}

//______________________________________________________________________________
TGroup TInterCommunicator::GetRemoteGroup() const
{
   MPI_Group group;
   ROOT_MPI_CHECK_CALL(MPI_Comm_remote_group, (fComm, &group), this);
   return group;
}

//______________________________________________________________________________
TInterCommunicator TInterCommunicator::Dup() const
{
   MPI_Comm dupcomm;
   ROOT_MPI_CHECK_CALL(MPI_Comm_dup, (fComm, &dupcomm), this);
   ROOT_MPI_CHECK_COMM(dupcomm, this);
   return  dupcomm;
}

//______________________________________________________________________________
TInterCommunicator TInterCommunicator::Create(const TGroup &group) const
{
   MPI_Comm ncomm;
   ROOT_MPI_CHECK_GROUP(group, this);
   ROOT_MPI_CHECK_CALL(MPI_Comm_create, (fComm, group, &ncomm), this);
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return  ncomm;
}

//______________________________________________________________________________
TInterCommunicator TInterCommunicator::Split(Int_t color, Int_t key) const
{
   MPI_Comm ncomm;
   ROOT_MPI_CHECK_CALL(MPI_Comm_split, (fComm, color, key, &ncomm), this);
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}
