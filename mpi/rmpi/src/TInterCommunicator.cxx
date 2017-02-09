
#include<Mpi/TInterCommunicator.h>
#include<Mpi/TIntraCommunicator.h>
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
   MPI_Comm_dup(fComm, &dupcomm);
   TInterCommunicator *icomm = new TInterCommunicator(dupcomm);
   return  *icomm;
}

//______________________________________________________________________________
TIntraCommunicator TInterCommunicator::Merge(bool high)
{
   MPI_Comm ncomm;
   MPI_Intercomm_merge(fComm, (Int_t)high, &ncomm);
   return ncomm;
}

//______________________________________________________________________________
Int_t TInterCommunicator::GetRemoteSize() const
{
   Int_t size;
   MPI_Comm_remote_size(fComm, &size);
   return size;
}

//______________________________________________________________________________
TGroup TInterCommunicator::GetRemoteGroup() const
{
   MPI_Group group;
   MPI_Comm_remote_group(fComm, &group);
   return group;
}

//______________________________________________________________________________
TInterCommunicator TInterCommunicator::Dup() const
{
   MPI_Comm dupcomm;
   MPI_Comm_dup(fComm, &dupcomm);
   ROOT_MPI_CHECK_COMM(dupcomm);
   return  dupcomm;
}

//______________________________________________________________________________
TInterCommunicator TInterCommunicator::Create(const TGroup &group) const
{
   MPI_Comm ncomm;
   ROOT_MPI_CHECK_GROUP(group);
   MPI_Comm_create(fComm, group, &ncomm);
   ROOT_MPI_CHECK_COMM(ncomm);
   return  ncomm;
}

//______________________________________________________________________________
TInterCommunicator TInterCommunicator::Split(int color, int key) const
{
   MPI_Comm ncomm;
   MPI_Comm_split(fComm, color, key, &ncomm);
   ROOT_MPI_CHECK_COMM(ncomm);
   return ncomm;
}
