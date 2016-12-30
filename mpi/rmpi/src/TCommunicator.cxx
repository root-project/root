#include<Mpi/TCommunicator.h>

using namespace ROOT::Mpi;

//______________________________________________________________________________
TCommunicator::TCommunicator(const TCommunicator& comm):TObject(comm)
{
  fComm=comm.fComm;
}

//______________________________________________________________________________
TCommunicator::TCommunicator(const MPI::Comm& comm):fComm(comm){}

//______________________________________________________________________________
TCommunicator::~TCommunicator(){}

