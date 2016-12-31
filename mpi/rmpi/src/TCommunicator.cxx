#include<Mpi/TCommunicator.h>

using namespace ROOT::Mpi;

//______________________________________________________________________________
TCommunicator::TCommunicator(const TCommunicator& comm):TObject(comm)
{
  fComm = comm.fComm;
  fMainProcess = comm.fMainProcess;
}

//______________________________________________________________________________
TCommunicator::TCommunicator(const TComm &comm):fComm(comm),fMainProcess(0){}

//______________________________________________________________________________
TCommunicator::~TCommunicator(){}

