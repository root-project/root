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

//______________________________________________________________________________
template<> void TCommunicator::Send<TMpiMessage>(TMpiMessage &var, Int_t dest, Int_t tag) const
{
  auto buffer = var.Buffer();
  auto size   = var.BufferSize();
  TMpiMessageInfo msgi(buffer,size);
  msgi.SetSource(GetRank());
  msgi.SetDestination(dest);
  msgi.SetTag(tag);
  msgi.SetDataTypeName(var.GetDataTypeName());
  
  TMpiMessage msg;
  msg.WriteObject(msgi);
  auto ibuffer = msg.Buffer();
  auto isize = msg.BufferSize();
  fComm.Send(&isize, 1, MPI::INT, dest, tag);
  fComm.Send(ibuffer, isize, MPI::CHAR, dest, tag);
}

//______________________________________________________________________________
template<> void TCommunicator::Recv<TMpiMessage>(TMpiMessage &var, Int_t source, Int_t tag) const
{
  UInt_t isize = 0;
  fComm.Recv(&isize, 1, MPI::INT, source, tag);

  Char_t *ibuffer = new Char_t[isize];
  fComm.Recv(ibuffer, isize, MPI::CHAR, source, tag);

  TMpiMessageInfo msgi;
  
  TMpiMessage msg(ibuffer, isize);
  auto cl = gROOT->GetClass(typeid(msgi));
  auto obj_tmp = (TMpiMessageInfo *)msg.ReadObjectAny(cl);
  memcpy((void *)&msgi, (void *)obj_tmp, sizeof(TMpiMessageInfo));
  
  //TODO: added error control here
  //check the tag, if destination is equal etc..
  
  //passing information from TMpiMessageInfo to TMpiMessage
  auto size=msgi.GetBufferSize();
  Char_t *buffer=new Char_t[size];
  memcpy(buffer,msgi.GetBuffer(),size);
  
  var.SetBuffer((void*)buffer,size,kFALSE);
  var.SetReadMode();
  var.Reset();
  
}
