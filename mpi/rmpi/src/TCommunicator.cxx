#include<Mpi/TCommunicator.h>
#include<iostream>
#include<TROOT.h>
using namespace ROOT::Mpi;

//______________________________________________________________________________
TCommunicator::TCommunicator(const TCommunicator &comm): TObject(comm)
{
   fComm = comm.fComm;
   fMainProcess = comm.fMainProcess;
}

//______________________________________________________________________________
TCommunicator::TCommunicator(const TComm &comm): fComm(comm), fMainProcess(0) {}

//______________________________________________________________________________
TCommunicator::~TCommunicator() {}

//______________________________________________________________________________
template<> void TCommunicator::Send<TMpiMessage>(const TMpiMessage &var, Int_t dest, Int_t tag) const
{
   auto buffer = var.Buffer();
   auto size   = var.BufferSize();
   TMpiMessageInfo msgi(buffer, size);
   msgi.SetSource(GetRank());
   msgi.SetDestination(dest);
   msgi.SetTag(tag);
   msgi.SetDataTypeName(var.GetDataTypeName());

   TMpiMessage msg;
   msg.WriteObject(msgi);
   auto ibuffer = msg.Buffer();
   auto isize = msg.BufferSize();
   fComm.Send(ibuffer, isize, MPI::CHAR, dest, tag);
}

//______________________________________________________________________________
template<> void TCommunicator::Recv<TMpiMessage>(TMpiMessage &var, Int_t source, Int_t tag) const
{
   UInt_t isize = 0;
   MPI::Status s;
   fComm.Probe(source, tag, s);
   isize = s.Get_elements(MPI::CHAR);

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
   auto size = msgi.GetBufferSize();
   Char_t *buffer = new Char_t[size];
   memcpy(buffer, msgi.GetBuffer(), size);

   var.SetBuffer((void *)buffer, size, kFALSE);
   var.SetReadMode();
   var.Reset();
}

//______________________________________________________________________________
template<> TRequest TCommunicator::ISend<TMpiMessage>(const TMpiMessage  &var, Int_t dest, Int_t tag)
{
   auto buffer = var.Buffer();
   auto size   = var.BufferSize();
   TMpiMessageInfo msgi(buffer, size);
   msgi.SetSource(GetRank());
   msgi.SetDestination(dest);
   msgi.SetTag(tag);
   msgi.SetDataTypeName(var.GetDataTypeName());

   TMpiMessage msg;
   msg.WriteObject(msgi);
   auto ibuffer = msg.Buffer();
   auto isize = msg.BufferSize();
   return  fComm.Isend(ibuffer, isize, MPI::CHAR, dest, tag);
}

//______________________________________________________________________________
template<> TRequest TCommunicator::ISsend<TMpiMessage>(const TMpiMessage  &var, Int_t dest, Int_t tag)
{
   auto buffer = var.Buffer();
   auto size   = var.BufferSize();
   TMpiMessageInfo msgi(buffer, size);
   msgi.SetSource(GetRank());
   msgi.SetDestination(dest);
   msgi.SetTag(tag);
   msgi.SetDataTypeName(var.GetDataTypeName());

   TMpiMessage msg;
   msg.WriteObject(msgi);
   auto ibuffer = msg.Buffer();
   auto isize = msg.BufferSize();
   return  fComm.Issend(ibuffer, isize, MPI::CHAR, dest, tag);
}

//______________________________________________________________________________
template<> TRequest TCommunicator::IRsend<TMpiMessage>(const TMpiMessage  &var, Int_t dest, Int_t tag)
{
   auto buffer = var.Buffer();
   auto size   = var.BufferSize();
   TMpiMessageInfo msgi(buffer, size);
   msgi.SetSource(GetRank());
   msgi.SetDestination(dest);
   msgi.SetTag(tag);
   msgi.SetDataTypeName(var.GetDataTypeName());

   TMpiMessage msg;
   msg.WriteObject(msgi);
   auto ibuffer = msg.Buffer();
   auto isize = msg.BufferSize();
   return  fComm.Irsend(ibuffer, isize, MPI::CHAR, dest, tag);
}


//______________________________________________________________________________
template<> TGrequest TCommunicator::IRecv<TMpiMessage>(TMpiMessage  &var, Int_t source, Int_t tag)
{

   IMsg *_imsg = new IMsg;
   _imsg->fMsg = &var;
   _imsg->fComm = &fComm;
   _imsg->fCommunicator = this;
   _imsg->fSource = source;
   _imsg->fTag = tag;

   //query lambda function
   auto query_fn = [ = ](void *extra_state, TStatus & status)->Int_t {
      if (status.IsCancelled())
      {
         return MPI_ERR_IN_STATUS;
      }
      IMsg  *imsg = (IMsg *)extra_state;

      Int_t isize = 0;
      TStatus s;
      imsg->fCommunicator->Probe(imsg->fSource, imsg->fTag, s);
      if (s.IsCancelled())
      {
         return MPI_ERR_IN_STATUS;
      }
      isize = s.fStatus.Get_elements(MPI::CHAR);
//       std::cout << "in query_fn = source = " << imsg->fSource << " tag = " << imsg->fTag << " size = " << isize << std::endl;

      Char_t *ibuffer = new Char_t[isize];
      TRequest req = imsg->fComm->Irecv(ibuffer, isize, MPI::CHAR, imsg->fSource, imsg->fTag);
      req.Wait();

      TMpiMessageInfo msgi;

      TMpiMessage msg(ibuffer, isize);
      auto cl = gROOT->GetClass(typeid(msgi));
      auto obj_tmp = (TMpiMessageInfo *)msg.ReadObjectAny(cl);
      memcpy((void *)&msgi, (void *)obj_tmp, sizeof(TMpiMessageInfo));

      //TODO: added error control here
      //check the tag, if destination is equal etc..

      //passing information from TMpiMessageInfo to TMpiMessage
      auto size = msgi.GetBufferSize();
      Char_t *buffer = new Char_t[size];
      memcpy(buffer, msgi.GetBuffer(), size);

      imsg->fMsg->SetBuffer((void *)buffer, size, kFALSE);
      imsg->fMsg->SetReadMode();
      imsg->fMsg->Reset();
      return MPI_SUCCESS;
   };

   //free function
   auto free_fn = [ = ](void *extra_state)->Int_t {
      IMsg *obj = (IMsg *)extra_state;
      if (obj) delete obj;
      return MPI_SUCCESS;
   };

   //cancel lambda function
   auto cancel_fn = [ = ](void *extra_state, Bool_t complete)->Int_t {
      if (!complete)
      {
         IMsg *imsg = (IMsg *)extra_state;
         Int_t isize = 0;
         MPI::Status s;
         imsg->fComm->Probe(imsg->fSource, imsg->fTag, s);
         isize = s.Get_elements(MPI::CHAR);
         Char_t *ibuffer = new Char_t[isize];
         TRequest req = imsg->fComm->Irecv(ibuffer, isize, MPI::CHAR, imsg->fSource, imsg->fTag);
         req.Cancel();
         delete ibuffer;
         std::cout << "incompleted!\n";
      }
      std::cout << "Cancelled!\n";
      return MPI_SUCCESS;
   };

   //creating General Request with lambda function
   auto greq = TGrequest::Start(query_fn, free_fn, cancel_fn, (void *)_imsg);
   return greq;
}


//______________________________________________________________________________
template<> void TCommunicator::Bcast<TMpiMessage>(TMpiMessage &var, Int_t root) const
{
   Char_t *ibuffer = nullptr ;
   UInt_t isize = 0;

   if (GetRank() == root) {
      auto buffer = var.Buffer();
      auto size   = var.BufferSize();
      TMpiMessageInfo msgi(buffer, size);
      msgi.SetSource(GetRank());
      msgi.SetDestination(-1);
      msgi.SetTag(root);
      msgi.SetDataTypeName(var.GetDataTypeName());

      TMpiMessage msg;
      msg.WriteObject(msgi);
      isize = msg.BufferSize();
      ibuffer = new Char_t[isize];
      memcpy(ibuffer, msg.Buffer(), isize);
   }

   fComm.Bcast(&isize, 1, MPI::INT, root);

   if (GetRank() != root) {
      ibuffer = new Char_t[isize];
   }
   fComm.Bcast(ibuffer, isize, MPI::CHAR, root);

   TMpiMessageInfo msgi;

   TMpiMessage msgr(ibuffer, isize);
   auto cl = gROOT->GetClass(typeid(msgi));
   auto obj_tmp = (TMpiMessageInfo *)msgr.ReadObjectAny(cl);
   memcpy((void *)&msgi, (void *)obj_tmp, sizeof(TMpiMessageInfo));

   //TODO: added error control here
   //check the tag, if destination is equal etc..

   //passing information from TMpiMessageInfo to TMpiMessage
   auto size = msgi.GetBufferSize();
   Char_t *buffer = new Char_t[size];
   memcpy(buffer, msgi.GetBuffer(), size);

   var.SetBuffer((void *)buffer, size, kFALSE);
   var.SetReadMode();
   var.Reset();

}

//______________________________________________________________________________
void  TCommunicator::Barrier() const
{
   fComm.Barrier();
}


//______________________________________________________________________________
Bool_t TCommunicator::Iprobe(Int_t source, Int_t tag, TStatus &status) const
{
   return fComm.Iprobe(source, tag, status.fStatus);
}

//______________________________________________________________________________
Bool_t TCommunicator::Iprobe(Int_t source, Int_t tag) const
{
   return fComm.Iprobe(source, tag);
}

//______________________________________________________________________________
void TCommunicator::Probe(Int_t source, Int_t tag, TStatus &status) const
{
   fComm.Probe(source, tag, status.fStatus);
}

//______________________________________________________________________________
void TCommunicator::Probe(Int_t source, Int_t tag) const
{
   fComm.Probe(source, tag);
}
