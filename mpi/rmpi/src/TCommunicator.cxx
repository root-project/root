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
   TRequest req = fComm.Isend(ibuffer, isize, MPI::CHAR, dest, tag);
   return req;
}

struct imsg_data {
   TMpiMessage *var;
   TComm *comm;
   Int_t source;
   Int_t tag;
};
int free_fn(void *extra_state);
Int_t query_fn(void *extra_state, MPI_Status *status) ;
int cancel_fn(void *extra_state, int complete);

//______________________________________________________________________________
template<> TRequest TCommunicator::IRecv<TMpiMessage>(TMpiMessage  &var, Int_t source, Int_t tag)
{

   imsg_data *imsg = new imsg_data;
   imsg->var = &var;
   imsg->comm = &fComm;
   imsg->source = source;
   imsg->tag = tag;

   MPI_Request greq;
   MPI_Grequest_start(query_fn, free_fn, cancel_fn, (void *)imsg, &greq);
//    MPI_Grequest_complete(greq);//I need to fix it here
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

Int_t query_fn(void *extra_state, MPI_Status *status)
{
   Int_t flag;
   MPI_Test_cancelled(status, &flag);
   if (flag) {
      std::cout << "in query_fn = CANCELLED\n";
      return MPI_ERR_IN_STATUS;
   }
//    MPI_Status_set_cancelled(status, flag);

   imsg_data *imsg = (imsg_data *)extra_state;

   Int_t isize = 0;
   MPI::Status s;
   imsg->comm->Probe(imsg->source, imsg->tag, s);
   isize = s.Get_elements(MPI::CHAR);
   std::cout << "in query_fn = source = " << imsg->source << " tag = " << imsg->tag << " size = " << isize << std::endl;

   Char_t *ibuffer = new Char_t[isize];
   TRequest req = imsg->comm->Irecv(ibuffer, isize, MPI::CHAR, imsg->source, imsg->tag);
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

   imsg->var->SetBuffer((void *)buffer, size, kFALSE);
   imsg->var->SetReadMode();
   imsg->var->Reset();
   return MPI_SUCCESS;
}

int free_fn(void *extra_state)
{
   imsg_data *obj = (imsg_data *)extra_state;
   if (obj) delete obj;
   return MPI_SUCCESS;
}


int cancel_fn(void *extra_state, int complete)
{
   if (!complete) {
      imsg_data *imsg = (imsg_data *)extra_state;
      Int_t isize = 0;
      MPI::Status s;
      imsg->comm->Probe(imsg->source, imsg->tag, s);
      isize = s.Get_elements(MPI::CHAR);
      Char_t *ibuffer = new Char_t[isize];
      TRequest req = imsg->comm->Irecv(ibuffer, isize, MPI::CHAR, imsg->source, imsg->tag);
      req.Cancel();
      delete ibuffer;
      std::cout << "incompleted!\n";
   }
   std::cout << "Cancelled!\n";
   return MPI_SUCCESS;
}
