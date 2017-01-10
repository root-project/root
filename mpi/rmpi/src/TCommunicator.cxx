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
TCommunicator::TCommunicator(const MPI_Comm &comm): fComm(comm), fMainProcess(0) {}

//______________________________________________________________________________
TCommunicator::~TCommunicator()
{
}

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
   MPI_Send((void *)ibuffer, isize, MPI_CHAR, dest, tag, fComm);
}

//______________________________________________________________________________
template<> void TCommunicator::Recv<TMpiMessage>(TMpiMessage &var, Int_t source, Int_t tag) const
{
   Int_t isize = 0;
   MPI_Status s;
   MPI_Probe(source, tag, fComm, &s);
   MPI_Get_elements(&s, MPI_CHAR, &isize);

   Char_t *ibuffer = new Char_t[isize];
   MPI_Recv((void *)ibuffer, isize, MPI_CHAR, source, tag, fComm, &s);

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
   MPI_Request req;
   MPI_Isend((void *)ibuffer, isize, MPI_CHAR, dest, tag, fComm, &req);
   return req;
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
   MPI_Request req;
   MPI_Issend((void *)ibuffer, isize, MPI_CHAR, dest, tag, fComm, &req);
   return req;
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
   MPI_Request req;
   MPI_Irsend((void *)ibuffer, isize, MPI_CHAR, dest, tag, fComm, &req);
   return req;
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
   auto query_fn = [](void *extra_state, TStatus & status)->Int_t {
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
      MPI_Request _req;
      MPI_Irecv((void *)ibuffer, isize, MPI_CHAR, imsg->fSource, imsg->fTag, *imsg->fComm, &_req);
//       TRequest req = imsg->fComm->Irecv(ibuffer, isize, MPI::CHAR, imsg->fSource, imsg->fTag);
      TRequest req = _req;
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
   auto free_fn = [](void *extra_state)->Int_t {
      IMsg *obj = (IMsg *)extra_state;
      if (obj) delete obj;
      return MPI_SUCCESS;
   };

   //cancel lambda function
   auto cancel_fn = [](void *extra_state, Bool_t complete)->Int_t {
      if (!complete)
      {
         IMsg *imsg = (IMsg *)extra_state;
         Int_t isize = 0;
         MPI_Status s;
         MPI_Probe(imsg->fSource, imsg->fTag, *imsg->fComm, &s);
         MPI_Get_elements(&s, MPI_CHAR, &isize);

         Char_t *ibuffer = new Char_t[isize];
         MPI_Request req;
         MPI_Irecv((void *)ibuffer, isize, MPI_CHAR, imsg->fSource, imsg->fTag, *imsg->fComm, &req);
         MPI_Cancel(&req);
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

   MPI_Bcast((void *)&isize, 1, MPI_INT, root, fComm);

   if (GetRank() != root) {
      ibuffer = new Char_t[isize];
   }
   MPI_Bcast((void *)ibuffer, isize, MPI_CHAR, root, fComm);

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
   MPI_Barrier(fComm);
}


//______________________________________________________________________________
Bool_t TCommunicator::Iprobe(Int_t source, Int_t tag, TStatus &status) const
{
   Int_t flag;
   MPI_Status stat;
   MPI_Iprobe(source, tag, fComm, &flag, &stat);
   status = stat;
   return (Bool_t)flag;
}

//______________________________________________________________________________
Bool_t TCommunicator::Iprobe(Int_t source, Int_t tag) const
{
   Int_t flag;
   MPI_Status status;
   MPI_Iprobe(source, tag, fComm, &flag, &status);
   return (Bool_t)flag;
}

//______________________________________________________________________________
void TCommunicator::Probe(Int_t source, Int_t tag, TStatus &status) const
{
   MPI_Status stat;
   MPI_Probe(source, tag, fComm, &stat);
   status = stat;
}

//______________________________________________________________________________
void TCommunicator::Probe(Int_t source, Int_t tag) const
{
   MPI_Status stat;
   MPI_Probe(source, tag, fComm, &stat);
}
