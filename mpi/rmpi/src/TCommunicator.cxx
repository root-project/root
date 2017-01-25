#include<Mpi/TCommunicator.h>
#include <Mpi/TMpiMessage.h>
#include<iostream>
#include<TSystem.h>
#include<TROOT.h>
using namespace ROOT::Mpi;

ROOT::Mpi::TCommunicator *gComm = new ROOT::Mpi::TCommunicator(MPI_COMM_WORLD);
ROOT::Mpi::TCommunicator COMM_WORLD(MPI_COMM_WORLD);

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
void  TCommunicator::Barrier() const
{
   MPI_Barrier(fComm);
}

//______________________________________________________________________________
void  TCommunicator::IBarrier(TRequest &req) const
{
   MPI_Ibarrier(fComm, &req.fRequest);
}

//______________________________________________________________________________
Bool_t TCommunicator::IProbe(Int_t source, Int_t tag, TStatus &status) const
{
   Int_t flag;
   MPI_Status stat;
   MPI_Iprobe(source, tag, fComm, &flag, &stat);
   status = stat;
   return (Bool_t)flag;
}

//______________________________________________________________________________
Bool_t TCommunicator::IProbe(Int_t source, Int_t tag) const
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
template<> void TCommunicator::Send<TMpiMessage>(const TMpiMessage *vars, Int_t count, Int_t dest, Int_t tag) const
{
   std::vector<TMpiMessageInfo>  msgis(count);
   for (auto i = 0; i < count; i++) {
      auto buffer = vars[i].Buffer();
      auto size   = vars[i].BufferSize();
      TMpiMessageInfo msgi(buffer, size);
      msgi.SetSource(GetRank());
      msgi.SetDestination(dest);
      msgi.SetTag(tag);
      msgi.SetDataTypeName(vars[i].GetDataTypeName());
      msgis[i] = msgi;
   }
   Send(msgis, dest, tag);
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
template<> void TCommunicator::Recv<TMpiMessage>(TMpiMessage *vars, Int_t count, Int_t source, Int_t tag) const
{
   std::vector<TMpiMessageInfo>  msgis(count);
   Recv(msgis, source, tag);
   //TODO: added error control here
   //check the tag, if destination is equal etc..

   for (auto i = 0; i < count; i++) {
      //passing information from TMpiMessageInfo to TMpiMessage
      auto size = msgis[i].GetBufferSize();
      Char_t *buffer = new Char_t[size];
      memcpy(buffer, msgis[i].GetBuffer(), size);

      vars[i].SetBuffer((void *)buffer, size, kFALSE);
      vars[i].SetReadMode();
      vars[i].Reset();
   }
}

//______________________________________________________________________________
template<> TRequest TCommunicator::ISend<TMpiMessage>(const TMpiMessage  &var, Int_t dest, Int_t tag)
{
   return ISend(&var, 1, dest, tag);
}

//______________________________________________________________________________
template<> TRequest TCommunicator::ISend<TMpiMessage>(const TMpiMessage *vars, Int_t count, Int_t dest, Int_t tag)
{
   std::vector<TMpiMessageInfo>  msgis(count);
   for (auto i = 0; i < count; i++) {
      auto buffer = vars[i].Buffer();
      auto size   = vars[i].BufferSize();
      TMpiMessageInfo msgi(buffer, size);
      msgi.SetSource(GetRank());
      msgi.SetDestination(dest);
      msgi.SetTag(tag);
      msgi.SetDataTypeName(vars[i].GetDataTypeName());
      msgis[i] = msgi;
   }
   TMpiMessage msg;
   msg.WriteObject(msgis);
   auto ibuffer = msg.Buffer();
   auto isize = msg.BufferSize();
   TRequest req;
   MPI_Isend((void *)ibuffer, isize, MPI_CHAR, dest, tag, fComm, &req.fRequest);
   return req;
}


//______________________________________________________________________________
template<> TRequest TCommunicator::ISsend<TMpiMessage>(const TMpiMessage  &var, Int_t dest, Int_t tag)
{
   return ISsend(&var, 1, dest, tag);
}

//______________________________________________________________________________
template<> TRequest TCommunicator::ISsend<TMpiMessage>(const TMpiMessage  *vars, Int_t count, Int_t dest, Int_t tag)
{
   std::vector<TMpiMessageInfo>  msgis(count);
   for (auto i = 0; i < count; i++) {
      auto buffer = vars[i].Buffer();
      auto size   = vars[i].BufferSize();
      TMpiMessageInfo msgi(buffer, size);
      msgi.SetSource(GetRank());
      msgi.SetDestination(dest);
      msgi.SetTag(tag);
      msgi.SetDataTypeName(vars[i].GetDataTypeName());
      msgis[i] = msgi;
   }
   TMpiMessage msg;
   msg.WriteObject(msgis);
   auto ibuffer = msg.Buffer();
   auto isize = msg.BufferSize();
   TRequest req;
   MPI_Issend((void *)ibuffer, isize, MPI_CHAR, dest, tag, fComm, &req.fRequest);
   return req;
}

//______________________________________________________________________________
template<> TRequest TCommunicator::IRsend<TMpiMessage>(const TMpiMessage  &var, Int_t dest, Int_t tag)
{
   return IRsend(&var, 1, dest, tag);
}

//______________________________________________________________________________
template<> TRequest TCommunicator::IRsend<TMpiMessage>(const TMpiMessage  *vars, Int_t count, Int_t dest, Int_t tag)
{
   std::vector<TMpiMessageInfo>  msgis(count);
   for (auto i = 0; i < count; i++) {
      auto buffer = vars[i].Buffer();
      auto size   = vars[i].BufferSize();
      TMpiMessageInfo msgi(buffer, size);
      msgi.SetSource(GetRank());
      msgi.SetDestination(dest);
      msgi.SetTag(tag);
      msgi.SetDataTypeName(vars[i].GetDataTypeName());
      msgis[i] = msgi;
   }
   TMpiMessage msg;
   msg.WriteObject(msgis);
   auto ibuffer = msg.Buffer();
   auto isize = msg.BufferSize();
   TRequest req;
   MPI_Irsend((void *)ibuffer, isize, MPI_CHAR, dest, tag, fComm, &req.fRequest);
   return req;
}


//______________________________________________________________________________
template<> TRequest TCommunicator::IRecv<TMpiMessage>(TMpiMessage  &var, Int_t source, Int_t tag) const
{
   return IRecv(&var, 1, source, tag);
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

template<> void TCommunicator::Bcast<TMpiMessage>(TMpiMessage *vars, Int_t count, Int_t root) const
{
   std::vector<TMpiMessageInfo>  msgis(count);
   if (GetRank() == root) {
      for (auto i = 0; i < count; i++) {
         auto buffer = vars[i].Buffer();
         auto size   = vars[i].BufferSize();
         TMpiMessageInfo msgi(buffer, size);
         msgi.SetSource(GetRank());
         msgi.SetDestination(ANY_TAG);
         msgi.SetDataTypeName(vars[i].GetDataTypeName());
         msgis[i] = msgi;
      }
   }

   Bcast(msgis, root);

   for (auto i = 0; i < count; i++) {
      //passing information from TMpiMessageInfo to TMpiMessage
      auto size = msgis[i].GetBufferSize();
      Char_t *buffer = new Char_t[size];
      memcpy(buffer, msgis[i].GetBuffer(), size);

      vars[i].SetBuffer((void *)buffer, size, kFALSE);
      vars[i].SetReadMode();
      vars[i].Reset();
   }
}

//______________________________________________________________________________
template<> TGrequest TCommunicator::IBcast<TMpiMessage>(TMpiMessage &var, Int_t root) const
{

   IMsg *_imsg = new IMsg;
   _imsg->fMsg = &var;
   _imsg->fComm = &fComm;
   _imsg->fCommunicator = this;
   _imsg->fRoot = root;


   if (GetRank() == root) {

      //query lambda function
      auto query_fn = [](void *extra_state, TStatus & status)->Int_t {
         if (status.IsCancelled())
         {
            return MPI_ERR_IN_STATUS;
         }
         IMsg  *imsg = (IMsg *)extra_state;

         auto buffer = imsg->fMsg->Buffer();
         auto size   = imsg->fMsg->BufferSize();
         TMpiMessageInfo msgi(buffer, size);
         msgi.SetSource(imsg->fCommunicator->GetRank());
         msgi.SetDestination(-1);
         msgi.SetTag(imsg->fRoot);
         msgi.SetDataTypeName(imsg->fMsg->GetDataTypeName());


         TMpiMessage msg;
         msg.WriteObject(msgi);
         auto isize = msg.BufferSize();
         Char_t *ibuffer = new Char_t[isize];
         memcpy(ibuffer, msg.Buffer(), isize);
         MPI_Request req;
         MPI_Ibcast((void *)&isize, 1, MPI_INT, imsg->fRoot, *imsg->fComm, &req);
         MPI_Wait(&req, MPI_STATUS_IGNORE);
         MPI_Ibcast((void *)ibuffer, isize, MPI_CHAR, imsg->fRoot, *imsg->fComm, &req);
         MPI_Wait(&req, MPI_STATUS_IGNORE);

         Char_t *mbuffer = new Char_t[size];
         memcpy(mbuffer, buffer, size);

         //setting current message in read mode
         imsg->fMsg->SetBuffer((void *)mbuffer, size, kFALSE);
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
      auto cancel_fn = [](void */*extra_state*/, Bool_t complete)->Int_t {
         if (!complete)
         {
            std::cout << "incompleted!\n";
         }
         std::cout << "Cancelled!\n";
         return MPI_SUCCESS;
      };

      //creating General Request with lambda function
      return TGrequest::Start(query_fn, free_fn, cancel_fn, (void *)_imsg);

   }


   //query lambda function
   auto query_fn = [](void *extra_state, TStatus & status)->Int_t {
      Int_t isize = 0;
      if (status.IsCancelled())
      {
         return MPI_ERR_IN_STATUS;
      }
      IMsg  *imsg = (IMsg *)extra_state;
      MPI_Request req;
      MPI_Ibcast((void *)&isize, 1, MPI_INT, imsg->fRoot, *imsg->fComm, &req);
      MPI_Wait(&req, MPI_STATUS_IGNORE); //TODO:error control here

      Char_t *ibuffer = new Char_t[isize];
      MPI_Ibcast((void *)ibuffer, isize, MPI_CHAR, imsg->fRoot, *imsg->fComm, &req);
      MPI_Wait(&req, MPI_STATUS_IGNORE); //TODO:error control here

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
         Int_t isize = 0;
         IMsg  *imsg = (IMsg *)extra_state;
         MPI_Request req;
         MPI_Ibcast((void *)&isize, 1, MPI_INT, imsg->fSource, *imsg->fComm, &req);
         MPI_Wait(&req, MPI_STATUS_IGNORE); //TODO:error control here

         Char_t *ibuffer = new Char_t[isize];
         MPI_Ibcast((void *)ibuffer, isize, MPI_CHAR, imsg->fSource, *imsg->fComm, &req);
         MPI_Cancel(&req);
         delete ibuffer;
      }
      std::cout << "Cancelled!\n";
      return MPI_SUCCESS;
   };

   //creating General Request with lambda function
   return  TGrequest::Start(query_fn, free_fn, cancel_fn, (void *)_imsg);
}

template<> TRequest TCommunicator::IRecv<TMpiMessage>(TMpiMessage *vars, Int_t count, Int_t source, Int_t tag) const
{
   TRequest req;
   Int_t isize = 0;
   TStatus s;
   while (!IProbe(source, tag, s)) {
      gSystem->Sleep(100);
   }

   MPI_Get_elements(const_cast<MPI_Status *>(&s.fStatus), MPI_CHAR, &isize);

   Char_t *ibuffer = new Char_t[isize];
   MPI_Irecv((void *)ibuffer, isize, MPI_CHAR, source, tag, fComm, &req.fRequest);

   req.fUnserialize = [ = ]() {
      std::vector<TMpiMessageInfo>  msgis;

      TMpiMessage msg(ibuffer, isize);
      auto cl = gROOT->GetClass(typeid(std::vector<TMpiMessageInfo>));
      auto obj_tmp = msg.ReadObjectAny(cl);
      memcpy((void *)&msgis, (void *)obj_tmp, sizeof(std::vector<TMpiMessageInfo>));


      for (auto i = 0; i < count; i++) {
         //passing information from TMpiMessageInfo to TMpiMessage
         auto size = msgis[i].GetBufferSize();
         Char_t *buffer = new Char_t[size];
         memcpy(buffer, msgis[i].GetBuffer(), size);

         vars[i].SetBuffer((void *)buffer, size, kFALSE);
         vars[i].SetReadMode();
         vars[i].Reset();
      }

   };

   return req;
}



