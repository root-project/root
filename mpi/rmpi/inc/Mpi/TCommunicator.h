// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2016-2017 http://oproject.org
#ifndef ROOT_Mpi_TCommunicator
#define ROOT_Mpi_TCommunicator

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#include<TObject.h>


#ifndef ROOT_Mpi_TMpiMessage
#include <Mpi/TMpiMessage.h>
#endif

#ifndef ROOT_Mpi_TStatus
#include <Mpi/TStatus.h>
#endif

#ifndef ROOT_Mpi_TRequest
#include<Mpi/TRequest.h>
#endif

#include<memory>

#include<mpi.h>

/**
 @namespace ROOT::Mpi
 namespace associated RMpi package for ROOT.
 @defgroup Mpi Message Passing Interface
 */


namespace ROOT {

   namespace Mpi {

      class TMpiMessage;
      /**
      \class TComm
         Base internal class of communicator, with this class you can hanlde MPI::Comm and MPI_Comm from C/C++ mpi libraries.
         \ingroup Mpi
       */

      class TComm: public MPI::Comm {
      public:
         TComm(): Comm() {}
         TComm(const MPI_Comm &comm): Comm(comm) {}
         TComm(const MPI::Comm &comm): Comm(comm) {}

         TComm &Clone() const
         {
            MPI_Comm newcomm;
            MPI_Comm_dup((MPI_Comm)*this, &newcomm);
            TComm *comm = new TComm(newcomm);
            return *comm;
         }
      };

      /**
      \class TCommunicator
         Class for communicator, with this class you can to communicate the processes using messages,
      the messages can be any serializable object supported by ROOT like object from standart c++ libraries or
      objects that inherits from TObject.

      You can to create your own classes and communicate it just creating its dictionaries
         \ingroup Mpi
       */

      class TCommunicator: public TObject {
      private:
         TComm fComm;           //! Raw communicator
         Int_t fMainProcess;    // Rank used like a main process
         template<class T> MPI::Datatype GetDataType() const;
      public:
         /**
         Copy constructor for communicator
              \param comm other TCommunicator object
              */
         TCommunicator(const TCommunicator &comm);
         /**
         Default constructor for communicator
              \param comm TComm object with raw mpi communication object like MPI_Comm or MPI::Comm, by default MPI::COMM_WORLD
              */
         TCommunicator(const TComm &comm = MPI::COMM_WORLD);
         ~TCommunicator();

         /**
         Method to get the current rank or process id
              \return integer with the rank value
              */
         inline Int_t GetRank() const
         {
            return fComm.Get_rank();
         }

         /**
         Method to get the total number of ranks or processes
              \return integer with the number of processes
              */
         inline Int_t GetSize() const
         {
            return fComm.Get_size();
         }

         /**
         Method to know if the current rank us the main process
              \return boolean true if it is the main rank
              */
         inline Bool_t IsMainProcess() const
         {
            return fComm.Get_rank() == fMainProcess;
         }

         /**
         Method to set the main process rank
              \param Int_t main process rank number
              */
         inline void SetMainProcess(Int_t p)
         {
            fMainProcess = p;
         }

         /**
         Method to get the main process id
              \return integer with the main rank
              */
         inline Int_t GetMainProcess() const
         {
            return fMainProcess;
         }

         /**
         Method to abort  processes
              \param integer with error code
              */
#if OPEN_MPI
         inline void Abort(Int_t err)
#else
         inline void Abort(Int_t err) const
#endif
         {
            fComm.Abort(err);
         }

         /**
         Method for synchronization between MPI processes in a communicator
         */
         virtual void Barrier() const;

         /**
            Nonblocking test for a message. Operations  allow checking of incoming messages without actual receipt of them.
              \param source Source rank or ROOT::Mpi::ANY_SOURCE (integer).
              \param tag Tag value or ROOT::Mpi::ANY_TAG (integer).
              \param status TStatus object with extra information.
              \return boolean true if the probe if ok
              */
         virtual Bool_t Iprobe(Int_t source, Int_t tag, TStatus &status) const;

         /**
            Nonblocking test for a message. Operations  allow checking of incoming messages without actual receipt of them.
              \param source Source rank or ROOT::Mpi::ANY_SOURCE (integer).
              \param tag Tag value or ROOT::Mpi::ANY_TAG (integer).
              \return boolean true if the probe if ok
              */
         virtual Bool_t Iprobe(Int_t source, Int_t tag) const;

         /**
            Test for a message. Operations  allow checking of incoming messages without actual receipt of them.
              \param source Source rank or ROOT::Mpi::ANY_SOURCE (integer).
              \param tag Tag value or ROOT::Mpi::ANY_TAG (integer).
              \param status TStatus object with extra information.
              \return boolean true if the probe if ok
              */
         virtual void Probe(Int_t source, Int_t tag, TStatus &status) const;

         /**
            Test for a message. Operations  allow checking of incoming messages without actual receipt of them.
              \param source Source rank or ROOT::Mpi::ANY_SOURCE (integer).
              \param tag Tag value or ROOT::Mpi::ANY_TAG (integer).
              \return boolean true if the probe if ok
              */
         virtual void Probe(Int_t source, Int_t tag) const;

         /**
         Method to send a message for p2p communication
              \param var any selializable object
              \param dest id with the destination(Rank/Process) of the message
              \param tag id of the message
              */
         template<class Type> void Send(const Type &var, Int_t dest, Int_t tag) const;
         /**
         Method to receive a message for p2p communication
              \param var any selializable object reference to receive the message
              \param source id with the origin(Rank/Process) of the message
              \param tag id of the message
              */
         template<class Type>  void Recv(Type &var, Int_t source, Int_t tag) const; //must be changed by ROOOT::Mpi::TStatus& Recv(...)

         /**
            Starts a standard-mode, nonblocking send.
              \param var any selializable object
              \param dest id with the destination(Rank/Process) of the message
              \param tag id of the message
              */
         template<class Type> TRequest ISend(const Type &var, Int_t dest, Int_t tag);

         /**
         Starts a nonblocking synchronous send
              \param var any selializable object
              \param dest id with the destination(Rank/Process) of the message
              \param tag id of the message
              */
         template<class Type> TRequest ISsend(const Type &var, Int_t dest, Int_t tag);
         /**
         Starts a ready-mode nonblocking send.
              \param var any selializable object
              \param dest id with the destination(Rank/Process) of the message
              \param tag id of the message
              */
         template<class Type> TRequest IRsend(const Type &var, Int_t dest, Int_t tag);

         /**
         Method to receive a message from nonblocking send (ISend, ISsend, IRsend)
         to receive the object you need to call the methods Complete() and Wait()
         TGrequest req=comm.IRecv(..);
         req.Complete();
         req.Wait();

              \param var any selializable object reference to receive the message
              \param source id with the origin(Rank/Process) of the message
              \param tag id of the message
              \return TGrequest object.
              */
         template<class Type> TGrequest IRecv(Type &var, Int_t source, Int_t tag);


         /**
         Method to broadcast a message for collective communication
              \param var any selializable object reference to send/receive the message
              \param root id of the main message where message was sent
              */
         template<class Type> void Bcast(Type &var, Int_t root) const;

         ClassDef(TCommunicator, 1)
      };
      //Nonblocking message for callbacks
      struct IMsg {
         TMpiMessage *fMsg;
         TCommunicator *fCommunicator;
         TComm *fComm;
         Int_t fSource;
         Int_t fTag;
         void *fVar;
         UInt_t fSizeof;
         TClass *fClass;
      };


      template<class T> MPI::Datatype TCommunicator::GetDataType() const
      {
         if (typeid(T) == typeid(int) || typeid(T) == typeid(Int_t)) return MPI::INT;
         if (typeid(T) == typeid(float) || typeid(T) == typeid(Float_t)) return MPI::FLOAT;
         if (typeid(T) == typeid(double) || typeid(T) == typeid(Double_t)) return MPI::DOUBLE;
         if (typeid(T) == typeid(bool) || typeid(T) == typeid(Bool_t)) return MPI::BYTE;
         MPI::Datatype None;

         return None;
         //TODO: error control here if type is not supported
      }

      //______________________________________________________________________________
      template<class Type> void TCommunicator::Send(const Type &var, Int_t dest, Int_t tag) const
      {
         if (std::is_class<Type>::value) {
            TMpiMessage msg;
            msg.WriteObject(var);
            Send(msg, dest, tag);
         } else {
            fComm.Send(&var, 1, GetDataType<Type>(), dest, tag);
         }
      }


      //______________________________________________________________________________
      template<class Type>  void TCommunicator::Recv(Type &var, Int_t source, Int_t tag) const
      {
         if (std::is_class<Type>::value) {
            TMpiMessage msg;
            Recv(msg, source, tag);

            auto cl = gROOT->GetClass(typeid(var));
            auto obj_tmp = (Type *)msg.ReadObjectAny(cl);
            memcpy((void *)&var, (void *)obj_tmp, sizeof(Type));
         } else {
            fComm.Recv(&var, 1, GetDataType<Type>(), source, tag);
         }
      }


      //______________________________________________________________________________
      template<class Type> TRequest TCommunicator::ISend(const Type &var, Int_t dest, Int_t tag)
      {
         TRequest req;
         if (std::is_class<Type>::value) {
            TMpiMessage msg;
            msg.WriteObject(var);
            req = ISend(msg, dest, tag);
         } else {
            req = fComm.Isend(&var, 1, GetDataType<Type>(), dest, tag);
         }
         return req;
      }

      //______________________________________________________________________________
      template<class Type> TRequest TCommunicator::ISsend(const Type &var, Int_t dest, Int_t tag)
      {
         TRequest req;
         if (std::is_class<Type>::value) {
            TMpiMessage msg;
            msg.WriteObject(var);
            req = ISsend(msg, dest, tag);
         } else {
            req = fComm.Issend(&var, 1, GetDataType<Type>(), dest, tag);
         }
         return req;
      }

      //______________________________________________________________________________
      template<class Type> TRequest TCommunicator::IRsend(const Type &var, Int_t dest, Int_t tag)
      {
         TRequest req;
         if (std::is_class<Type>::value) {
            TMpiMessage msg;
            msg.WriteObject(var);
            req = IRsend(msg, dest, tag);
         } else {
            req = fComm.Irsend(&var, 1, GetDataType<Type>(), dest, tag);
         }
         return req;
      }


      //______________________________________________________________________________
      template<class Type> TGrequest TCommunicator::IRecv(Type &var, Int_t source, Int_t tag)
      {
         TGrequest req;
         if (std::is_class<Type>::value) {
            IMsg *_imsg = new IMsg;
            _imsg->fVar = &var;
            _imsg->fCommunicator = this;
            _imsg->fComm = &fComm;
            _imsg->fSource = source;
            _imsg->fTag = tag;
            _imsg->fSizeof = sizeof(Type);
            _imsg->fClass = gROOT->GetClass(typeid(var));

            //query lambda function
            auto query_fn = [](void *extra_state, TStatus & status)->Int_t {

               if (status.IsCancelled())
               {
                  return MPI_ERR_IN_STATUS;
               }
               IMsg  *imsg = (IMsg *)extra_state;
               TMpiMessage msg;
               auto ireq = imsg->fCommunicator->IRecv(msg, imsg->fSource, imsg->fTag);
               TStatus s;
               ireq.GetStatus(s);
               if (s.IsCancelled())
               {
                  return MPI_ERR_IN_STATUS;
               }
               ireq.Complete();
               ireq.Wait();
               auto obj_tmp = (Type *)msg.ReadObjectAny(imsg->fClass);
               memcpy(imsg->fVar, (void *)obj_tmp, imsg->fSizeof);
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
               return MPI_SUCCESS;
            };
            req = TGrequest::Start(query_fn, free_fn, cancel_fn, (void *)_imsg);
         } else {
            req = fComm.Irecv(&var, 1, GetDataType<Type>(), source, tag);
         }
         return req;
      }

      //______________________________________________________________________________
      template<class Type> void TCommunicator::Bcast(Type &var, Int_t root) const
      {
         if (std::is_class<Type>::value) {
            TMpiMessage msg;
            if (GetRank() == root) {
               msg.WriteObject(var);
            }
            Bcast(msg, root);

            if (GetRank() != root) {
               auto cl = gROOT->GetClass(typeid(var));
               auto obj_tmp = (Type *)msg.ReadObjectAny(cl);
               memcpy((void *)&var, (void *)obj_tmp, sizeof(Type));
            }

         } else {
            fComm.Bcast(&var, 1, GetDataType<Type>(), root);
         }

      }

      ////////////////////////////////
      //specialized template methods
      //______________________________________________________________________________
      template<> void TCommunicator::Send<TMpiMessage>(const TMpiMessage &var, Int_t dest, Int_t tag) const;
      //______________________________________________________________________________
      template<> void TCommunicator::Recv<TMpiMessage>(TMpiMessage &var, Int_t source, Int_t tag) const;
      //______________________________________________________________________________
      template<> void TCommunicator::Bcast<TMpiMessage>(TMpiMessage &var, Int_t root) const;
      //______________________________________________________________________________
      template<> TRequest TCommunicator::ISend<TMpiMessage>(const TMpiMessage  &var, Int_t dest, Int_t tag);
      //______________________________________________________________________________
      template<> TRequest TCommunicator::ISsend<TMpiMessage>(const TMpiMessage  &var, Int_t dest, Int_t tag);
      //______________________________________________________________________________
      template<> TRequest TCommunicator::IRsend<TMpiMessage>(const TMpiMessage  &var, Int_t dest, Int_t tag);
      //______________________________________________________________________________
      template<> TGrequest TCommunicator::IRecv<TMpiMessage>(TMpiMessage  &var, Int_t source, Int_t tag);

   }

}

#endif
