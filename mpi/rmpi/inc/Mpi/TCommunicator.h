// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2016 http://oproject.org
#ifndef ROOT_Mpi_TCommunicator
#define ROOT_Mpi_TCommunicator

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#include<TObject.h>


#ifndef ROOT_TMpiMessage
#include <Mpi/TMpiMessage.h>
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
         inline void Abort(Int_t err) const
         {
              fComm.Abort(err);
         }
         

         /**
         Method to send a message for p2p communication
              \param var any selializable object
              \param dest id with the destination(Rank/Process) of the message
              \param tag id of the message
              */
         template<class Type> void Send(Type &var, Int_t dest, Int_t tag) const;
         /**
         Method to receive a message for p2p communication
              \param var any selializable object reference to receive the message
              \param source id with the origin(Rank/Process) of the message
              \param tag id of the message
              */
         template<class Type>  void Recv(Type &var, Int_t source, Int_t tag) const; //must be changed by ROOOT::Mpi::TStatus& Recv(...)

         /**
         Method to broadcast a message for collective communication
              \param var any selializable object reference to send/receive the message
              \param root id of the main message where message was sent
              */
         template<class Type> void Bcast(Type &var, Int_t root) const;

         ClassDef(TCommunicator, 1)
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
      template<class Type> void TCommunicator::Send(Type &var, Int_t dest, Int_t tag) const
      {
         if (std::is_class<Type>::value) {
            TMpiMessage msg;
            msg.WriteObject(var);
            const Char_t *buffer = msg.Buffer();
            const UInt_t size = msg.BufferSize();
            fComm.Send(&size, 1, MPI::INT, dest, tag);
            fComm.Send(buffer, size, MPI::CHAR, dest, tag);
         } else {
            fComm.Send(&var, 1, GetDataType<Type>(), dest, tag);
         }
      }

      //______________________________________________________________________________
      template<class Type>  void TCommunicator::Recv(Type &var, Int_t source, Int_t tag) const
      {
         if (std::is_class<Type>::value) {
            UInt_t size = 0;
            fComm.Recv(&size, 1, MPI::INT, source, tag);

            Char_t *buffer = new Char_t[size];
            fComm.Recv(buffer, size, MPI::CHAR, source, tag);


            TMpiMessage msg(buffer, size);
            auto cl = gROOT->GetClass(typeid(var));
            auto obj_tmp = (Type *)msg.ReadObjectAny(cl);
            memcpy((void *)&var, (void *)obj_tmp, sizeof(Type));
         } else {
            fComm.Recv(&var, 1, GetDataType<Type>(), source, tag);
         }
      }

      //______________________________________________________________________________
      template<class Type> void TCommunicator::Bcast(Type &var, Int_t root) const
      {
         if (std::is_class<Type>::value) {
            Char_t *buffer = nullptr ;
            UInt_t size = 0;
            if (GetRank() == root) {
               TMpiMessage msg;
               msg.WriteObject(var);
               size = msg.BufferSize();
               buffer = new Char_t[size];
               memcpy(buffer, msg.Buffer(), size * sizeof(Char_t));
            }
            fComm.Bcast(&size, 1, MPI::INT, root);
            if (GetRank() != root) {
               buffer = new Char_t[size];
            }
            fComm.Bcast(buffer, size, MPI::CHAR, root);
            if (GetRank() != root) {
               TMpiMessage msg(buffer, size);
               auto cl = gROOT->GetClass(typeid(var));
               auto obj_tmp = (Type *)msg.ReadObjectAny(cl);
               memcpy((void *)&var, (void *)obj_tmp, sizeof(Type));
            }

         } else {
            fComm.Bcast(&var, 1, GetDataType<Type>(), root);
         }

      }

   }

}

#endif
