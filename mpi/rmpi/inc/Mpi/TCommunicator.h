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

namespace ROOT {

   namespace Mpi {

      class TMpiMessage;

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

      class TCommunicator: public TObject {
      private:
         TComm fComm;
         template<class T> MPI::Datatype GetDataType() const;
      public:
         TCommunicator(const TCommunicator &comm);
         TCommunicator(const MPI::Comm &comm = MPI::COMM_WORLD);
         ~TCommunicator();

         inline Int_t GetRank() const
         {
            return fComm.Get_rank();
         }
         inline Int_t GetSize() const
         {
            return fComm.Get_size();
         }

         template<class Type> void Send(Type &var, Int_t dest, Int_t tag) const;
         template<class Type>  void Recv(Type &var, Int_t source, Int_t tag) const; //must be changed by ROOOT::Mpi::TStatus& Recv(...)

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
            memcpy((void*)&var,(void*)obj_tmp, sizeof(Type));
         } else {
            fComm.Recv(&var, 1, GetDataType<Type>(), source, tag);
         }
      }

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
	       memcpy(buffer,msg.Buffer(), size * sizeof(Char_t));
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
               memcpy((void*)&var,(void*)obj_tmp, sizeof(Type));
            }

         } else {
            fComm.Bcast(&var, 1, GetDataType<Type>(), root);
         }

      }

   }

}

#endif
