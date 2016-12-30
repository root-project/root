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
         TComm fComm;
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

         template<class ClassType> void Send(ClassType &obj, Int_t dest, Int_t tag) const;
         template<class ClassType>  void Recv(ClassType &obj, Int_t source, Int_t tag) const; //must be changed by ROOOT::Mpi::TStatus& Recv(...)

         ClassDef(TCommunicator, 1)
      };

      template<class ClassType> void TCommunicator::Send(ClassType &obj, Int_t dest, Int_t tag) const
      {
         TMpiMessage msg;
         msg.WriteObject(obj);
         const Char_t *buffer = msg.Buffer();
         const UInt_t size = msg.BufferSize();
         fComm.Send(&size, 1, MPI::INT, dest, tag);
         fComm.Send(buffer, size, MPI::CHAR, dest, tag);
      }
      template<class ClassType>  void TCommunicator::Recv(ClassType &obj, Int_t source, Int_t tag) const
      {
         UInt_t size = 0;
         MPI::COMM_WORLD.Recv(&size, 1, MPI::INT, source, tag);

         Char_t *buffer = new Char_t[size];
         MPI::COMM_WORLD.Recv(buffer, size, MPI::CHAR, source, tag);


         TMpiMessage msg(buffer, size);
         TClass *cl = gROOT->GetClass(typeid(obj));
         ClassType *obj_tmp = (ClassType *)msg.ReadObjectAny(cl);
         memcpy(&obj, obj_tmp, sizeof(ClassType));
      }


   }

}

#endif
