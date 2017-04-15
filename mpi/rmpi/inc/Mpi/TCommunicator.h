// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2016-2017 http://oproject.org
#ifndef ROOT_Mpi_TCommunicator
#define ROOT_Mpi_TCommunicator

#include<Mpi/Globals.h>
#include<Mpi/TErrorHandler.h>
#include<Mpi/TMpiMessage.h>
#include<Mpi/TStatus.h>
#include<Mpi/TRequest.h>
#include<Mpi/TGroup.h>


/**
 * @namespace ROOT::Mpi
 * namespace associated RMpi package for ROOT.
 * @defgroup Mpi Message Passing Interface
 */


namespace ROOT {

   namespace Mpi {

      class TMpiMessage;
      class TInterCommunicator;
      class TIntraCommunicator;
      /**
       * \class TNullCommunicator
       * Class for null communicator, base class to create by default a null communicator in any communicator class.
       * \ingroup Mpi
       */

      class TNullCommunicator: public TObject {
      protected:
         MPI_Comm fComm;           //! Raw communicator
      public:
         TNullCommunicator() : fComm(MPI_COMM_NULL) { }

         TNullCommunicator(const TNullCommunicator &comm) : TObject(comm), fComm(comm.fComm) { }

         TNullCommunicator(const MPI_Comm &comm) : fComm(comm) { }

         virtual inline ~TNullCommunicator() { }

         inline Bool_t operator==(const TNullCommunicator &comm) const
         {
            return (Bool_t)(fComm == comm.fComm);
         }

         inline Bool_t operator!=(const TNullCommunicator &comm) const
         {
            return (Bool_t) !(*this == comm);
         }

         inline operator MPI_Comm() const
         {
            return fComm;
         }

         ClassDef(TNullCommunicator, 1)
      };


      /**
       * \class TCommunicator
       * Class for communicator, with this class you can to communicate the processes using messages,
       * the messages can be any serializable object supported by ROOT like object from standart c++ libraries or
       * objects that inherits from TObject.
       *
       * You can to create your own classes and communicate it just creating its dictionaries
       * \ingroup Mpi
       */

      class TCommunicator: public TNullCommunicator {
      protected:
         Int_t GetInternalTag() const;
      public:
         TCommunicator();
         TCommunicator(const TCommunicator &comm);
         TCommunicator(const MPI_Comm &comm);
         ~TCommunicator();

         TCommunicator &operator=(const MPI_Comm &comm)
         {
            fComm = comm;
            return *this;
         }

         MPI_Comm &operator=(const TCommunicator &comm) const;
         MPI_Comm &operator=(const TInterCommunicator &comm) const;
         MPI_Comm &operator=(const TIntraCommunicator &comm) const;

         virtual TCommunicator &Clone() const = 0;

         Int_t GetRank() const;

         Int_t GetSize() const;

         Bool_t IsMainProcess() const;

         Int_t GetMainProcess() const;

         void Abort(Int_t error) const;

         Int_t GetMaxTag() const;

         virtual TString GetCommName() const;

         virtual void SetCommName(const TString name);


         virtual void Barrier() const;

         virtual void IBarrier(TRequest &req) const;

         virtual Bool_t IProbe(Int_t source, Int_t tag, TStatus &status) const;

         virtual Bool_t IProbe(Int_t source, Int_t tag) const;

         virtual void Probe(Int_t source, Int_t tag, TStatus &status) const;

         virtual void Probe(Int_t source, Int_t tag) const;

         ////////////////////////////////////////
         //utility methods with single argument//
         ////////////////////////////////////////

         template<class Type> void Send(const Type &var, Int_t dest, Int_t tag) const;

         template<class Type>  void Recv(Type &var, Int_t source, Int_t tag) const; //must be changed by ROOOT::Mpi::TStatus& Recv(...)

         template<class Type> TRequest ISend(const Type &var, Int_t dest, Int_t tag);

         template<class Type> TRequest ISsend(const Type &var, Int_t dest, Int_t tag);

         template<class Type> TRequest IRsend(const Type &var, Int_t dest, Int_t tag);

         template<class Type> TRequest IRecv(Type &var, Int_t source, Int_t tag) const;

         template<class Type> void Bcast(Type &var, Int_t root) const;

         template<class Type> TRequest IBcast(Type &var, Int_t root) const;

         template<class Type> void Reduce(const Type &in_var, Type &out_var, Op<Type> (*opf)(), Int_t root) const;


         ////////////////////////////////
         //methods with arrar arguments//
         ////////////////////////////////

         template<class Type> void Send(const Type *vars, Int_t count, Int_t dest, Int_t tag) const;

         template<class Type>  void Recv(Type *vars, Int_t count, Int_t source, Int_t tag) const;

         //methods with nonblocking//

         template<class Type> TRequest ISend(const Type *vars, Int_t count, Int_t dest, Int_t tag);

         template<class Type> TRequest ISsend(const Type *vars, Int_t count, Int_t dest, Int_t tag);

         template<class Type> TRequest IRsend(const Type *vars, Int_t count, Int_t dest, Int_t tag);

         template<class Type> TRequest IRecv(Type *vars, Int_t count, Int_t source, Int_t tag) const;

         template<class Type> TRequest IBcast(Type *vars, Int_t count, Int_t root) const;

         template<class Type> void Bcast(Type *vars, Int_t count, Int_t root) const;

         template<class Type> void Reduce(const Type *in_vars, Type *out_vars, Int_t count, Op<Type> (*opf)(), Int_t root) const;


         template<class Type> void Scatter(const Type *in_vars, Int_t incount, Type *out_vars, Int_t outcount, Int_t root) const;

         template<class Type> void Gather(const Type *in_vars, Int_t incount, Type *out_vars, Int_t outcount, Int_t root) const;

         /////////////////////////////////////
         //methods with results in all ranks//
         /////////////////////////////////////

         template<class Type> void AllReduce(const Type *in_vars, Type *out_vars, Int_t count, Op<Type> (*opf)()) const;

         template<class Type> void AllGather(const Type *in_vars, Int_t incount, Type *out_vars, Int_t outcount) const;


         /////////////////////////////////////////
         // Groups, Contexts, and Communicators //
         /////////////////////////////////////////

         TGroup GetGroup() const;

         static Int_t Compare(const TCommunicator &comm1, const TCommunicator &comm2);

         Int_t Compare(const TCommunicator &comm2);

         virtual void Free(void);

         virtual Bool_t IsInter() const;

         //////////////////////
         // Process Creation //
         //////////////////////

         virtual void Disconnect();

         static TInterCommunicator GetParent();

         static TInterCommunicator Join(const Int_t fd);

         /**
          * static method to serialize objects. used in the multiple communication schemas.
          * \param buffer double pointer to Char_t to save the serialized data
          * \param size   reference to Int_t with the size of the buffer with serialized data
          * \param vars   any selializable object
          * \param count  number of elements to serialize in \p in_vars
          * \param comm   communicator object
          * \param dest   (optional) destination of the serialized information, must be the same unserializing
          * \param source (optional) source of the serialized information, must be the same unserializing
          * \param tag    (optional) tag of the serialized information, must be the same unserializing
          * \param root   (optional) root of collective operation, must be the same unserializing
          */
         template<class T> static void Serialize(Char_t **buffer, Int_t &size, const T *vars, Int_t count, const TCommunicator *comm, Int_t dest = 0, Int_t source = 0, Int_t tag = 0, Int_t root = 0)
         {
            std::vector<TMpiMessageInfo> msgis(count);
            for (auto i = 0; i < count; i++) {
               TMpiMessage msg;
               msg.WriteObject(vars[i]);
               auto mbuffer = msg.Buffer();
               auto msize   = msg.BufferSize();
               if (mbuffer == NULL) {
                  comm->Error(__FUNCTION__, "Error serializing object type %s \n", ROOT_MPI_TYPE_NAME(T));
                  comm->Abort(ERR_BUFFER);
               }
               TMpiMessageInfo msgi(mbuffer, msize);
               msgi.SetSource(comm->GetRank());
               msgi.SetDestination(dest);
               msgi.SetSource(source);
               msgi.SetRoot(root);
               msgi.SetTag(tag);
               msgi.SetDataTypeName(ROOT_MPI_TYPE_NAME(T));
               msgis[i] = msgi;
            }
            TMpiMessage msg;
            msg.WriteObject(msgis);
            auto ibuffer = msg.Buffer();
            size = msg.BufferSize();
            *buffer = new Char_t[size];
            if (ibuffer == NULL) {
               comm->Error(__FUNCTION__, "Error serializing object type %s \n", ROOT_MPI_TYPE_NAME(msgis));
               comm->Abort(ERR_BUFFER);
            }
            memcpy(*buffer, ibuffer, size);
         }

         /**
          * static method to unserialize objects. used in the multiple communication schemas.
          * \param buffer pointer to Char_t to read the serialized data
          * \param size   size of the buffer with serialized data
          * \param vars   any selializable object
          * \param count  number of elements to serialize in \p in_vars
          * \param comm   communicator object
          * \param dest   (optional) destination of the serialized information, must be the same serializing
          * \param source (optional) source of the serialized information, must be the same serializing
          * \param tag    (optional) tag of the serialized information, must be the same serializing
          * \param root   (optional) root of collective operation, must be the same serializing
          */
         template<class T> static  void Unserialize(Char_t *buffer, Int_t size, T *vars, Int_t count, const TCommunicator *comm, Int_t dest = 0, Int_t source = 0, Int_t tag = 0, Int_t root = 0)
         {
            TMpiMessage msg(buffer, size);
            auto cl = gROOT->GetClass(typeid(std::vector<TMpiMessageInfo>));
            auto msgis = (std::vector<TMpiMessageInfo> *)msg.ReadObjectAny(cl);
            if (msgis == NULL) {
               comm->Error(__FUNCTION__, "Error unserializing object type %s \n", cl->GetName());
               comm->Abort(ERR_BUFFER);
            }

            if (msgis->data()->GetDataTypeName() != ROOT_MPI_TYPE_NAME(T)) {
               comm->Error(__FUNCTION__, "Error unserializing objects type %s where objects are %s \n", ROOT_MPI_TYPE_NAME(T), msgis->data()->GetDataTypeName().Data());
               comm->Abort(ERR_TYPE);
            }

            ROOT_MPI_ASSERT(msgis->data()->GetDestination() == dest, comm)
            ROOT_MPI_ASSERT(msgis->data()->GetSource() == source, comm)
            ROOT_MPI_ASSERT(msgis->data()->GetRoot() == root, comm)
            ROOT_MPI_ASSERT(msgis->data()->GetTag() == tag, comm)

            for (auto i = 0; i < count; i++) {
               //passing information from TMpiMessageInfo to TMpiMessage
               auto isize = msgis->data()[i].GetBufferSize();
               Char_t *ibuffer = new Char_t[isize];//this memory dies when the unserialized object dies
               memcpy(ibuffer, msgis->data()[i].GetBuffer(), isize);
               TMpiMessage vmsg(ibuffer, isize);
               auto vcl = gROOT->GetClass(typeid(T));
               auto vobj_tmp = vmsg.ReadObjectAny(vcl);
               if (vobj_tmp == NULL) {
                  comm->Error(__FUNCTION__, "Error unserializing objects type %s \n", vcl->GetName());
                  comm->Abort(ERR_BUFFER);
               }
               memmove((void *)&vars[i], vobj_tmp, sizeof(T));
            }
         }
         ClassDef(TCommunicator, 2)
      };

      //______________________________________________________________________________
      /**
          * Method to send a message for p2p communication
          * \param var any selializable object
          * \param dest id with the destination(Rank/Process) of the message
          * \param tag id of the message
          */
      template<class Type> void TCommunicator::Send(const Type &var, Int_t dest, Int_t tag) const
      {
         Send(&var, 1, dest, tag);
      }

      //______________________________________________________________________________
      /**
          * Method to send a message for p2p communication
          * \param vars any selializable object
          * \param count number of elements in array \p vars
          * \param dest id with the destination(Rank/Process) of the message
          * \param tag id of the message
          */
      template<class Type> void TCommunicator::Send(const Type *vars, Int_t count, Int_t dest, Int_t tag) const
      {
         if (std::is_class<Type>::value) {
            Char_t *buffer;
            Int_t size;
            Serialize(&buffer, size, vars, count, this, dest, GetRank(), tag);
            MPI_Send(buffer, size, MPI_CHAR, dest, tag, fComm);
            delete buffer;
         } else {
            ROOT_MPI_CHECK_DATATYPE(Type);
            MPI_Send((void *)vars, count, GetDataType<Type>(), dest, tag, fComm);
         }
      }


      //______________________________________________________________________________
      /**
      * Method to receive a message for p2p communication
      * \param var any selializable object reference to receive the message
      * \param source id with the origin(Rank/Process) of the message
      * \param tag id of the message
      */
      template<class Type>  void TCommunicator::Recv(Type &var, Int_t source, Int_t tag) const
      {
         Recv(&var, 1, source, tag);
      }

      //______________________________________________________________________________
      /**
       * Method to receive a message for p2p communication
       * \param vars array of any selializable objects
       * \param count number of elements in array \p vars
       * \param source id with the origin(Rank/Process) of the message
       * \param tag id of the message
       */

      template<class Type>  void TCommunicator::Recv(Type *vars, Int_t count, Int_t source, Int_t tag) const
      {
         if (std::is_class<Type>::value) {
            Int_t size;
            TStatus s;
            Probe(source, tag, s);

            MPI_Get_elements(&s.fStatus, MPI_CHAR, &size);

            Char_t *buffer = new Char_t[size];
            MPI_Recv(buffer, size, MPI_CHAR, source, tag, fComm, MPI_STATUS_IGNORE);
            Unserialize<Type>(buffer, size, vars, count, this, GetRank(), source, tag, 0);

         } else {
            ROOT_MPI_CHECK_DATATYPE(Type);
            //TODO: added status argument to this method
            MPI_Recv((void *)vars, count, GetDataType<Type>(), source, tag, fComm, MPI_STATUS_IGNORE);
         }
      }

      //______________________________________________________________________________
      /**
       *    Starts a standard-mode, nonblocking send.
       * \param var any selializable object
       * \param dest id with the destination(Rank/Process) of the message
       * \param tag id of the message
       */
      template<class Type> TRequest TCommunicator::ISend(const Type &var, Int_t dest, Int_t tag)
      {
         return ISend(&var, 1, dest, tag);
      }

      //______________________________________________________________________________
      /**
       *    Starts a standard-mode, nonblocking send.
       * \param vars any selializable object
       * \param count number of elements in array \p vars
       * \param dest id with the destination(Rank/Process) of the message
       * \param tag id of the message
       */
      template<class Type> TRequest TCommunicator::ISend(const Type *vars, Int_t count, Int_t dest, Int_t tag)
      {
         TRequest req;
         if (std::is_class<Type>::value) {
            Char_t *buffer;
            Int_t size;
            Serialize(&buffer, size, vars, count, this, dest, GetRank(), tag);
            MPI_Isend(buffer, size, MPI_CHAR, dest, tag, fComm, &req.fRequest);
            req.fCallback = [buffer]()mutable { //use to clean memory after wait
               if (buffer) delete buffer;
               buffer = NULL;
            };
         } else {
            ROOT_MPI_CHECK_DATATYPE(Type);
            MPI_Isend((void *)vars, count, GetDataType<Type>(), dest, tag, fComm, &req.fRequest);
         }
         return req;
      }

      //______________________________________________________________________________
      /**
       * Starts a nonblocking synchronous send
       * \param var any selializable object
       * \param dest id with the destination(Rank/Process) of the message
       * \param tag id of the message
       */

      template<class Type> TRequest TCommunicator::ISsend(const Type &var, Int_t dest, Int_t tag)
      {
         return ISsend(&var, 1, dest, tag);
      }

      //______________________________________________________________________________
      /**
       * Starts a nonblocking synchronous send
       * \param vars any selializable object
       * \param count number of elements in array \p vars
       * \param dest id with the destination(Rank/Process) of the message
       * \param tag id of the message
       */
      template<class Type> TRequest TCommunicator::ISsend(const Type *vars, Int_t count, Int_t dest, Int_t tag)
      {
         TRequest req;
         if (std::is_class<Type>::value) {
            Char_t *buffer;
            Int_t size;
            Serialize(&buffer, size, vars, count, this, dest, GetRank(), tag);
            MPI_Issend(buffer, size, MPI_CHAR, dest, tag, fComm, &req.fRequest);
            req.fCallback = [buffer]()mutable { //use to clean memory after wait
               if (buffer) delete buffer;
               buffer = NULL;
            };
         } else {
            ROOT_MPI_CHECK_DATATYPE(Type);
            MPI_Issend((void *)vars, count, GetDataType<Type>(), dest, tag, fComm, &req.fRequest);
         }
         return req;
      }

      //______________________________________________________________________________
      /**
       * Starts a ready-mode nonblocking send.
       * \param var any selializable object
       * \param dest id with the destination(Rank/Process) of the message
       * \param tag id of the message
       */

      template<class Type> TRequest TCommunicator::IRsend(const Type &var, Int_t dest, Int_t tag)
      {
         return IRsend(&var, 1, dest, tag);
      }

      //______________________________________________________________________________
      /**
       * Starts a ready-mode nonblocking send.
       * \param vars any selializable object
       * \param count number of elements in array \p vars
       * \param dest id with the destination(Rank/Process) of the message
       * \param tag id of the message
       */
      template<class Type> TRequest TCommunicator::IRsend(const Type *vars, Int_t count, Int_t dest, Int_t tag)
      {
         TRequest req;
         if (std::is_class<Type>::value) {
            //TODO: objects is not sopported for ready mode,
            // because you need to call firts the IRecv method and the size of serialized buffer is unknow then
            // ADDED A GOOD ERROR HANDLING HERE!
            //
         } else {
            ROOT_MPI_CHECK_DATATYPE(Type);
            MPI_Irsend((void *)vars, count, GetDataType<Type>(), dest, tag, fComm, &req.fRequest);
         }
         return req;
      }

      //______________________________________________________________________________
      /**
       * Method to receive a message from nonblocking send (ISend, ISsend, IRsend)
       * to receive the object you need to call the methods Test() or Wait()
       * TRequest req=comm.IRecv(..);
       * req.Wait();
       *
       * \param var any selializable object reference to receive the message
       * \param source id with the origin(Rank/Process) of the message
       * \param tag id of the message
       * \return TRequest object.
       */
      template<class Type> TRequest TCommunicator::IRecv(Type &var, Int_t source, Int_t tag) const
      {
         return IRecv(&var, 1, source, tag);
      }

      //______________________________________________________________________________
      /**
       * Method to receive a message from nonblocking send (ISend, ISsend, IRsend)
       * to receive the object you need to call the methods Test() or Wait()
       * TRequest req=comm.IRecv(..);
       * req.Wait();
       *
       * \param vars any selializable object reference to receive the message
       * \param count number of elements in array \p vars
       * \param source id with the origin(Rank/Process) of the message
       * \param tag id of the message
       * \return TRequest object.
       */
      template<class Type> TRequest TCommunicator::IRecv(Type *vars, Int_t count, Int_t source, Int_t tag) const
      {
         TRequest req;
         if (std::is_class<Type>::value) {
            Int_t size;
            TStatus s;
            while (!IProbe(source, tag, s)) {
               gSystem->Sleep(100);
            }
            MPI_Get_elements(&s.fStatus, MPI_CHAR, &size);

            Char_t *buffer = new Char_t[size];
            MPI_Irecv(buffer, size, MPI_CHAR, source, tag, fComm, &req.fRequest);

            req.fCallback = std::bind(Unserialize<Type>, buffer, size, vars, count, this, GetRank(), source, tag, 0);

         } else {
            ROOT_MPI_CHECK_DATATYPE(Type);
            MPI_Irecv((void *)vars, count, GetDataType<Type>(), source, tag, fComm, &req.fRequest);
         }
         return req;
      }

      //______________________________________________________________________________
      /**
      *  Broadcasts a message from the process with rank root to all other processes of the group.
      * \param var any selializable object reference to send/receive the message
      * \param root id of the main message where message was sent
      */
      template<class Type> void TCommunicator::Bcast(Type &var, Int_t root) const
      {
         Bcast(&var, 1, root);
      }

      //______________________________________________________________________________
      /**
       *  Broadcasts a message from the process with rank root to all other processes of the group.
       * \param vars any selializable objects pointer to send/receive the message
       * \param count Number of elements to broadcast in \p in_vars
       * \param root id of the main message where message was sent
       */
      template<class Type> void TCommunicator::Bcast(Type *vars, Int_t count, Int_t root) const
      {
         if (std::is_class<Type>::value) {
            Int_t size;
            Char_t *buffer;

            if (GetRank() == root) Serialize(&buffer, size, vars, count, this, 0, 0, 0, root);

            Bcast(size, root);

            if (GetRank() != root) {
               buffer = new Char_t[size];
            }

            Bcast(buffer, size, root);

            Unserialize(buffer, size, vars, count, this, 0, 0, 0, root);

         } else {
            ROOT_MPI_CHECK_DATATYPE(Type);
            MPI_Bcast((void *)vars, count, GetDataType<Type>(), root, fComm);
         }

      }

      //______________________________________________________________________________
      /**
       *  Broadcasts a message from the process with rank root to all other processes of the group.
       * \param var any selializable object reference to send/receive the message
       * \param root id of the main message where message was sent
       * \return TGrequest obj
       */
      template<class Type> TRequest TCommunicator::IBcast(Type &var, Int_t root) const
      {
         return IBcast(&var, 1, root);
      }

      //______________________________________________________________________________
      /**
       *  Broadcasts a message from the process with rank root to all other processes of the group.
       * \param vars any selializable object reference to send/receive the message
       * \param count number of elements in array \p vars
       * \param root id of the main message where message was sent
       * \return TRequest obj
       */
      template<class Type> TRequest TCommunicator::IBcast(Type *vars, Int_t count, Int_t root) const
      {
         //NOTE: may is good idea to consider to implement tree broadcast algorithm,
         //because I am sending just one integer with the size firts
         TRequest req;
         if (std::is_class<Type>::value) {
            TRequest prereq;
            Int_t size;
            Char_t *buffer;

            if (GetRank() == root) {
               Serialize(&buffer, size, vars, count, this, 0, 0, 0, root);
            }

            prereq = IBcast(size, root);
            prereq.Wait();

            if (GetRank() != root) buffer = new Char_t[size];
            req = IBcast(buffer, size, root);
            req.fCallback = std::bind(Unserialize<Type>, buffer, size, vars, count, this, 0, 0, 0, root);

         } else {
            ROOT_MPI_CHECK_DATATYPE(Type);
            MPI_Ibcast((void *)vars, count, GetDataType<Type>(), root, fComm, &req.fRequest);
         }
         return req;
      }


      //______________________________________________________________________________
      /**
       *  Sends data from one task to all tasks in a group.
       * \param in_vars any selializable object vector reference to send the message
       * \param incount Number of elements in receive in \p in_vars
       * \param out_vars any selializable object vector reference to receive the message
       * \param outcount Number of elements in receive in \p out_vars
       * \param root id of the main message where message was sent
       * \return TGrequest obj
       */

      template<class Type> void TCommunicator::Scatter(const Type *in_vars, Int_t incount, Type *out_vars, Int_t outcount, Int_t root) const
      {
         if (GetRank() == root) {
            if (incount % (GetSize()*outcount) != 0) {
               Fatal("TCommunicator::Scatter", "Number of elements sent and elements in receive are not divisible. Can't no split to scatter message");
               Abort(ERR_COUNT);
            }
            for (auto i = 0 ; i < GetSize(); i++) {
               if (i == root) continue;
               auto stride = outcount * i;
               Send(&in_vars[stride], outcount, i, GetInternalTag());
            }
            auto stride = outcount * root;
            memcpy((void *)out_vars, (void *)&in_vars[stride], sizeof(Type)*outcount);
         } else {
            Recv(out_vars, outcount, root, GetInternalTag());
         }
      }

      //______________________________________________________________________________
      /**
       *  Each process (root process included) sends the contents of its send buffer to the root process.
      *  The root process receives the messages and stores them in rank order.
      *  The outcome is as if each of the n processes in the group (including the root process)
       * \param in_vars any selializable object vector reference to send the message
       * \param incount Number of elements in receive in \p in_vars
       * \param out_vars any selializable object vector reference to receive the message
       * \param outcount Number of elements in receive in \p out_vars
       * \param root id of the main message where message was sent
       * \return TGrequest obj
       */
      template<class Type> void TCommunicator::Gather(const Type *in_vars, Int_t incount, Type *out_vars, Int_t outcount, Int_t root) const
      {
         if (GetRank() == root) {
            //TODO: check special cases to improved this error handling
            if ((GetSize()*incount) % outcount   != 0) {
               Fatal("TCommunicator::Gather", "Number of elements sent can't be fitted in gather message");
               Abort(ERR_COUNT);
            }
            for (auto i = 0 ; i < GetSize(); i++) {
               if (i == root) continue;
               auto stride = incount * i;
               Recv(&out_vars[stride], incount, i, GetInternalTag());
            }
            //NOTE: copy memory with memmove because memcpy() with overlapping areas produces undefined behavior
            //In scatter is not same because out_vars have not overlapping, I mean I just need to fill the entire vector not a region
            auto stride = incount * root;
            memmove((void *)&out_vars[stride], (void *)in_vars, sizeof(Type)*incount);
         } else {
            Send(in_vars, incount, root, GetInternalTag());
         }
      }

      //______________________________________________________________________________
      /**
        *Method to apply reduce operation using binary tree reduction.
        * \param in_var variable to eval in the reduce operation
        * \param out_var variable to receive the variable operation
        * \param opf function the perform operation
        * \param root id of the main process where the result was received
        */
      template<class Type> void TCommunicator::Reduce(const Type &in_var, Type &out_var, Op<Type> (*opf)(), Int_t root) const
      {
         Reduce(&in_var, &out_var, 1, opf, root);
      }

      //______________________________________________________________________________
      /**
       * Method to apply reduce operation over and array of elements using binary tree reduction.
       * \param in_var variable to eval in the reduce operation
       * \param out_var variable to receive the variable operation
       * \param count Number of elements to reduce in \p in_var and \p out_var
       * \param opf function the perform operation
       * \param root id of the main process where the result was received
       */
      template<class Type> void TCommunicator::Reduce(const Type *in_var, Type *out_var, Int_t count, Op<Type> (*opf)(), Int_t root) const
      {
         auto op = opf();

         if (!std::is_class<Type>::value) memmove((void *)out_var, (void *)in_var, sizeof(Type)*count);
         else {
            for (auto i = 0; i < count; i++) {
               TMpiMessage msgi;
               msgi.WriteObject(in_var[i]);
               Char_t *buffer = new Char_t[msgi.BufferSize()]; //this pointer can't be freed, it will be free when the object dies
               memcpy((void *)buffer, (void *)msgi.Buffer(), sizeof(Char_t)*msgi.BufferSize());
               TMpiMessage msgo(buffer, msgi.BufferSize()); //using serialization to copy memory without copy constructor
               auto cl = gROOT->GetClass(typeid(Type));
               auto obj_tmp = msgo.ReadObjectAny(cl);
               memmove((void *)&out_var[i], obj_tmp, sizeof(Type));
            }
         }

         auto size = GetSize();
         auto lastpower = 1 << (Int_t)log2(size);

         for (Int_t i = lastpower; i < size; i++)
            if (GetRank() == i)
               Send(in_var, count, i - lastpower, GetInternalTag());
         for (Int_t i = 0; i < size - lastpower; i++)
            if (GetRank() == i) {
               Type recvbuffer[count];
               Recv(recvbuffer, count, i + lastpower, GetInternalTag());
               for (Int_t j = 0; j < count; j++) out_var[j] = op(in_var[j], recvbuffer[j]);
            }

         for (Int_t d = 0; d < (Int_t)log2(lastpower); d++)
            for (Int_t k = 0; k < lastpower; k += 1 << (d + 1)) {
               auto receiver = k;
               auto sender = k + (1 << d);
               if (GetRank() == receiver) {
                  Type recvbuffer[count];
                  Recv(recvbuffer, count, sender, GetInternalTag());
                  for (Int_t j = 0; j < count; j++) out_var[j]  = op(out_var[j], recvbuffer[j]);
               } else if (GetRank() == sender)
                  Send(out_var, count, receiver, GetInternalTag());
            }
         if (root != 0 && GetRank() == 0) Send(out_var, count, root, GetInternalTag());
         if (root == GetRank() && GetRank() != 0) Recv(out_var, count, 0, GetInternalTag());
      }

      //______________________________________________________________________________
      /**
       * Method to apply reduce operation over and array of elements using binary tree reduction. and the results is send to all processes.
       * \param in_var variable to eval in the reduce operation
       * \param out_var variable to receive the variable operation
       * \param count Number of elements to reduce in \p in_var and \p out_var
       * \param opf function the perform operation
       */
      template<class Type> void TCommunicator::AllReduce(const Type *in_vars, Type *out_vars, Int_t count, Op<Type> (*opf)()) const
      {
         Reduce(in_vars, out_vars, count, opf, GetMainProcess());
         Bcast(out_vars, count, GetMainProcess());
      }
      //______________________________________________________________________________
      /**
       *  Each process (TCommunicator::GetMainProcess() process included) sends the contents of its send buffer to the TCommunicator::GetMainProcess() process.
      *  The TCommunicator::GetMainProcess() process receives the messages and stores them in rank order, after that send a bcast message with the results.
      *  The outcome is as if each of the n processes in the group (including the root process)
       * \param in_vars any selializable object vector reference to send the message
       * \param incount Number of elements in receive in \p in_vars
       * \param out_vars any selializable object vector reference to receive the message
       * \param outcount Number of elements in receive in \p out_vars
       * \return TGrequest obj
       */
      template<class Type> void TCommunicator::AllGather(const Type *in_vars, Int_t incount, Type *out_vars, Int_t outcount) const
      {
         Gather(in_vars, incount, out_vars, outcount, GetMainProcess());
         Bcast(out_vars, outcount, GetMainProcess());
      }
      //______________________________________________________________________________
      template<> void TCommunicator::Serialize<TMpiMessage>(Char_t **buffer, Int_t &size, const TMpiMessage *vars, Int_t count, const TCommunicator *comm, Int_t dest, Int_t source, Int_t tag, Int_t root);

      //______________________________________________________________________________
      template<> void TCommunicator::Unserialize<TMpiMessage>(Char_t *ibuffer, Int_t isize, TMpiMessage *vars, Int_t count, const TCommunicator *comm, Int_t dest, Int_t source, Int_t tag, Int_t root);
   }
}

#endif
