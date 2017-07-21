// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TCartCommunicator
#define ROOT_Mpi_TCartCommunicator

#include <Mpi/TIntraCommunicator.h>

namespace ROOT {

namespace Mpi {
/**
 * \class TCartCommunicator
 * TODO:
 \see TGroup TIntraCommunicator TCommunicator TInterCommunicator
 \ingroup Mpi
*/

class TCartCommunicator : public TIntraCommunicator {
public:
   TCartCommunicator();

   TCartCommunicator(const TCartCommunicator &data);

   TCartCommunicator(const MPI_Comm &comm);

   TCartCommunicator &operator=(const TCartCommunicator &comm)
   {
      fComm = comm.fComm;
      return *this;
   }

   TCartCommunicator &operator=(const MPI_Comm &comm)
   {
      fComm = comm;
      return *this;
   }

   virtual TCartCommunicator *Dup() const;

   inline operator MPI_Comm() const { return fComm; }

   virtual Int_t GetDim() const;

   virtual void GetTopo(Int_t maxdims, Int_t dims[], Bool_t periods[], Int_t coords[]) const;

   virtual Int_t GetCartRank(const Int_t coords[]) const;

   virtual Int_t GetCartRank(const std::vector<Int_t> coords) const;

   virtual void GetCoords(Int_t rank, Int_t maxdims, Int_t coords[]) const;

   virtual void GetCoords(Int_t rank, Int_t maxdims, std::vector<Int_t> &coords) const;

   virtual void Shift(Int_t direction, Int_t disp, Int_t &rank_source, Int_t &rank_dest) const;

   virtual TCartCommunicator Sub(const Bool_t remain_dims[]) const;

   virtual Int_t Map(Int_t ndims, const Int_t dims[], const Bool_t periods[]) const;

   ClassDef(TCartCommunicator, 4)
};
}
}

#endif
