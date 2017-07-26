// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TGraphCommunicator
#define ROOT_Mpi_TGraphCommunicator

#include <Mpi/TIntraCommunicator.h>

namespace ROOT {

namespace Mpi {
/**
 * \class TGraphCommunicator
 * Class to map ranks(processes) into graph structure, according edges and nodes.
 *
 * \see TGroup TIntraCommunicator TCommunicator TInterCommunicator TCartCommunicator
 * \ingroup Mpi
*/

class TGraphCommunicator : public TIntraCommunicator {
public:
   TGraphCommunicator();

   TGraphCommunicator(const TGraphCommunicator &data);

   TGraphCommunicator(const MPI_Comm &comm);

   TGraphCommunicator &operator=(const TGraphCommunicator &comm)
   {
      fComm = comm.fComm;
      return *this;
   }

   TGraphCommunicator &operator=(const MPI_Comm &comm)
   {
      fComm = comm;
      return *this;
   }

   virtual TGraphCommunicator *Dup() const;

   inline operator MPI_Comm() const { return fComm; }

   ClassDef(TGraphCommunicator, 4)
};
}
}

#endif
