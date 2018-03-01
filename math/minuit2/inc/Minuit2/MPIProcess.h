// @(#)root/minuit2:$Id$
// Author: A. Lazzaro 2009
/***************************************************************************
 * Package: Minuit2                                                        *
 *    File: $Id$         *
 *  Author: Alfio Lazzaro, alfio.lazzaro@mi.infn.it                        *
 *                                                                         *
 * Copyright: (C) 2008 by Universita' and INFN, Milan                      *
 ***************************************************************************/

#ifndef ROOT_Minuit2_MPIProcess
#define ROOT_Minuit2_MPIProcess

// disable MPI calls
//#define MPIPROC

#include "Minuit2/MnMatrix.h"

#ifdef MPIPROC
#include "mpi.h"
#include <iostream>
#endif


namespace ROOT {

namespace Minuit2 {

   class MPITerminate {
   public:
      ~MPITerminate() {
#ifdef MPIPROC
         int initialized = 0;
         MPI_Initialized(&initialized);
         if (initialized) {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) {
               int rank = 0;
               MPI_Comm_rank(MPI_COMM_WORLD, &rank);
               std::cout << "Info --> MPITerminate:: End MPI on #"
                         << rank << " processor"
                         << std::endl;

               MPI_Finalize();
            }
         }
#endif
      }

   };


   class MPIProcess {
   public:
      MPIProcess(unsigned int nelements, unsigned int indexComm);
      ~MPIProcess();

      inline unsigned int NumElements4JobIn() const { return fNumElements4JobIn; }
      inline unsigned int NumElements4JobOut() const { return fNumElements4JobOut; }

      inline unsigned int NumElements4Job(unsigned int rank) const
      { return NumElements4JobIn()+((rank<NumElements4JobOut()) ? 1 : 0); }

      inline unsigned int StartElementIndex() const
      { return ((fRank<NumElements4JobOut()) ? (fRank*NumElements4Job(fRank)) :
                (fNelements-(fSize-fRank)*NumElements4Job(fRank))); }

      inline unsigned int EndElementIndex() const
      { return StartElementIndex()+NumElements4Job(fRank); }

      inline unsigned int GetMPISize() const { return fSize; }
      inline unsigned int GetMPIRank() const { return fRank; }

      bool SyncVector(ROOT::Minuit2::MnAlgebraicVector &mnvector);
      bool SyncSymMatrixOffDiagonal(ROOT::Minuit2::MnAlgebraicSymMatrix &mnmatrix);

      static unsigned int GetMPIGlobalRank() { StartMPI(); return fgGlobalRank; }
      static unsigned int GetMPIGlobalSize() { StartMPI(); return fgGlobalSize; }
      static inline void StartMPI() {
#ifdef MPIPROC
         int initialized = 0;
         MPI_Initialized(&initialized);
         if (!(initialized)) {
            int argc = 0;
            char** argv = NULL;
            MPI_Init(&argc, &argv);
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            std::cout << "Info --> MPIProcess::StartMPI: Start MPI on #"
                      << rank << " processor"
                      << std::endl;
         }
         MPI_Comm_size(MPI_COMM_WORLD, &fgGlobalSize);
         MPI_Comm_rank(MPI_COMM_WORLD, &fgGlobalRank);
#endif
      }

      static void TerminateMPI() {
#ifdef MPIPROC

         MPITerminate();

#endif
      }

      static bool SetCartDimension(unsigned int dimX, unsigned int dimY);
      static bool SetDoFirstMPICall(bool doFirstMPICall = true);

      inline void SumReduce(const double& sub, double& total) {
         total = sub;

#ifdef MPIPROC
         if (fSize>1) {
            MPI_Allreduce(&sub, &total, 1, MPI_DOUBLE, MPI_SUM, *fgCommunicator);
         }
#endif
      }

   private:

#ifdef MPIPROC
      void MPISyncVector(double *ivector, int svector, double *ovector);
#endif

   private:
      unsigned int fNelements;
      int fSize;
      int fRank;

      static int fgGlobalSize;
      static int fgGlobalRank;

      static unsigned int fgCartSizeX;
      static unsigned int fgCartSizeY;
      static unsigned int fgCartDimension;
      static bool fgNewCart;

      unsigned int fNumElements4JobIn;
      unsigned int fNumElements4JobOut;

#ifdef MPIPROC
      static MPI_Comm fgCommunicatorWorld; //Need to be able to point to the world communicator
      static MPI_Comm *fgCommunicator;
      static int fgIndexComm; // maximum 2 communicators, so index can be 0 and 1
      static MPI_Comm fgCommunicators[2]; // maximum 2 communicators
      static unsigned int fgIndecesComm[2];
#endif

   };

} // namespace Minuit2
} // namespace ROOT

#endif


