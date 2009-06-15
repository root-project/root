// @(#)root/minuit2:$Id$
// Author: A. Lazzaro 2009
/***************************************************************************
 * Package: Minuit2                                                        *
 *    File: $Id$         *
 *  Author: Alfio Lazzaro, alfio.lazzaro@mi.infn.it                        *
 *                                                                         *
 * Copyright: (C) 2008 by Universita' and INFN, Milan                      *
 ***************************************************************************/

#ifndef MPIPROCESS
#define MPIPROCESS

// disable MPI calls
//#define MPIPROC

#include "Minuit2/MnMatrix.h"

#ifdef MPIPROC
#include "mpi.h"
#endif


namespace ROOT {

   class MPITerminate {
   public:
      ~MPITerminate() { 
#ifdef MPIPROC
         if (MPI::Is_initialized() && !(MPI::Is_finalized())) {
            std::cout << "Info --> MPITerminate:: End MPI on #"
                      << MPI::COMM_WORLD.Get_rank() << " processor"
                      << std::endl;

            MPI::Finalize();
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

      static unsigned int GetMPIGlobalRank() { StartMPI(); return fGlobalRank; }
      static unsigned int GetMPIGlobalSize() { StartMPI(); return fGlobalSize; }
      static inline void StartMPI() {
#ifdef MPIPROC  
         if (!(MPI::Is_initialized())) {    
            MPI::Init();    
            std::cout << "Info --> MPIProcess::StartMPI: Start MPI on #" 
                      << MPI::COMM_WORLD.Get_rank() << " processor"
                      << std::endl;
         }
         fGlobalSize = MPI::COMM_WORLD.Get_size();
         fGlobalRank = MPI::COMM_WORLD.Get_rank();
#endif
      }

      static void TerminateMPI() { 
#ifdef MPIPROC
         if (fCommunicators[0]!=0 && fCommunicators[1]!=0) {
            delete fCommunicators[0]; fCommunicators[0] = 0; fIndecesComm[0] = 0;
            delete fCommunicators[1]; fCommunicators[1] = 0; fIndecesComm[1] = 0;
         }
      
         MPITerminate terminate; 

#endif
      }

      static bool SetCartDimension(unsigned int dimX, unsigned int dimY);
      static bool SetDoFirstMPICall(bool doFirstMPICall = true);

      inline void SumReduce(const double& sub, double& total) {
         total = sub;

#ifdef MPIPROC
         if (fSize>1) {
            fCommunicator->Allreduce(&sub,&total,1,MPI::DOUBLE,MPI::SUM);
         }
#endif
      }

   private:

#ifdef MPIPROC
      void MPISyncVector(double *ivector, int svector, double *ovector);
#endif

   private:
      unsigned int fNelements;
      unsigned int fSize;
      unsigned int fRank;  

      static unsigned int fGlobalSize;
      static unsigned int fGlobalRank;  

      static unsigned int fCartSizeX;
      static unsigned int fCartSizeY;
      static unsigned int fCartDimension;
      static bool fNewCart;

      unsigned int fNumElements4JobIn;
      unsigned int fNumElements4JobOut;

#ifdef MPIPROC
      static MPI::Intracomm *fCommunicator;
      static int fIndexComm; // maximum 2 communicators, so index can be 0 and 1
      static MPI::Intracomm *fCommunicators[2]; // maximum 2 communicators
      static unsigned int fIndecesComm[2];
#endif

   };

} // namespace ROOT

#endif


