// @(#)root/tmva/pymva $Id$
// Authors: Omar Zapata, Lorenzo Moneta, Sergei Gleyzer 2015, Satyarth Praveem

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodPyKMeans                                                  *
 * Web    : http://oproject.org                                                   *
 *                                                                                *
 * Description:                                                                   *
 *      scikit-learn Package KMeansClassifier  method based on python       *
 *                                                                                *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodPyKMeans
#define ROOT_TMVA_MethodPyKMeans

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodPyKMeans                                                       //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_PyMethodBase
#include "TMVA/PyMethodBase.h"
#endif

namespace TMVA {

   class Factory;  // DSMTEST
   class Reader;   // DSMTEST
   class DataSetManager;  // DSMTEST
   class Types;
   class MethodPyKMeans : public PyMethodBase {

   public :

      // constructors
      MethodPyKMeans(const TString &jobName,
                           const TString &methodTitle,
                           DataSetInfo &theData,
                           const TString &theOption = "");

      MethodPyKMeans(DataSetInfo &dsi,
                           const TString &theWeightFile);


      ~MethodPyKMeans(void);
      void     Train();
      // options treatment
      void     Init();
      void     DeclareOptions();
      void     ProcessOptions();
      // create ranking
      const Ranking *CreateRanking()
      {
         return NULL;  // = 0;
      }


      Bool_t HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets);

      // performs classifier testing
      virtual void     TestClassification();


      Double_t GetMvaValue(Double_t *errLower = 0, Double_t *errUpper = 0);

      using MethodBase::ReadWeightsFromStream;
      // the actual "weights"
      virtual void AddWeightsXMLTo(void * /* parent */) const {}        // = 0;
      virtual void ReadWeightsFromXML(void * /* wghtnode */) {}    // = 0;
      virtual void ReadWeightsFromStream(std::istream &) {} //= 0;       // backward compatibility

      void ReadModelFromFile();

   private :
      DataSetManager    *fDataSetManager;     // DSMTEST
      friend class Factory;                   // DSMTEST
      friend class Reader;                    // DSMTEST
   protected:

      Int_t n_clusters; // int, optional, default: 8
      // The number of clusters to form as well as the number of centroids to generate.
      Int_t max_iter; // int, default: 300
      // Maximum number of iterations of the k-means algorithm for a single run.
      Int_t n_init; // int, default: 10
      // Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
      TString init; // {‘k-means++’, ‘random’ or an ndarray}
      // Method for initialization, defaults to ‘k-means++’:
      //     ‘k-means++’ : selects initial cluster centers for k-mean 
      //     clustering in a smart way to speed up convergence.
      //     See section Notes in k_init for more details.
      //     ‘random’: choose k observations (rows) at random
      //     from data for the initial centroids.
      //     If an ndarray is passed, it should be of shape
      //     (n_clusters, n_features) and gives the initial centers.
      TString algorithm; // “auto”, “full” or “elkan”, default=”auto”
      // K-means algorithm to use. The classical EM-style algorithm is “full”.
      // The “elkan” variation is more efficient by using the triangle inequality,
      // but currently doesn’t support sparse data.
      // “auto” chooses “elkan” for dense data and “full” for sparse data.
      TString precompute_distances; // {‘auto’, True, False}
      // Precompute distances (faster but takes more memory).
      // ‘auto’ : do not precompute distances if n_samples * n_clusters > 12 million.
      //     This corresponds to about 100MB overhead per job using double precision.
      // True : always precompute distances
      // False : never precompute distances
      Double_t tol; // float, default: 1e-4
      // Relative tolerance with regards to inertia to declare convergence
      Int_t n_jobs; // int
      // The number of jobs to use for the computation.
      // This works by computing each of the n_init runs in parallel.
      // If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all,
      // which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
      // Thus for n_jobs = -2, all CPUs but one are used.
      TString random_state; // integer or numpy.RandomState, optional
      // The generator used to initialize the centers.
      // If an integer is given, it fixes the seed.
      // Defaults to the global numpy random number generator.
      Int_t verbose; // int, default 0
      // Verbosity mode.
      Bool_t copy_x; // boolean, default True
      // When pre-computing distances it is more numerically accurate
      // to center the data first. If copy_x is True, then
      // the original data is not modified.
      // If False, the original data is modified, and put back before
      // the function returns, but small numerical differences may be
      // introduced by subtracting and then adding the data mean.

      // get help message text
      void GetHelpMessage() const;


      ClassDef(MethodPyKMeans, 0)
   };
} // namespace TMVA
#endif
