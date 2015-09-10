// @(#)root/tmva/pymva $Id$
// Authors: Omar Zapata, Lorenzo Moneta, Sergei Gleyzer 2015

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodPyAdaBoost                                                      *
 * Web    : http://oproject.org                                                   *
 *                                                                                *
 * Description:                                                                   *
 *      scikit-learn Package AdaBoostClassifier      method based on python       *
 *                                                                                *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodPyAdaBoost
#define ROOT_TMVA_MethodPyAdaBoost

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodPyAdaBoost                                                     //
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
   class MethodPyAdaBoost : public PyMethodBase {

   public :

      // constructors
      MethodPyAdaBoost(const TString &jobName,
                       const TString &methodTitle,
                       DataSetInfo &theData,
                       const TString &theOption = "",
                       TDirectory *theTargetDir = NULL);

      MethodPyAdaBoost(DataSetInfo &dsi,
                       const TString &theWeightFile,
                       TDirectory *theTargetDir = NULL);


      ~MethodPyAdaBoost(void);
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
      virtual void AddWeightsXMLTo(void *parent) const {}        // = 0;
      virtual void ReadWeightsFromXML(void *wghtnode) {}    // = 0;
      virtual void ReadWeightsFromStream(std::istream &) {} //= 0;       // backward compatibility
      void ReadStateFromFile();
   private :
      DataSetManager    *fDataSetManager;     // DSMTEST
      friend class Factory;                   // DSMTEST
      friend class Reader;                    // DSMTEST
   protected:
      //AdaBoost options
      TString base_estimator;//object, optional (default=DecisionTreeClassifier)
      //The base estimator from which the boosted ensemble is built.
      //Support for sample weighting is required, as well as proper `classes_`
      //and `n_classes_` attributes.
      Int_t n_estimators;//integer, optional (default=10)
      //The number of trees in the forest.
      Double_t learning_rate;//loat, optional (default=1.)
      //Learning rate shrinks the contribution of each classifier by
      //``learning_rate``. There is a trade-off between ``learning_rate`` and ``n_estimators``.
      TString algorithm;//{'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
      //If 'SAMME.R' then use the SAMME.R real boosting algorithm.
      //``base_estimator`` must support calculation of class probabilities.
      //If 'SAMME' then use the SAMME discrete boosting algorithm.
      //The SAMME.R algorithm typically converges faster than SAMME,
      //achieving a lower test error with fewer boosting iterations.
      TString random_state;//int, RandomState instance or None, optional (default=None)
      //If int, random_state is the seed used by the random number generator;
      //If RandomState instance, random_state is the random number generator;
      //If None, the random number generator is the RandomState instance used by `np.random`.
      // get help message text
      void GetHelpMessage() const;


      ClassDef(MethodPyAdaBoost, 0)
   };
} // namespace TMVA
#endif
