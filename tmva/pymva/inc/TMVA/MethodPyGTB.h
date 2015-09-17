// @(#)root/tmva/pymva $Id$
// Authors: Omar Zapata, Lorenzo Moneta, Sergei Gleyzer 2015

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodPyGTB                                                           *
 * Web    : http://oproject.org                                                   *
 *                                                                                *
 * Description:                                                                   *
 *      scikit-learn Package GradientBoostingClassifier  method based on python   *
 *                                                                                *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodPyGTB
#define ROOT_TMVA_MethodPyGTB

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodPyGTB                                                     //
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
   class MethodPyGTB : public PyMethodBase {

   public :

      // constructors
      MethodPyGTB(const TString &jobName,
                  const TString &methodTitle,
                  DataSetInfo &theData,
                  const TString &theOption = "",
                  TDirectory *theTargetDir = NULL);

      MethodPyGTB(DataSetInfo &dsi,
                  const TString &theWeightFile,
                  TDirectory *theTargetDir = NULL);


      ~MethodPyGTB(void);
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
      virtual void AddWeightsXMLTo(void * /* parent */ ) const {}        // = 0;
      virtual void ReadWeightsFromXML(void * /*wghtnode*/) {}    // = 0;
      virtual void ReadWeightsFromStream(std::istream &) {} //= 0;       // backward compatibility
      void ReadStateFromFile();
   private :
      DataSetManager    *fDataSetManager;     // DSMTEST
      friend class Factory;                   // DSMTEST
      friend class Reader;                    // DSMTEST
   protected:
      //GTB options
      TString loss;// {'deviance', 'exponential'}, optional (default='deviance')
      //loss function to be optimized. 'deviance' refers to
      //deviance (= logistic regression) for classification
      //with probabilistic outputs. For loss 'exponential' gradient
      //boosting recovers the AdaBoost algorithm.
      Double_t learning_rate;//float, optional (default=0.1)
      //learning rate shrinks the contribution of each tree by `learning_rate`.
      //There is a trade-off between learning_rate and n_estimators.

      Int_t n_estimators;//integer, optional (default=10)
      //The number of trees in the forest.
      Double_t subsample;//float, optional (default=1.0)
      //The fraction of samples to be used for fitting the individual base
      //learners. If smaller than 1.0 this results in Stochastic Gradient
      //Boosting. `subsample` interacts with the parameter `n_estimators`.
      //Choosing `subsample < 1.0` leads to a reduction of variance
      //and an increase in bias.
      Int_t min_samples_split;// integer, optional (default=2)
      //The minimum number of samples required to split an internal node.
      Int_t min_samples_leaf;//integer, optional (default=1)
      //The minimum number of samples required to be at a leaf node.
      Double_t min_weight_fraction_leaf;//float, optional (default=0.)
      //The minimum weighted fraction of the input samples required to be at a leaf node.
      Int_t max_depth;//integer, optional (default=3)
      //maximum depth of the individual regression estimators. The maximum
      //depth limits the number of nodes in the tree. Tune this parameter
      //for best performance; the best value depends on the interaction
      //of the input variables.
      //Ignored if ``max_leaf_nodes`` is not None.

      TString init;//BaseEstimator, None, optional (default=None)
      //An estimator object that is used to compute the initial
      //predictions. ``init`` has to provide ``fit`` and ``predict``.
      //If None it uses ``loss.init_estimator``.
      TString random_state;//int, RandomState instance or None, optional (default=None)
      //If int, random_state is the seed used by the random number generator;
      //If RandomState instance, random_state is the random number generator;
      //If None, the random number generator is the RandomState instance used
      //by `np.random`.
      TString max_features;//int, float, string or None, optional (default="auto")
      //The number of features to consider when looking for the best split:
      //- If int, then consider `max_features` features at each split.
      //- If float, then `max_features` is a percentage and
      //`int(max_features * n_features)` features are considered at each split.
      //- If "auto", then `max_features=sqrt(n_features)`.
      //- If "sqrt", then `max_features=sqrt(n_features)`.
      //- If "log2", then `max_features=log2(n_features)`.
      //- If None, then `max_features=n_features`.
      //     Note: the search for a split does not stop until at least one
      //     valid partition of the node samples is found, even if it requires to
      //     effectively inspect more than ``max_features`` features.
      //     Note: this parameter is tree-specific.
      Int_t verbose;//Controls the verbosity of the tree building process.
      TString max_leaf_nodes;//int or None, optional (default=None)
      //Grow trees with ``max_leaf_nodes`` in best-first fashion.
      //Best nodes are defined as relative reduction in impurity.
      //If None then unlimited number of leaf nodes.
      //If not None then ``max_depth`` will be ignored.

      Bool_t warm_start;//bool, optional (default=False)
      //When set to ``True``, reuse the solution of the previous call to fit
      //and add more estimators to the ensemble, otherwise, just fit a whole
      //new forest.
      // get help message text
      void GetHelpMessage() const;


      ClassDef(MethodPyGTB, 0)
   };
} // namespace TMVA
#endif
