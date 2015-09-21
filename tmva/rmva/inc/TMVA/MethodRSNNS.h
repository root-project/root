// @(#)root/tmva/rmva $Id$
// Author: Omar Zapata,Lorenzo Moneta, Sergei Gleyzer 2015

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : RMethodRSNNS                                                          *
 *                                                                                *
 * Description:                                                                   *
 *      R´s Package RSNNS  method based on ROOTR                                  *
 *                                                                                *
 **********************************************************************************/

#ifndef ROOT_TMVA_RMethodRSNNS
#define ROOT_TMVA_RMethodRSNNS

#ifndef ROOT_TMVA_RMethodBase
#include "TMVA/RMethodBase.h"
#endif
/**
 @namespace TMVA
 namespace associated TMVA package for ROOT.
 
 */

namespace TMVA {

   class Factory;  // DSMTEST
   class Reader;   // DSMTEST
   class DataSetManager;  // DSMTEST
   class Types;
   
       /**
        \class MethodRSNNS
         \brief RMVA class for all methods  based on RSNNS
          The Stuttgart Neural Network Simulator (SNNS) is a library containing many standard implementations of neural networks. 
          This package wraps the SNNS functionality to make it available from within R. 
          Using the RSNNS low-level interface, all of the algorithmic functionality and flexibility of SNNS can be accessed. 
          Furthermore, the package contains a convenient high-level interface, so that the most common neural network topologies and learning algorithms integrate seamlessly into R.          
           This method uses the package https://cran.r-project.org/web/packages/RSNNS/index.html
          
          \section BookingRMLP The Booking options for RMLP are
          <table >
          <tr><td colspan="4"><span style="color:#06F; background-color:">Configuration options reference for MVA method: RSNNS/MLP</span></td></tr>
          <tr><td><span style="color:#06F; background-color:">Option</span></td><td><span style="color:#06F; background-color:">Default Value</span></td><td><span style="color:#06F; background-color:">Predefined values</span></td><td><span style="color:#06F; background-color:">Description</span></td></tr>
          <tr><td><strong>Size</strong></td><td> c(5)  </td><td> - </td><td>(R's vector type given in string) with the number of units in the hidden layer(s)   </td></tr>
          <tr><td><strong>Maxit</strong></td><td>  100 </td><td> - </td><td>maximum of iterations to learn </td></tr>
          <tr><td><strong>InitFunc</strong></td><td>Randomize_Weights</td><td>Randomize_Weights<br> ART1_Weights<br>ART2_Weights<br> ARTMAP_Weights<br> CC_Weights<br> ClippHebb<br> CPN_Weights_v3.2<br> CPN_Weights_v3.3<br> CPN_Rand_Pat<br> DLVQ_Weights<br> Hebb<br> Hebb_Fixed_Act<br> JE_Weights<br>  Kohonen_Rand_Pat<br> Kohonen_Weights_v3.2<br> Kohonen_Const<br> PseudoInv<br> Random_Weights_Perc<br> RBF_Weights<br> RBF_Weights_Kohonen<br> RBF_Weights_Redo<br> RM_Random_Weights<br> ENZO_noinit</td><td> the initialization function to use</td></tr>
          <tr><td><strong>InitFuncParams</strong></td><td>c(-0.3, 0.3)</td><td> -</td><td>(R's vector type given in string) the parameters for the initialization function</td></tr>
          <tr><td><strong>LearnFunc</strong> </td><td> Std_Backpropagation </td><td>Std_Backpropagation <br>BackpropBatch<br> BackpropChunk<br> BackpropClassJogChunk<br>  BackpropMomentum<br> BackpropWeightDecay<br> TimeDelayBackprop<br> Quickprop<br> Rprop<br> RpropMAP<br> BPTT<br> CC<br> TACOMA<br>     BBPTT<br> QPTT<br> JE_BP<br> JE_BP_Momentum<br> JE_Quickprop<br> JE_Rprop<br> Monte-Carlo<br> SCG<br> Sim_Ann_SS<br> Sim_Ann_WTA<br> Sim_Ann_WWTA</td><td>  the learning function to use</td></tr>
          <tr><td><strong>LearnFuncParams</strong></td><td>c(0.2, 0)</td><td>-</td><td>(R's vector type given in string) the parameters for the learning function  e.g. ‘Std_Backpropagation’, ‘BackpropBatch’ have two parameters, the learning rate and the maximum output difference. The learning rate is usually a value between 0.1 and 1. It specifies the gradient descent step width. The maximum difference defines, how much difference between output and target value is treated as zero error, and not backpropagated. This parameter is used to prevent overtraining. For a complete list of the parameters of all the learning functions, see the SNNS User Manual, pp. 67.</td></tr>
          <tr><td><strong>UpdateFunc</strong> </td><td> Topological_Order </td><td> Topological_Order<br> ART1_Stable<br> ART1_Synchronous<br> ART2_Stable<br> ART2_Synchronous<br> ARTMAP_Stable<br> ARTMAP_Synchronous<br> Auto_Synchronous<br> BAM_Order<br> BPTT_Order<br> CC_Order<br>   CounterPropagation<br> Dynamic_LVQ<br> Hopfield_Fixed_Act<br> Hopfield_Synchronous<br> JE_Order<br> JE_Special<br> Kohonen_Order<br> Random_Order<br> Random_Permutation<br> Serial_Order<br> Synchonous_Order<br> TimeDelay_Order<br>ENZO_prop</td><td>   the update function to use</td></tr>
          <tr><td><strong>UpdateFuncParams</strong></td><td>c(0)</td><td>-</td><td>the parameters for the update function</td></tr>
          <tr><td><strong>HiddenActFunc</strong></td><td>Act_Logistic</td><td>Act_Logistic<br> Act_Elliott<br> Act_BSB<br> Act_TanH<br> Act_TanHPlusBias<br> Act_TanH_Xdiv2<br> Act_Perceptron<br> Act_Signum<br> Act_Signum0<br> Act_Softmax<br> Act_StepFunc<br> Act_HystStep<br> Act_BAM<br> Logistic_notInhibit<br> Act_MinOutPlusWeight<br> Act_Identity<br> Act_IdentityPlusBias<br> Act_LogisticTbl<br> Act_RBF_Gaussian<br> Act_RBF_MultiQuadratic<br> Act_RBF_ThinPlateSpline<br> Act_less_than_0<br> Act_at_most_0<br> Act_at_least_1<br> Act_at_least_2<br> Act_exactly_1<br> Act_Product<br> Act_ART1_NC<br> Act_ART2_Identity<br> Act_ART2_NormP<br> Act_ART2_NormV<br> Act_ART2_NormW<br> Act_ART2_NormIP<br> Act_ART2_Rec<br> Act_ART2_Rst<br> Act_ARTMAP_NCa<br> Act_ARTMAP_NCb<br> Act_ARTMAP_DRho<br> Act_LogSym<br> Act_TD_Logistic<br> Act_TD_Elliott<br> Act_Euclid<br> Act_Component<br> Act_RM<br> Act_TACOMA<br> Act_CC_Thresh<br> Act_Sinus<br> Act_Exponential </td><td>  the activation function of all hidden units</td></tr>
          <tr><td><strong>ShufflePatterns</strong></td><td>kTRUE</td><td>-</td><td>should the patterns be shuffled?</td></tr>
          <tr><td><strong>LinOut</strong></td><td>kFALSE</td><td>-</td><td>sets the activation function of the output units to linear or logistic</td></tr>
          <tr><td><strong>PruneFunc</strong></td><td>NULL</td><td>MagPruning<br>OptimalBrainDamage<br>OptimalBrainSurgeon<br>Skeletonization<br>Noncontributing_Units<br>None<br>Binary<br>Inverse<br>Clip<br>LinearScale<br>Norm<br>Threshold</td><td>the pruning function to use</td></tr>
          <tr><td><strong>PruneFuncParams</strong></td><td>NULL</td><td>-</td><td>the parameters for the pruning function. </td></tr>
          </table>         
         \link http://oproject.org/RMVA
         \ingroup TMVA
       */
   
   class MethodRSNNS : public RMethodBase {

   public :

         /**
         Default constructor that inherits from TMVA::RMethodBase and it have a ROOT::R::TRInterface instance for internal use.
         \param jobName Name taken from method type
         \param methodType Associate TMVA::Types::EMVA (available MVA methods)
         \param methodTitle Sub method associate to method type.
         \param dsi TMVA::DataSetInfo object
         \param theOption Booking options for method
         \param theBaseDir object to TDirectory with the path to calculate histograms and results for current method.
         */
      MethodRSNNS(const TString &jobName,
                  const TString &methodTitle,
                  DataSetInfo &theData,
                  const TString &theOption = "",
                  TDirectory *theTargetDir = NULL);
         /**
         Constructor used for Testing + Application of the MVA, only (no training), using given weight file. 
         inherits from TMVA::MethodBase and it have a ROOT::R::TRInterface instance for internal use.
         \param methodType Associate TMVA::Types::EMVA (available MVA methods)
         \param dsi TMVA::DataSetInfo object
         \param theBaseDir object to TDirectory with the path to calculate histograms and results for current method.
         */
      MethodRSNNS(DataSetInfo &dsi,
                  const TString &theWeightFile,
                  TDirectory *theTargetDir = NULL);


      ~MethodRSNNS(void);
         /**
         Pure abstract method to build the train system 
         and to save the fModel object in a .RData file for model persistence
         */

      void     Train();
         /**
         Pure abstract method for options treatment(some default options initialization)
         and package options initialization
         */
      void     Init();
         /**
         Pure abstract method to declare booking options associate to multivariate algorithm.
        \see BookingRMLP
         */      
      void     DeclareOptions();
         /**
         Pure abstract method to parse booking options associate to multivariate algorithm.
         */      
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
      UInt_t fMvaCounter;
      std::vector<Float_t> fProbResultForTrainSig;
      std::vector<Float_t> fProbResultForTestSig;

      TString fNetType;//default RMPL
      //RSNNS Options for all NN methods
      TString  fSize;//number of units in the hidden layer(s)
      UInt_t fMaxit;//maximum of iterations to learn

      TString fInitFunc;//the initialization function to use
      TString fInitFuncParams;//the parameters for the initialization function (type 6 see getSnnsRFunctionTable() in RSNNS package)

      TString fLearnFunc;//the learning function to use
      TString fLearnFuncParams;//the parameters for the learning function

      TString fUpdateFunc;//the update function to use
      TString fUpdateFuncParams;//the parameters for the update function

      TString fHiddenActFunc;//the activation function of all hidden units
      Bool_t fShufflePatterns;//should the patterns be shuffled?
      Bool_t fLinOut;//sets the activation function of the output units to linear or logistic

      TString fPruneFunc;//the pruning function to use
      TString fPruneFuncParams;//the parameters for the pruning function. Unlike the
      //other functions, these have to be given in a named list. See
      //the pruning demos for further explanation.
      std::vector<UInt_t>  fFactorNumeric;   //factors creations
      //RSNNS mlp require a numeric factor then background=0 signal=1 from fFactorTrain
      static Bool_t IsModuleLoaded;
      ROOT::R::TRFunctionImport predict;
      ROOT::R::TRFunctionImport mlp;
      ROOT::R::TRFunctionImport asfactor;
      ROOT::R::TRObject         *fModel;
      // get help message text
      void GetHelpMessage() const;

      ClassDef(MethodRSNNS, 0)
   };
} // namespace TMVA
#endif
