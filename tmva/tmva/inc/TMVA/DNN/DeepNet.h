// @(#)root/tmva/tmva/cnn:$Id$
// Author: Vladimir Ilievski 24/06/17

/*************************************************************************
 * Copyright (C) 2017, Vladimir Ilievski                                 *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef TMVA_DNN_DEEPNET
#define TMVA_DNN_DEEPNET

namespace TMVA
{
namespace DNN
{

template<typename Architecture_t>
   class VDeepNet
{
public:
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;

private:
   size_t fBatchSize;                  ///< Batch size used for training and evaluation.
   size_t fInputDepth;                 ///< The depth of the input.
   size_t fInputHeight;                ///< The height of the input.
   size_t fInputWidth;                 ///< The width of the input.
    
    
   ELossFunction fJ;                   ///< The loss function of the network.
   ERegularization fR;                 ///< The regularization used for the network.
   Scalar_t fWeightDecay;              ///< The weight decay factor.
    
public:
    
   /*! Default Constructor */
   VDeepNet();

   /*! Constructor */
   VDeepNet(size_t BatchSize,
            size_t InputDepth,
            size_t InputHeight,
            size_t InputWidth,
            ELossFunction fJ,
            ERegularization fR = ERegularization::kNone,
            Scalar_t fWeightDecay = 0.0);
    
   /*! Copy-constructor */
   VDeepNet(const VDeepNet &);
    
   /*! Virtual destructor */
   virtual ~VDeepNet();
    
   /*! Function for initialization of the CNN. */
   virtual void Initialize(EInitialization m) = 0;
    
   /*! Initialize the gradients of the CNN to zero. Required if the CNN
    *  is optimized by the momentum-based techniques. */
   virtual void InitializeGradients() = 0;
    
   /*! Function that executes the entire forward pass in the network. */
   virtual void Forward(std::vector<Matrix_t> input,
                        bool applyDropout = false) = 0;
    
   /*! Function that executes the entire backward pass in the network. */
   virtual void Backward(std::vector<Matrix_t> input,
                         const Matrix_t &groundTruth) = 0;
    
   /*! Function for evaluating the loss, based on the activations stored
    *  in the last Fully Connected Layer. */
   virtual Scalar_t Loss(const Matrix_t &groundTruth,
                         bool includeRegularization = true) const = 0;
    
   /*! Function for evaluating the loss, based on the propagation of the given input. */
   virtual Scalar_t Loss(std::vector<Matrix_t> input,
                         const Matrix_t &groundTruth,
                         bool applyDropout = false) = 0;
    
   /*! Prediction for the given inputs, based on what network learned. */
   virtual void Prediction(Matrix_t &predictions,
                           std::vector<Matrix_t> input,
                           EOutputFunction f) = 0;
    
   /*! Prediction based on activations stored in the last Fully Connected Layer. */
   virtual void Prediction(Matrix_t &predictions,
                           EOutputFunction f) const = 0;
    
   virtual void Print() = 0;
    
    
   /*! Getters */
   inline size_t GetBatchSize()   const {return fBatchSize;}
   inline size_t GetInputDepth()  const {return fInputDepth;}
   inline size_t GetInputHeight() const {return fInputHeight;}
   inline size_t GetInputWidth()  const {return fInputWidth;}
    
   inline ELossFunction GetLossFunction()      const {return fJ;}
   inline ERegularization GetRegularization()  const {return fR;}
   inline Scalar_t GetWeightDecay()            const {return fWeightDecay;}
    
    
   /*! Setters */
   inline void SetBatchSize(size_t batchSize) {
      fBatchSize = batchSize;
   }
    
   inline void SetInputDepth(size_t inputDepth) {
      fInputDepth = inputDepth;
   }
    
   inline void SetInputHeight(size_t inputHeight) {
      fInputHeight = inputHeight;
    }
    
   inline void SetInputWidth(size_t inputWidth) {
      fInputWidth = inputWidth;
   }
    
   inline void SetLossFunction(ELossFunction J) {
      fJ = J;
   }
    
   inline void SetRegularization(ERegularization R) {
      fR = R;
   }
    
   inline void SetWeightDecay(Scalar_t weightDecay) {
      fWeightDecay = weightDecay;
   }
};

//______________________________________________________________________________
template<typename Architecture_t>
   VDeepNet<Architecture_t>::VDeepNet()
   : fBatchSize(0), fInputDepth(0), fInputHeight(0), fInputWidth(0),
     fJ(ELossFunction::kMeanSquaredError), fR(ERegularization::kNone),
     fWeightDecay(0.0)
{
   // Nothing to do here.
}
    
//______________________________________________________________________________
template<typename Architecture_t>
   VDeepNet<Architecture_t>::VDeepNet(size_t batchSize,
                                      size_t inputDepth,
                                      size_t inputHeight,
                                      size_t inputWidth,
                                      ELossFunction J,
                                      ERegularization R,
                                      Scalar_t weightDecay)
   : fBatchSize(batchSize), fInputDepth(inputDepth), fInputHeight(inputHeight),
     fInputWidth(inputWidth), fJ(J), fR(R), fWeightDecay(weightDecay)
{
   // Nothing to do here.
}


//______________________________________________________________________________
template<typename Architecture_t>
   VDeepNet<Architecture_t>::VDeepNet(const VDeepNet &deepNet)
   : fBatchSize(deepNet.fBatchSize), fInputDepth(deepNet.fInputDepth),
     fInputHeight(deepNet.fInputHeight), fInputWidth(deepNet.fInputWidth),
     fJ(deepNet.fJ), fR(deepNet.fR), fWeightDecay(deepNet.fWeightDecay)
{
   // Nothing to be done here
}

//______________________________________________________________________________
template<typename Architecture_t>
   VDeepNet<Architecture_t>::~VDeepNet()
{
   // Nothing to be done here
}

    
} // namespace DNN
} // namespace TMVA


#endif
