// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh 21/06/16

/*************************************************************************
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TMVA_DNN_MINIMIZERS
#define TMVA_DNN_MINIMIZERS

#include "DataLoader.h"
#include <chrono>

namespace TMVA {
namespace DNN {

//______________________________________________________________________________
//
// Generic Gradient Descent Class
//______________________________________________________________________________
//

/** \class TGradientDescent

    Generic implementation of gradient descent minimization.

    The TGradientDescent class implements an architecture and input data
    independent implementation of the gradient descent minimization algorithm.

    The Train(...) method trains a given neural network using the provided training
    and test (validation) data. The interface between input data and the matrix
    representation required by the net is given by the prepare_batches(...) and
    prepare_test_data(...) methods as well as the TBatch class of the
    architecture-specific back end.

    The prepare_batches(...) is expected to generate an iterable of batches. On
    each of these batches, the Step(...) routine of the minimizer is called, which
    for the gradient descent method just adds the corresponding gradients scaled
    by \f$-\alpha\f$ to the weights and biases of each layer. Here \f$\alpha\f$ is
    the learning rate of the gradient descent method.

    The prepare_test_data(...) routine should return a batch representing
    the test data, which is used to evaluate the performance of the net
    every testInterval steps.

    \tparam Architecture_t Type representing which implementation of the low-level
    interface to use.
 */
template<typename Architecture_t>
class TGradientDescent
{
public:
    using Scalar_t = typename Architecture_t::Scalar_t;
    using Matrix_t = typename Architecture_t::Matrix_t;

private:
    size_t   fBatchSize; ///< Batch size to use for the training.
    size_t   fStepCount; ///< Number of steps performed in the current training sessiong.
    size_t   fConvergenceSteps; ///< Number of training epochs without considerable decrease in the test error for convergence.
    size_t   fConvergenceCount; ///< Current number of training epochs without considerable decrease in the test error.
    size_t   fTestInterval; ///< Interval for the computation of the test error.
    Scalar_t fTrainingError;///< Holds the most recently computed training loss.
    Scalar_t fTestError;    ///< Holds the most recently computed test loss.
    Scalar_t fLearningRate; ///< Learning rate \f$\alpha\f$
    Scalar_t fMinimumError; ///< The minimum loss achieved on the training set during the current traning session.

public:
    TGradientDescent();
    TGradientDescent(Scalar_t learningRate,
                     size_t convergenceSteps,
                     size_t testInterval);
    /*! Reset minimizer object to initial state. Does nothing for this minimizer. */
    void Reset() {};
    /*! Train the given net using the given training input data (events), training
      output data (labels), test input data (events), test output data (labels). */
    template <typename Data_t, typename Net_t>
        Scalar_t Train(const Data_t & TrainingDataIn, size_t nTrainingSamples,
                       const Data_t & TestDataIn, size_t nTestSamples,
                       Net_t & net, size_t nThreads = 1);
    /*! Perform a single optimization step on a given batch. Propagates the input
      matrix foward through the net, evaluates the loss and propagates the gradients
      backward through the net. The computed gradients are scaled by the learning
      rate \f$\alpha\f$ and subtracted from the weights and bias values of each
      layer. */
    template <typename Net_t>
    void Step(Net_t &net, Matrix_t &input, const Matrix_t &output);
    template <typename Net_t>
    void Step(Net_t &master, Net_t &net, Matrix_t &input, const Matrix_t &output);
    /** Does not evaluate the loss and therefore not trigger a possible synchronization
     *  with the device. Trains the weights of each layer, but only the bias terms of
     *  the first layer for compatibility with the previous implementation. */
    template <typename Net_t>
    void StepReducedWeights(Net_t &net,
                            Matrix_t &input,
                            const Matrix_t &output);
    /** Similar to StepReducedWeights(...) but also evaluates the loss. May trigger
     * synchronization with the device. */
    template <typename Net_t>
    Scalar_t StepReducedWeightsLoss(Net_t &net,
                                    Matrix_t &input,
                                    const Matrix_t &output);
    template <typename Net_t>
    inline void TestError(Net_t &net,
                          Matrix_t &input,
                          const Matrix_t &output);
    bool HasConverged();

    size_t   GetConvergenceCount() const {return fConvergenceCount;}
    size_t   getConvergenceSteps() const {return fConvergenceSteps;}
    Scalar_t GetTrainingError() const {return fTrainingError;}
    Scalar_t GetTestError() const     {return fTestError;}
    size_t   GetTestInterval() const  {return fTestInterval;}

    void SetConvergenceSteps(size_t steps) {fConvergenceSteps = steps;}
    void SetTestInterval(size_t interval)  {fTestInterval = interval;}
    void SetLearningRate(Scalar_t rate)    {fLearningRate = rate;}
};

//______________________________________________________________________________
//
// Implementation
//______________________________________________________________________________
template<typename Architecture_t>
    TGradientDescent<Architecture_t>::TGradientDescent()
   : fBatchSize(0), fStepCount(0), fConvergenceSteps(0),
     fConvergenceCount(0), fTestInterval(0), fLearningRate(0),
     fMinimumError(1e100)
{
   // Nothing to do here.
}
//______________________________________________________________________________
template<typename Architecture_t>
TGradientDescent<Architecture_t>::TGradientDescent(Scalar_t learningRate,
                                                   size_t convergenceSteps,
                                                   size_t testInterval)
   : fBatchSize(0), fStepCount(0), fConvergenceSteps(convergenceSteps),
     fConvergenceCount(0), fTestInterval(testInterval), fLearningRate(learningRate),
     fMinimumError(1e100)
{
   // Nothing to do here.
}

//______________________________________________________________________________
template<typename Architecture_t>
template <typename Data_t, typename Net_t>
    auto TGradientDescent<Architecture_t>::Train(const Data_t & trainingData,
                                                 size_t nTrainingSamples,
                                                 const Data_t & testData,
                                                 size_t nTestSamples,
                                                 Net_t & net,
                                                 size_t nThreads)
   -> Scalar_t
{
   // Reset iteration state.
   fMinimumError = 1e100;
   fConvergenceCount = 0;
   fStepCount = 0;

   // Prepare training data.
   bool converged = false;

   TDataLoader<Data_t, Architecture_t> trainLoader(trainingData, nTrainingSamples,
                                                   net.GetBatchSize(),
                                                   net.GetInputWidth(),
                                                   net.GetOutputWidth(), nThreads);
   auto testNet = net.CreateClone(nTestSamples);
   TDataLoader<Data_t, Architecture_t> testLoader(testData, nTestSamples,
                                                  testNet.GetBatchSize(),
                                                  testNet.GetInputWidth(),
                                                  net.GetOutputWidth());
   std::vector<Net_t> nets{};
   nets.reserve(nThreads);
   for (size_t i = 0; i < nThreads; i++) {
       nets.push_back(net);
       for (size_t j = 0; j < net.GetDepth(); j++)
       {
           std::cout << "copy" << std::endl;
           auto &masterLayer = net.GetLayer(j);
           auto &layer = nets.back().GetLayer(j);
           Architecture_t::Copy(layer.GetWeights(),
                                masterLayer.GetWeights());
           Architecture_t::Copy(layer.GetBiases(),
                                masterLayer.GetBiases());
       }
   }

   std::chrono::time_point<std::chrono::system_clock> start, end;
   start = std::chrono::system_clock::now();

   while (!converged)
   {
      fStepCount++;

      size_t netIndex = 0;
      for (auto b : trainLoader) {
         // Perform minimization step.
         auto inputMatrix  = b.GetInput();
         auto outputMatrix = b.GetOutput();
         Step(net, nets[netIndex % nThreads], inputMatrix, outputMatrix);
         netIndex++;
      }

      // Compute test error.
      if ((fStepCount % fTestInterval) == 0) {
         end   = std::chrono::system_clock::now();
         std::chrono::duration<double> elapsed_seconds = end - start;
         start = std::chrono::system_clock::now();
         double seconds = elapsed_seconds.count();
         double nFlops  = (double) (fTestInterval * (nTrainingSamples / net.GetBatchSize()));
                nFlops *= net.GetNFlops();
         std::cout << "Elapsed time for " << fTestInterval << " Epochs: "
                   << seconds << " [s] => " << nFlops * 1e-9 / seconds
                   << " GFlop/s" << std::endl;
         auto b = *testLoader.begin();
         auto inputMatrix  = b.GetInput();
         auto outputMatrix = b.GetOutput();

         Scalar_t loss = testNet.Loss(inputMatrix, outputMatrix);
         std::cout << fStepCount << ": " << loss << std::endl;
         converged = HasConverged();
      }

   }
   return fMinimumError;
}

//______________________________________________________________________________
template<typename Architecture_t>
    template <typename Net_t>
    void inline TGradientDescent<Architecture_t>::Step(Net_t & net,
                                                       Matrix_t &input,
                                                       const Matrix_t &output)
{
    //Scalar_t loss = net.Loss(input, output);
    //fTrainingError = loss;
    net.Forward(input);
    net.Backward(input, output);

    for (size_t i = 0; i < net.GetDepth(); i++)
    {
        auto &layer = net.GetLayer(i);
        Architecture_t::ScaleAdd(layer.GetWeights(),
                                 layer.GetWeightGradients(),
                                 -fLearningRate);
        Architecture_t::ScaleAdd(layer.GetBiases(),
                                 layer.GetBiasGradients(),
                                 -fLearningRate);
    }
}

//______________________________________________________________________________
template<typename Architecture_t>
    template <typename Net_t>
    void inline TGradientDescent<Architecture_t>::Step(Net_t & master,
                                                       Net_t & net,
                                                       Matrix_t &input,
                                                       const Matrix_t &output)
{
    //Scalar_t loss = net.Loss(input, output);
    //fTrainingError = loss;
    net.Forward(input);
    net.Backward(input, output);

    for (size_t i = 0; i < net.GetDepth(); i++)
    {
        auto &masterLayer = master.GetLayer(i);
        auto &layer = net.GetLayer(i);
        Architecture_t::ScaleAdd(masterLayer.GetWeights(),
                                 layer.GetWeightGradients(),
                                 -fLearningRate);
        Architecture_t::Copy(layer.GetWeights(),
                             masterLayer.GetWeights());
        Architecture_t::ScaleAdd(masterLayer.GetBiases(),
                                 layer.GetBiasGradients(),
                                 -fLearningRate);
        Architecture_t::Copy(layer.GetBiases(),
                             masterLayer.GetBiases());
    }
}


//______________________________________________________________________________
template<typename Architecture_t>
template <typename Net_t>
void inline TGradientDescent<Architecture_t>::StepReducedWeights(
    Net_t & net,
    Matrix_t &input,
    const Matrix_t &output)
{
   net.Forward(input);
   net.Backward(input, output);

   for (size_t i = 0; i < net.GetDepth(); i++)
   {
      auto &layer = net.GetLayer(i);
      Architecture_t::ScaleAdd(layer.GetWeights(),
                               layer.GetWeightGradients(),
                               -fLearningRate);
      if (i == 0) {
         Architecture_t::ScaleAdd(layer.GetBiases(),
                                  layer.GetBiasGradients(),
                                  -fLearningRate);
      }
   }
}

//______________________________________________________________________________
template<typename Architecture_t>
    template <typename Net_t>
    auto inline TGradientDescent<Architecture_t>::StepReducedWeightsLoss(
        Net_t & net,
        Matrix_t &input,
        const Matrix_t &output)
    -> Scalar_t
{
   Scalar_t loss = net.Loss(input, output);
   fTrainingError = loss;
   net.Backward(input, output);

   for (size_t i = 0; i < net.GetDepth(); i++)
   {
      auto &layer = net.GetLayer(i);
      Architecture_t::ScaleAdd(layer.GetWeights(),
                               layer.GetWeightGradients(),
                               -fLearningRate);
      if (i == 0) {
         Architecture_t::ScaleAdd(layer.GetBiases(),
                                  layer.GetBiasGradients(),
                                  -fLearningRate);
      }
   }
   return loss;
}

//______________________________________________________________________________
template<typename Architecture_t>
    template <typename Net_t>
    inline void TGradientDescent<Architecture_t>::TestError(Net_t & net,
                                                            Matrix_t &input,
                                                            const Matrix_t &output)
{
   fTestError = net.Loss(input, output, false);
}

//______________________________________________________________________________
template<typename Architecture_t>
bool inline TGradientDescent<Architecture_t>::HasConverged()
{
   if (fTestError < fMinimumError * 0.999) {
      fConvergenceCount = 0;
      fMinimumError     = fTestError;
   } else {
      fConvergenceCount++;
   }

   return (fConvergenceCount >= fConvergenceSteps);
}

} // namespace DNN
} // namespace TMVA

#endif
