#include <iostream>
#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestFullRNN.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::RNN;

int main()
{
   std::cout << "Training RNN to identity first";

   // testFullRNN(size_t batchSize, size_t stateSize, size_t inputSize, size_t outputSize)
   // reconstruct 8 bit vector
   // batchsize, statesize, inputsize, outputsize
   testFullRNN<TCudnn<double>>(2, 3, 2, 2);
   // testFullRNN<TReference<double>>(64, 10, 8, 8) ;
   // testFullRNN<TReference<double>>(3, 8, 100, 50) ;

   // test a full RNN with 5 time steps and different signal/backgrund time dependent shapes
   // batchsize, statesize , inputsize, seed
   int seed = 111;
   std::cout << "Training RNN to simple time dependent data ";
   testFullRNN2<TCudnn<double>>(64, 10, 5, seed);

   return 0;
}
