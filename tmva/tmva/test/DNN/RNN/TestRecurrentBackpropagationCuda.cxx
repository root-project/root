#include <iostream>
#include "TMVA/DNN/Architectures/Cuda.h"
#include "TestRecurrentBackpropagation.h"
#include "TROOT.h"
#include "TMath.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::RNN;

int main()
{

   bool debug = true;
   std::cout << "Testing RNN backward pass\n";

   using Scalar_t = Double_t;

   using Architecture_t = TCuda<Scalar_t>;

   gRandom->SetSeed(12345);
   Architecture_t::SetRandomSeed(gRandom->Integer(TMath::Limits<UInt_t>::Max()));

   bool fail = false;
   if (debug) {
      //fail |= testRecurrentBackpropagation<Architecture_t>(2, 1, 1, 3, 1e-5, {true, true, false}, true);
      fail |= testRecurrentBackpropagation<Architecture_t>(2, 1, 4, 5, 1e-5, {true, true, false}, true);
      return fail;
   }

   // timesteps, batchsize, statesize, inputsize  { fixed input, with dense layer, with extra RNN }

   fail |= testRecurrentBackpropagation<Architecture_t>(1, 2, 1, 2, 1e-5);

   return fail;
}
