## RooBatchCompute Library
_Contains optimized computation functions for PDFs that enable significantly faster fittings._
#### Note: This library is still at an experimental stage. Tests are being conducted continiously to ensure correctness of the results, but the interfaces and the instructions on how to use might change.

### Purpose
While fitting, a significant amount of time and processing power is spent on computing the probability function for every event and PDF involved in the fitting model. To speed up this process, roofit can use the computation functions provided in this library. The functions provided here process whole data arrays (batches) instead of a single event at a time, as in the legacy evaluate() function in roofit. In addition, the code is written in a manner that allows for compiler optimizations, notably auto-vectorization. This library is compiled multiple times for different [vector instuction set architectures](https://en.wikipedia.org/wiki/SIMD) and the optimal code is executed during runtime, as a result of an automatic hardware detection mechanism that this library contains. **As a result, fits can benefit by a speedup of 3x-16x.**

As of ROOT v6.26, RooBatchComputes also provides multithread and [CUDA](https://en.wikipedia.org/wiki/CUDA) instances of the computation functions, resulting in even greater improvements for fitting times.

### How to use
This library is an internal component of RooFit, so users are not supposed to actively interact with it. Instead, they can benefit from significantly faster times for fitting by calling `fitTo()` and providing a `BatchMode("cpu")` or a `BatchMode("cuda")` option. 
```c++
  // fit using the most efficient library that the computer's CPU can support
  RooMyPDF.fitTo(data, BatchMode("cpu")); 
  
  // fit using the CUDA library along with the most efficient library that the computer's CPU can support
  RooMyPDF.fitTo(data, BatchMode("cuda")); 
```
**Note: In case the system does not support vector instructions, the `RooBatchCompute::Cpu` option is guaranteed to work properly by using a generic CPU library. In contrast, users must first make sure that their system supports CUDA in order to use the `RooBatchCompute::Cuda` option. If this is not the case, an exception will be thrown.**

If `"cuda"` is selected, RooFit will launch CUDA kernels for computing likelihoods and potentially other intense computations. At the same time, the most efficent CPU library loaded will also handle parts of the computations in parallel with the GPU (or potentially, if it's faster, all of them), thus gaining full advantage of the available hardware. For this purpose `RooFitDriver`, a newly created RooFit class (in roofitcore) takes over the task of analyzing the computations and assigning each to the correct piece of hardware, taking into consideration the performance boost or penalty that may arise with every method of computing.

#### Multithread computations
The CPU instance of the computing library can furthermore execute multithread computations. This also applies for computations handled by the CPU in the `"cuda"` mode. To use them, one needs to set the desired number of parallel tasks before calling `fitTo()` as shown below:
```c++
  ROOT::EnableImplicitMT(nThreads);
  RooMyPDF.fitTo(data, BatchMode("cuda")); // can also use "cuda"
```

### User-made PDFs
The easiest and most efficient way of accelerating your PDFs is to request their addition to the official RooFit by submiting a ticket [here](https://github.com/root-project/root/issues/new). The ROOT team will gladly assist you and take care of the details.

While your code is integrated, you are able to significantly improve the speed of fitting (but not take full advantage of the RooBatchCompute library), at least by using the batch evaluation feature.
To make use of it, one should override `RooAbsReal::computeBatch()`
```c++
  void RooMyPDF::computeBatch(RooBatchCompute::RooBatchComputeInterface*, double* output, size_t nEvents, RooBatchCompute::DataMap& dataMap) const
```
This method must be implemented so that it fills the `output` array with the **normalized** probabilities computed for `nEvents` events, the data of which can be retrieved from `dataMap`. `dataMap` is a simple `std::map<RooRealVar*, RooSpan<const double>>`. Note that it is not necessary to evaluate any of the objects that the PDF relies to, because they have already been evaluated by the RooFitDriver, so that their updated results are always present in `dataMap`. The `RooBatchCompute::RooBatchComputeInterface` pointer should be ignored.

```c++
void RooMyPDF::computeBatch(RooBatchCompute::RooBatchComputeInterface*, double* output, size_t nEvents, RooBatchCompute::DataMap& dataMap) const
{
  // Retrieve `RooSpan`s for each parameter of the PDF
  RooSpan<const double> span1 = dataMap.at(&*proxyVar1);
  // or: auto span1 = dataMap.at(&*proxyVar1);
  RooSpan<const double> span2 = dataMap.at(&*proxyVar2);
  
  // let's assume c is a scalar parameter of the PDF. In this case the dataMap contains a RooSpan with only one value.
  RooSpan<const double> scalar = dataMap.at(&*c);
  
  // Perform computations in a for-loop
  // Use VDT if possible to facilitate auto-vectorization
  for (size_t i=0; i<nEvents; ++i) {
    output[i] = RooBatchCompute::fast_log(span1[i]+span2[i]) + scalar[0]; //scalar is a RooSpan of length 1
  }
}
```
Make sure to add the `computeBatch()` function signature in the header `RooMyPDF.h` and mark it as `override` to ensure that you have successfully overriden the method. As a final note, always remember to append `RooBatchCompute::` to the classes defined in the RooBatchCompute library, or write `using namespace RooBatchCompute`.
