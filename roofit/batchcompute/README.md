## RooBatchCompute Library
_Contains optimized computation functions for PDFs that enable significantly faster fittings._
#### Note: This library is still at an experimental stage. Tests are being conducted continiously to ensure correctness of the results, but the interfaces and the instructions on how to use might change.

### Purpose
While fitting, a significant amount of time and processing power is spent on computing the probability function for every event and PDF involved in the fitting model. To speed up this process, roofit can use the computation functions provided in this library. The functions provided here process whole data arrays (batches) instead of a single event at a time, as in the legacy evaluate() function in roofit. In addition, the code is written in a manner that allows for compiler optimizations, notably auto-vectorization. This library is compiled multiple times for different [vector instuction set architectures](https://en.wikipedia.org/wiki/SIMD) and the optimal code is executed during runtime, as a result of an automatic hardware detection mechanism that this library contains. **As a result, fits can benefit by a speedup of 3x-16x.**

### How to use
The easiest and most efficient way of accelerating your PDFs is to request their addition to the official RooFit by submiting a ticket [here](https://github.com/root-project/root/issues/new). The ROOT team will gladly assist you and take care of the details.

The above process might take some time and the users will be required to update ROOT to use the newly introduced PDFs. In the meantime, you are able to significantly improve the speed of fitting (but not take full advantage of the RooBatchCompute library), at least by using the batch evaluation feature.
To make use of it, one should override [`RooAbsReal::evaluateSpan()`](https://root.cern.ch/doc/master/classRooAbsReal.html#a1e5129ffbc63bfd04c01511fd354b1b8)
```c++
  RooSpan<double> RooMyPDF::evaluateSpan(RooBatchCompute::RunContext& evalData, const RooArgSet* normSet) const
```
The evalData is a simple struct that holds the vector data for the fitting in the form of `RooSpan<double>`.
The normSet (normalization set) is used for invoking the computation and retrieving the values of the variables of the PDF.
You don't need to worry about these arguments as they will be provided by the RooFit internal functions that will call `evaluateSpan()`.
```c++
RooSpan<double> RooMyPDF::evaluateSpan(RooBatchCompute::RunContext& evalData, const RooArgSet* normSet) const
{
  // Retrieve `RooSpan`s for each parameter of the PDF
  RooSpan<const double> span1 = var1->getValues(evalData, normSet);
  // or: auto span1 = var1->getValues(evalData, normSet);
  RooSpan<const double> span2 = var2->getValues(evalData, normSet);
  
  // let's assume c is a scalar parameter of the PDF. In this case getValues will return a RooSpan with only one value.
  RooSpan<const double> scalar = c->getValues(evalData, normset);

  // Get the number of nEvents
  size_t nEvents=0;
  for (auto& i:{xData,meanData,sigmaData})
    nEvents = std::max(nEvents,i.size());

  // Allocate the output array
  evalData.makeBatch(this, nEvents);
  
  // Perform computations in a for-loop
  // Use VDT if possible to facilitate auto-vectorization
  for (size_t i=0; i<nEvents; ++i) {
    output[i] = RooBatchCompute::fast_log(span1[i]+span2[i]) + scalar[0]; //scalar is a RooSpan of length 1
  }
  return output;
}
```
Make sure to add the `evaluateSpan()` function signature in the header `RooMyPDF.h` and mark it as `override` to ensure that you have successfully overriden the method. In case the data types (scalar or vector) for the variables can not be predicted when writing the source code, you can use [BracketAdapterWithMask](https://github.com/root-project/root/blob/2b84398d4f52462a120083b3c5d1e0b952cc5221/roofit/batchcompute/inc/BracketAdapter.h#L55). This class overloads the `operator[]` and is constructed by a RooSpan. In case the RooSpan used for construction has a length of 1, ie represents a scalar variable, then `BracketAdapterWithMask::operator[]` always returns the scalar value, regrdless of the index used. This allows us to write:

```c++
RooSpan<double> RooMyPDF::evaluateSpan(RooBatchCompute::RunContext& evalData, const RooArgSet* normSet) const
{
  // Construct BracketAdapterWithMasks for each variable if we're not sure whether they are scalar of vectors.
  BracketAdapterWithMask adapter1(var1->getValues(evalData, normSet));
  BracketAdapterWithMask adapter2(var2->getValues(evalData, normSet));
  BracketAdapterWithMask scalar(c->getValues(evalData, normSet));
 
  // prepare the computations as above
  ...
  
  // by calling adapter[i] we either get the i-th or 0-th element, if the variable is a vector or a scalar respectively.
  for (size_t i=0; i<nEvents; ++i) {
    output[i] = RooBatchCompute::fast_log(adapter1[i]+adapter2[i]) + scalar[i]; 
  }
  return output;
}
  ```
  
  As a final note, always remember to append `RooBatchCompute::` to the classes defined in the RooBatchCompute library, or write `using namespace RooBatchCompute`.
