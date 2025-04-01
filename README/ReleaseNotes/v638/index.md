## Deprecation and Removal

* The `RooDataSet` constructors to construct a dataset from a part of an existing dataset were deprecated in ROOT 6.36 and are now removed. This is to avoid interface duplication. Please use `RooAbsData::reduce()` instead, or if you need to change the weight column, use the universal constructor with the `Import()`, `Cut()`, and `WeightVar()` arguments.
* The `RooStats::HLFactory` class that was deprecated in ROOT 6.36 is now removed. It provided little advantage over using the RooWorkspace directly or any of the other higher-level frameworks that exist in the RooFit ecosystem.

## RooFit

### Error out when setting out-of-range variable value instead of silent clipping

In previous versions, if you set the value of a variable with `RooRealVar::setVal()`, the value was silently clippend when it was outside the variable range.
This silent mutation of data can be dangerous.
With ROOT 6.38, an exception will be thrown instead.
If you know what you are doing and want to restore the old clipping behavior, you can do so with `RooRealVar::enableSilentClipping()`, but this is not recommended.

## RDataFrame
- Memory savings in RDataFrame: When many Histo3D are filled in RDataFrame, the memory consumption in multi-threaded runs can be prohibitively large, because
  RDF uses one copy of each histogram per thread. Now, RDataFrame can reduce the number of clones using `ROOT::RDF::Experimental::ThreadsPerTH3()`. Setting this
  to numbers such as 8 would share one 3-d histogram among 8 threads, greatly reducing the memory consumption. This might slow down execution if the histograms
  are filled at very high rates. Use lower number in this case.

## Versions of built-in packages

* The version of openssl has been updated to 3.5.0
