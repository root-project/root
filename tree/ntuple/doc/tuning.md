# Cluster Sizes

A cluster contains all the data of a given event range.
As clusters are usually compressed and tied to event boundaries, an exact size cannot be enforced.
Instead, RNTuple uses a *target size* for the compressed data as a guideline for when to flush a cluster.

The default cluster target size is 128 MiB of compressed data.
The default can be changed by the `RNTupleWriteOptions`.
The default should work well in the majority of cases.
In general, larger clusters provide room for more and larger pages and should improve compression ratio and speed.
However, clusters also need to be buffered during write and (partially) during read,
so larger clusters increase the memory footprint.

A second option in `RNTupleWriteOptions` specifies the maximum uncompressed cluster size.
The default is 10x the default cluster target size, i.e. ~1.2 GiB.
This setting acts as an "emergency break" and should prevent very compressible clusters from growing too large.

Given the two settings, writing works as follows:
when the current cluster is larger than the maximum uncompressed size, it will be flushed unconditionally.
When the current cluster size reaches the estimate for the compressed cluster size, it will be flushed, too.
The estimated compression ratio for the first cluster is 0.5 if compression is used, and 1 otherwise.
The following clusters use the average compression ratio of all so-far written clusters as an estimate.
See the notes below on a discussion of this approximation.


# Page Sizes

Pages contain consecutive elements of a certain column.
They are the unit of compression and of addressability on storage.
RNTuple puts a configurable maximum uncompressed size for pages.
This limit is by default set to 1 MiB.
When the limit is reached, a page will be flushed to disk.

In addition, RNTuple maintains a memory budget for the combined allocated size of the pages that are currently filled.
By default, this limit is set to twice the compressed target cluster size when compression is used,
and to the cluster target size for uncompressed data.
Initially, and after flushing, all columns use small pages,
just big enough to hold the configurable minimum number of elements (64 by default).
Page sizes are doubled as more data is filled into them.
When a page reaches the maximum page size (see above), it is flushed.
When the overall page budget is reached,
pages larger than the page at hand are flushed before the page at hand is flushed.
For the parallel writer, every fill context maintains the page memory budget independently.

Note that the total amount of memory consumed for writing is usually larger than the write page budget.
For instance, if buffered writing is used (the default), additional memory is required.
Use RNTupleModel::EstimateWriteMemoryUsage() for the total estimated memory use for writing.

The default values are tuned for a total write memory of around 300 MB per writer resp. fill context.
In order to decrease the memory consumption,
users should decrease the target cluster size before tuning more intricate memory settings.

# Notes

## Approximation of the compressed cluster size

The estimator for the compressed cluster size uses the average compression factor
of the so far written clusters.
This has been choosen as a simple, yet expectedly accurate enough estimator (to be validated).
The following alternative strategies were discussed:

  - The average compression factor of all so-far written pages.
    Easy to implement.
    It would better prevent outlier clusters from skewing the estimate of the successor clusters.
    It would be slower though in adjusting to systematic changes in the data set,
    e.g. ones that are caused by changing experimental conditions during data taking

  - The average over a window of the last $k$ clusters, possibly with exponential smoothing.
    More code compared to the average compression factor or all so-far written clusters.
    It would be faster in adjusting to systematic changes in the data set,
    e.g. ones that are caused by changing experimental conditions during data taking.
    Could be a viable option if cluster compression ratios turn out to change significantly in a single file.

  - Calculate the cluster compression ratio from column-based individual estimators.
    More complex to implement and to recalculate the estimator on every fill,
    requires additional state for every column.
    One might reduce the additional state and complexity by only applying the fine-grained estimator for collections.
    Such an estimator would react better to a sudden change in the amount of data written for collections / columns
    that have substentially different compression ratios.

## Page Checksums

By default, RNTuple appends xxhash-3 64bit checksums to every compressed page.
Typically, checksums increase the data size in the region of a per mille.
As a side effect, page checksums allow for efficient "same page merging":
identical pages in the same cluster will be written only once.
On typical datasets, same page merging saves a few percent.
Conversely, turning off page checksums also disables the same page merging optimization.
