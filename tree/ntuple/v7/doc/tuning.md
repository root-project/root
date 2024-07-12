Cluster Sizes
=============

A cluster contains all the data of a given event range.
As clusters are usually compressed and tied to event boundaries, an exact size cannot be enforced.
Instead, RNTuple uses a *target size* for the compressed data as a guideline for when to flush a cluster.

The default cluster target size is 50MB of compressed data.
The default can be changed by the `RNTupleWriteOptions`.
The default should work well in the majority of cases.
In general, larger clusters provide room for more and larger pages and should improve compression ratio and speed.
However, clusters also need to be buffered during write and (partially) during read,
so larger clusters increase the memory footprint.

A second option in `RNTupleWriteOptions` specifies the maximum uncompressed cluster size.
The default is 512MiB.
This setting acts as an "emergency break" and should prevent very compressible clusters from growing too large.

Given the two settings, writing works as follows:
when the current cluster is larger than the maximum uncompressed size, it will be flushed unconditionally.
When the current cluster size reaches the estimate for the compressed cluster size, it will be flushed, too.
The estimated compression ratio for the first cluster is 0.5 if compression is used, and 1 otherwise.
The following clusters use the average compression ratio of all so-far written clusters as an estimate.
See the notes below on a discussion of this approximation.


Page Sizes
==========

Pages contain consecutive elements of a certain columns.
They are the unit of compression and of addressability on storage.
RNTuple uses a *target size* for the uncompressed data as a guideline for when to flush a page.

The default page target size is 64KiB.
The default can be changed by the `RNTupleWriteOptions`.
In general, larger pages give better compression ratios; smaller pages reduce the memory footprint.
When reading, every active column requires at least one page buffer.
For the number of read requests, the page size does not matter
because pages of the same column are written consecutively and therefore read in one go.

Given the target size, writing works as follows:
In the beginning, the first page is filled until the target size.
Afterwards there is a mechanism to prevent undersized tail pages:
writing uses two page buffers in turns and flushes the previous buffer filled to its target size only once the next buffer is at least at 50%.
Then writing continues until the target size, at which point writing switches back to the other page.
If the cluster gets flushed with an undersized tail page,
the small page is appended to the previous page before flushing.
Therefore, tail pages sizes are between `[0.5 * target size .. 1.5 * target size]`
(unless the column doesn't have enough elements to fill 50% of the first page).

Concretely, writing will fill and flush two pages `A` and `B` as follows:
1. Writing starts to fill page `A`.
2. When page `A` reached its target size, writing switches to page `B` while the contents of page `A` are kept in memory.
3. Once page `B` is at least at 50%, page `A` is flushed.
4. When page `B` reaches its target size, writing switches to page `A`.
5. Once page `A` is at least at 50%, page `B` is flushed.
6. ... and so on, going back to 2.


Notes
=====

Approximation of the compressed cluster size
--------------------------------------------

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

Page Checksums
--------------

By default, RNTuple appends xxhash-3 64bit checksums to every compressed page.
Typically, checksums increase the data size in the region of a per mille.
As a side effect, page checksums allow for efficient "same page merging":
identical pages in the same cluster will be written only once.
On typical datasets, same page merging saves a few percent.
Conversely, turning off page checksums also disables the same page merging optimization.
