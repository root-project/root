# rootreadspeed

`rootreadspeed` is a tool used to help identify bottlenecks in root analysis programs
by providing an idea of what throughput you can expect when reading ROOT files in
certain configurations.

It does this by providing information about the number of bytes read from your files,
how long this takes, and the different throughputs in MB/s, both in total and per thread.


## Compressed vs Uncompressed Throughput:

Throughput speeds are provided as compressed and uncompressed - ROOT files are usually
saved in compressed format, so these will often differ. Compressed bytes is the total
number of bytes read from TFiles during the readspeed test (possibly including meta-data).
Uncompressed bytes is the number of bytes processed by reading the branch values in the TTree.
Throughput is calculated as the total number of bytes over the total runtime (including
decompression time) in the uncompressed and compressed cases.


## Interpreting results:

### There are three possible scenarios when using rootreadspeed, namely:

- The 'Real Time' is significantly lower than your own analysis runtime.
  This would imply your actual application code is dominating the runtime of your analysis,
  ie. your analysis logic or framework is taking up the time.
  The best way to decrease the runtime would be to optimize your code (or the framework's),
  parallelize it onto multiple threads if possible (for example with
  [RDataFrame](https://root.cern/doc/master/classROOT_1_1RDataFrame.html)
  and [EnableImplicitMT](https://root.cern/doc/master/namespaceROOT.html#a06f2b8b216b615e5abbc872c9feff40f))
  or switch to a machine with a more performant CPU.
- The 'Real Time' is significantly higher than 'CPU Time / number of threads'*.
  If the real time is higher than the CPU time per core it implies the reading of data is the
  bottleneck, as the CPU cores are wasting time waiting for data to arrive from your disk/drive
  or network connection in order to decompress it.
  The best way to decrease your runtime would be transferring the data you need onto a faster
  storage medium (ie. a faster disk/drive such as an SSD, or connecting to a faster network
  for remote file access), or to use a compression algorithm with a higher compression ratio,
  possibly at the cost of the decompression rate.
  Changing the number of threads is unlikely to help, and in fact using too many threads may
  degrade performance if they make requests to different regions of your local storage. 
  * If no '--threads' argument was provided this is 1, otherwise it is the minimum of the value
    provided and the number of threads your CPU can run in parallel. It is worth noting that -
    on shared systems or if running other heavy applications - the number of your own threads
    running at any time may be lower than the limit due to demand on the CPU.
- The 'Real Time' is similar to 'CPU Time / number of threads' AND 'Compressed Throughput' is lower than expected
  for your storage medium: this would imply that your CPU threads aren't decompressing data as fast as your storage
  medium can provide it, and so decompression is the bottleneck.
  The best way to decrease your runtime would be to utilise a system with a faster CPU, or make use
  use of more threads when running, or use a compression algorithm with a higher decompression rate such as LZ4,
  possibly at the cost of some extra file size.


### A note on caching

If your data is stored on a local disk, the system may cache some/all of the file in memory after it is
first read. If this is realistic of how your analysis will run - then there is no concern. However, if
you expect to only read files once in a while - and as such the files are unlikely to be in the cache -
consider clearing the cache before running rootreadspeed.
On Linux this can be done by running 'echo 3 > /proc/sys/vm/drop_caches' as a superuser,
or a specific file can be dropped from the cache with
`dd of=<FILENAME> oflag=nocache conv=notrunc,fdatasync count=0 > /dev/null 2>&1`.


### Known overhead of TTreeReader, RDataFrame

`rootreadspeed` is designed to read all data present in the specified branches, trees and files at the highest
possible speed. When the application bottleneck is not in the computations performed by analysis logic,
higher-level interfaces built on top of TTree such as TTreeReader and RDataFrame are known to add a significant
runtime overhead with respect to the runtimes reported by `rootreadspeed` (up to a factor 2). In realistic analysis
applications it has been observed that a large part of that overhead is compensated by the ability of TTreeReader and
RDataFrame to read branch values selectively, based on event cuts, and this overhead will be reduced significantly
when using RDataFrame in conjunction with RNTuple.
See also [this talk](https://indico.cern.ch/e/PPP138) (slides 16 to 19).
