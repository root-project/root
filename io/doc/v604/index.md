## I/O Libraries

### I/O Behavior change.

#### TTreeCache

The TTreeCache is now enabled by default.  The default size of the TTreeCache
is the estimated size of a cluster size for the TTree.  The TTreeCache
prefilling is also enabled by default; when in learning phase rather than
reading each requested branch individually, the TTreeCache will read all the
branches thus trading off the latencies inherent to multiple small reads for
the potential of requesting more data than needed by read from the disk or
server the baskets for too many branches.
