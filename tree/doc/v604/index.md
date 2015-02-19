
## TTree Libraries

### TTree Behavior change.

#### TTreeCache

The TTreeCache is now enabled by default.  The default size of the TTreeCache
is the estimated size of a cluster size for the TTree.  The TTreeCache
prefilling is also enabled by default; when in learning phase rather than
reading each requested branch individually, the TTreeCache will read all the
branches thus trading off the latencies inherent to multiple small reads for
the potential of requesting more data than needed by read from the disk or
server the baskets for too many branches.

The default behavior can be changed by either updating one of the rootrc files
or by setting environment variables.  The rootrc files, both the global and the
local ones, now support the following the resource variable TTreeCache.Size
which set the default ize factor for auto sizing TTreeCache for TTrees. The
estimated cluster size for the TTree and this factor is used to give the cache
size. If option is set to zero auto cache creation is disabled and the default
cache size is the historical one (equivalent to factor 1.0). If set to
non zero auto cache creation is enabled and both auto created and
default sized caches will use the configured factor: 0.0 no automatic cache
and greater than 0.0 to enable cache.  This value can be overridden by the
environment variable ROOT_TTREECACHE_SIZE.

The resource variable TTreeCache.Prefill sets the default TTreeCache prefilling
type.  The prefill type may be: 0 for no prefilling and 1 to prefill all
the branches.  It can be overridden by the environment variable ROOT_TTREECACHE_PREFILL

In particular the default can be set back to the same as in version 5 by
setting TTreeCache.Size (or ROOT_TTREECACHE_SIZE) and TTreeCache.Prefill
(or ROOT_TTREECACHE_PREFILL) both to zero.
