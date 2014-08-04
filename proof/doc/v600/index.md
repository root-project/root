## PROOF System

All the fixes and improvements in the PROOF system occured since the release of 5.34/00 are available both in the latest 5.34 tags and in 6.00/00.
The following is a summary of the major modifications since 5.34 .

### New developments/functionality

-   Several improvements in the merging phase; in particular:
   -   Modification of output sending protocol to control memory usage, significantly reducing the memory footprint on the master, in particular when merging
        large numbers of histograms.
   -   Use an hash table for the output list to significantly speed up names lookups during merging.
-   Add support for dynamic addition of workers to a currently running process (currently supported by the unit packetizer).
-   Automatization of the usage of file-based technology to handle outputs (see [Handling Outputs](http://root.cern.ch/drupal/content/handling-outputs)).
-   [Improved dataset management model](http://proof.web.cern.ch/proof/TDataSetManagerAliEn.html#a-proof-interface-to-the-alien-file-catalog)
    where the PROOF (ROOT) dataset manager is a light frontend to the experiment file catalogs; TDataSetManagerFile is still
    used as local cache of the experiment information or to store the work-in-progress status of the dataset manager daemon. This model addresses the scalability issues observed at ALICE AFs.
-   Improvements in [TProofBench](http://root.cern.ch/drupal/content/proof-benchmark-framework-tproofbench):
    -   Recording and display of the maximum rate during query, CPU efficiency calculation for PROOF-Lite runs, better measurement of wall time.
    -   Support for dynamic startup mode

-   Test program xpdtest to test the status of xproofd (see also man page under $ROOTSYS/man/man1):

``` {.sh}
    $ xpdtest [options]
       --help, -h
              Gives a short list of options avaliable, and exit
       -t <test>
              type of test to be run:
                  0       ping the daemon (includes process existence check if pid specified; see below)
                  1       ping the daemon and check connection for default user
                  2       ping the daemon and check connection for the default user and all recent users
...
```
-   Interface with **igprof** for fast statistic profiling. Like valgrind, it can be specified as option to TProof::Open and the output is available via the log viewer technology:

``` {.cpp}
    root[] p = TProof::Open("master", "igprof-pp")
```
-   Miscellanea:
   -   Added functions [Getenv](http://root.cern.ch/root/htmldoc/TProof.html#TProof:Getenv) and [GetRC](http://root.cern.ch/root/htmldoc/TProof.html#TProof:GetRC)
        in TProof to retrieve environment information from the nodes, typically from the master.
   -   Add support unix secondary groups in group access control. This allows more flexibility in, for example, assigning group-shared credential files to the daemon.
   -   Several new tests and options in the test program _stressProof_.

### Bug fixes

Several consolidation fixes in several parts of the system (see the [5.34 patch release notes for details](http://root.cern.ch/drupal/content/root-version-v5-34-00-patch-release-notes)). In particular, those for 'xproofd' were provided by B. Butler and  M. Swiatlowski and greatly contributed to consolidate the daemon.


