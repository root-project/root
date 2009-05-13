
   PQ2 (Proof Quick Query) dataset handling scripts
   ------------------------------------------------

   These scripts implement a shell-based interface to the PROOF dataset
   handling.

   The URL of the PROOF cluster can be passed on the command line or via
   the enviromnent variable PROOFURL.

   The scripts write temporary files in the temporary directory.

   1. Browsing scripts

      .1 pq2-ls [masterurl]

         list the available datasets

      .2 pq2-ls-files datasetname [masterurl]

         list the file content of a dataset

      .3 pq2-ls-files-server datasetname server [masterurl]

         list the files of a dataset available from a given server

      .4 pq2-info-server server [masterurl]

         show info about the datasets on a given server

   2. Modifiers scripts (may fail if the remote settings do not allow
      modification)

      .1 pq2-put datasetfile [masterurl]

         register one or more datasets

      .2 pq2-verify datasetname [masterurl]

         verify the content one or more datasets, caching the relevant info

      .3 pq2-rm datasetname [masterurl]

         remove one or more datasets


   Each script called with first argument '-h' or '--help' prints some
   additional information about the meaning of the arguments.

   --------------------------------------------------------------
   May 13, 2009       Creation
