The Dataset Stager
==================

Overview
--------

The [Dataset Stager (afdsmgrd)](http://afdsmgrd.googlecode.com/) is
a daemon that coordinates the transfer of data from a remote storage
to your local storage.

For each file to transfer, a script is called. The script can be
customized to support your source and destination protocol.

Staging requests are issued from the ROOT console, where you can also
control the progress of your staging. How to request stagings and how to
check current transfer progress from ROOT is explained in the [PROOF interface
to AliEn file catalog documentation](TDataSetManagerAliEn.html).

Installation
------------

The Dataset Stager is distributed both on a repository on its own and as
part of ROOT. The easiest way to compile it is to do it inside ROOT.

Installing from ROOT
--------------------

When configuring the ROOT source, enable the Dataset Stager by adding
`--enable-afdsmgrd`. Check in the list of enabled features if you have
"afdsmgrd".

After running `make` (and, optionally, `make install`) you'll find the
daemon in the same directory of `root.exe`.

The configuration file and init.d startup script will be in
`$ROOTSYS/etc/proof`. The daemon can and **must** run as unprivileged
user.

Configuration
-------------

The Dataset Stager can share its configuration file with PROOF, as
some directives are the same and unknown directives are just ignored.

Directives are one per line and lines beginning with a pound sign (`#`)
are used for comments.

> The configuration file is automatically checked at each loop: this
> means you can change configuration without restarting the daemon or
> stopping your current transfers.

A detailed description of each directive follows.

set *VARIABLE=value*
:   This statement will substitute every occurrence of `$VARIABLE` with
    its *value* in the rest of the configuration file. You can have
    multiple `set` statements.

xpd.stagereqrepo [dir:]*directory*
:   This directive is shared with PROOF: *directory* is the full path to
    the dataset repository. **Defaults to empty:** without this
    directive the daemon is not operative.

    The `dir:` prefix is optional.

dsmgrd.purgenoopds *true|false*
:   Set it to *true* **(default is false)** to remove a dataset when no file to stage
    is found. If no file to stage is found, but corrupted files exist, the
    dataset is kept to signal failures. Used in combination with `xpd.stagereqrepo`
    makes it "disposable": only the datasets effectively needed for signaling
    the staging status will be kept, improving scalability and stability.

dsmgrd.urlregex *regex* *subst*
:   Each source URL present in the datasets will be matched to *regex*
    and substituted to *subst*. *regex* supports grouping using
    parentheses, and groups can be referenced in order using the dollar
    sign with a number (`$1` for instance) in *subst*.

    Matching and substitution for multiple URL schemas are supported by
    using in addition directives `dsmgrd.urlregex2` up to
    `dsmgrd.urlregex4` which have the same syntax of this one.

    Example of URL translation via regexp:

    > -   Configuration line:
    >
    >         dsmgrd.urlregex alien://(.*)$ root://xrd.cern.ch/$1
    >
    > -   Source URL:
    >
    >         alien:///alice/data/2012/LHC12b/000178209/ESDs/pass1/12000178209061.17/AliESDs.root
    >
    > -   Resulting URL:
    >
    >         root://xrd.cern.ch//alice/data/2012/LHC12b/000178209/ESDs/pass1/12000178209061.17/AliESDs.root
    >
dsmgrd.sleepsecs *secs*
:   Seconds to sleep between each loop. The dataset stager checks at
    each loop the status of the managed transfers. Defaults to **30
    seconds**.

dsmgrd.scandseveryloops *n*
:   Every `n` loops, the dataset repository is checked for newly
    incoming staging requests. Defaults to **10**.

dsmgrd.parallelxfrs *n*
:   Number of concurrent transfers. Defaults to **8**.

dsmgrd.stagecmd *shell\_command*
:   Command to run in order to stage each file. It might be whatever you
    want (executable, shell script...). If you add `$URLTOSTAGE` and/or
    `$TREENAME` in the *shell\_command*, they'll be substituted
    respectively with the destination URL and the default ROOT tree name
    in the file (as specified in the dataset staging request from ROOT).

    An example:

        dsmgrd.stagecmd /path/to/afdsmgrd-xrd-stage-verify.sh "$URLTOSTAGE" "$TREENAME"

    Return value of the command is ignored: standard output is
    considered, as explained here.

    Defaults to `/bin/false`.

dsmgrd.cmdtimeoutsecs *secs*
:   Timeout on staging command, expressed in seconds: after this
    timeout, the command is considered failed and it is killed (in first
    place with `SIGSTOP`, then if it is unresponsive with `SIGKILL`).
    Defaults to **0 (no timeout)**.

dsmgrd.corruptafterfails *n*
:   Set this to a number above zero to tell the daemon to mark files as
    corrupted after a certain number of either download or verification
    failures. A value of **0 (default)** tells the daemon to retry
    forever.

Configuring the MonALISA monitoring plugin
------------------------------------------

The Dataset Stager supports generic monitoring plugins. The only plugin
distributed with the stager is the MonALISA monitoring plugin.

dsmgrd.notifyplugin */path/to/libafdsmgrd\_notify\_apmon.so*
:   Set it to the path of the MonALISA plugin shared object. By default,
    notification plugin is disabled.

dsmgrd.apmonurl *apmon://apmon.cern.ch*
:   This variable tells the ApMon notification plugin how to contact one
    or more MonALISA server(s) to activate monitoring via ApMon. It
    supports two kinds of URLs:

    -   `http[s]://host/path/configuration_file.conf` (a remote file
        where to fetch the list of servers from)

    -   `apmon://[:password@]monalisahost[:8884]` (a single server to
        contact directly)

    If the variable is not set, yet the plugin is loaded, MonALISA
    monitoring is inhibited until a valid configuration variable is
    provided.

dsmgrd.apmonprefix *MY::CLUSTER::PREFIX*
:   Since MonALISA organizes information in "clusters" and "hosts", here
    you can specify what to use as cluster prefix for monitoring
    datasets information and daemon status. If this variable is not set,
    MonALISA monitoring is inhibited. Please note that the suffix
    `_datasets` or `_status` is appended for each of the two types of
    monitoring.

A sample configuration file
---------------------------

    xpd.stagereqrepo /opt/aaf/var/proof/datasets
    dsmgrd.purgenoopds true
    dsmgrd.urlregex alien://(.*)$ /storage$1
    dsmgrd.sleepsecs 20
    dsmgrd.scandseveryloops 30
    dsmgrd.parallelxfrs 10
    dsmgrd.stagecmd /opt/aaf/bin/af-xrddm-verify.sh "$URLTOSTAGE" "$TREENAME"
    dsmgrd.cmdtimeoutsecs 3600
    dsmgrd.corruptafterfails 0
