A PROOF interface to the AliEn file catalog
===========================================

Overview
--------

Datasets have been invented to provide PROOF users a cleaner access to
sets of uniform data: each dataset has a name which helps identifying
the kind of data stored, plus some meta-information, such as:

-   default tree name

-   number of events in the default tree

-   file size

-   integrity information: *is my file corrupted?*

-   locality information: *is my remote file available on a local
    storage?*

Datasets are also used by the [staging daemon
afdsmgrd](http://afdsmgrd.googlecode.com/) to trigger data staging,
*i.e.* to request some data from being transferred from a remote storage
to the local analysis facility disks.

PROOF datasets are handled by the *dataset manager*, a generic catalog of
datasets which has been historically implemented by the class
`TDataSetManagerFile`, which stored each dataset inside a ROOT file.

This dataset manager has been conceived for a small *(i.e., hundreds)*
number of datasets which reflected data stored on the local analysis facility
disks. As the PROOF analysis model became popular in ALICE, the number
of datasets grew posing many problems.

-   To give the possibility to process remote data, current datasets
    mimick file catalog functionalities by including also lists of files
    currently not staged on the local analysis facility.

-   Since users can create their own datasets, in many cases containing
    duplicate data, it has become demanding to provide maintenance and
    support.

-   Locality information in datasets is static: this means that, if a
    file gets deleted from a disk, the corresponding dataset(s) must be
    synchronized manually.

### An interface to the AliEn file catalog

The new `TDataSetManagerAliEn` class is a new dataset manager which acts
as an intermediate layer between PROOF datasets and the AliEn file
catalog.

Dataset names do not represent any longer a static list of files:
instead, it represents a **query string** to the AliEn file catalog that
creates a dataset dynamically.

**Locality information** is also filled on the fly by contacting the local
file server: for instance, in case a *xrootd* pool of disks is used,
fresh online information along with the exact host (endpoint) where each
file is located is provided dynamically in a reasonable amount of time.

Both file catalog queries and locality information are cached on ROOT
files: cache is shared between users and its expiration time is
configurable.

Since dataset information is now volatile, a separate and more
straightforward method for issuing staging requests has also been
provided.

Configuration
-------------

### PROOF

Using the new dataset manager requires the `xpd.datasetsrc` directive in
the xproofd configuration file:

    xpd.datasetsrc alien cache:/path/to/dataset/cache urltemplate:http://myserver:1234/data<path> cacheexpiresecs:86400

alien
:   Tells PROOF that the dataset manager is the AliEn interface (as
    opposed to `file`).

cache
:   Specify a path *on the local filesystem* of the host running user's
    PROOF master.

    > This path is not a URL but just a local path. Moreover, the path
    > must be visible from the host that will run each user's master,
    > since a separate dataset manager instance is created per user.

    > If the cache directory does not exist, it is created, if possible,
    > with open permissions (`rwxrwxrwx`). On a production environment
    > it is advisable to create the cache directory manually beforehand
    > with the same permissions.

urltemplate
:   Template used for translating between an `alien://` URL and the
    local storage's URL.

    `<path>` is written literally and will be substituted with the full
    AliEn path without the protocol.

    > An example on how URL translation works:
    >
    > -   Template URL:
    >
    >         root://alice-caf.cern.ch/<path>
    >
    > -   Source URL:
    >
    >         alien:///alice/data/2012/LHC12b/000178209/ESDs/pass1/12000178209061.17/AliESDs.root
    >
    > -   Resulting URL:
    >
    >         root://alice-caf.cern.ch//alice/data/2012/LHC12b/000178209/ESDs/pass1/12000178209061.17/AliESDs.root
    >
cacheexpiresecs
:   Number of seconds before cached information is considered expired
    and refetched *(e.g., 86400 for one day)*.

### PROOF-Lite

One of the advantages of such a dynamic AliEn catalog interface is that
it is possible to use it with PROOF-Lite.

By default, PROOF-Lite creates on the client session (which acts as a
master as well) a file-based dataset manager. To enable the AliEn
dataset manager in a PROOF-Lite session, run:

``` {.cpp}
gEnv->SetValue("Proof.DataSetManager",
  "alien cache:/path/to/dataset/cache "
  "urltemplate:root://alice-caf.cern.ch/<path> "
  "cacheexpiresecs:86400");
TProof::Open("");
```

where the parameters meaning has been described in the previous section.

> Please note that the environment must be set **before** opening the
> PROOF-Lite session!

Usage
-----

The new dataset manager is backwards-compatible with the legacy
interface: each time you want to process or obtain a dataset, instead of
specifying a string containing a dataset name you will specify a query
string to the file catalog.

### Query string format

The query string is the string you will use in place of the dataset
name. It does not correspond to a static dataset: instead it represents
a virtual dataset whose information is filled in on the fly.

There are two different formats you can use:

-   specify data features (such as period and run numbers) for **official
    data or Monte Carlo**

-   specify the **AliEn find** command parameters directly

In the query string it is also possible to specify if you want to
process data from AliEn, only staged data or data from AliEn in "cache
mode".

#### Official data and Monte Carlo format

These are the string formats to be used respectively for official data
and official Monte Carlo productions:

    Data;Period=<LHCPERIOD>;Variant=[ESDs|AODXXX];Run=<RUNLIST>;Pass=<PASS>

    Sim;Period=<LHCPERIOD>;Variant=[ESDs|AODXXX];Run=<RUNLIST>;

Period
:   The LHC period.

    Example of valid values: `LHC10h`, `LHC11h_2`, `LHC11f_Technical`

Variant
:   Data variant, which might be `ESDs` (or `ESD`) for ESDs and `AODXXX`
    for AODs corresponding to the *XXX* set.

    Example of valid values: `ESDs`, `AOD073`, `AOD086`

Run
:   Runs to be processed, in the form of a single run (`130831`), an
    inclusive range (`130831-130833`), or a list of runs and/or ranges
    (`130831-130835,130840,130842`).

    Duplicate runs are automatically removed, so in case you specify
    `130831-130835,130833` run number 130833 will be processed only
    once.

Pass *(only for data, not for Monte Carlo)*
:   The pass number or name. In case you specify only a number `X`, it
    will be expanded to `passX`.

    Example of valid values: `1`, `pass1`, `pass2`, `cpass1_muon`

This is an example of a full valid string:

    Data;Period=LHC10h;Variant=AOD086;Run=130831-130833;Pass=pass1

#### AliEn find format

Whenever a user would like to process data which has not been produced
officially, or whose directory structure in the AliEn file catalog is
non-standard, an interface to the AliEn shell's `find` command is
provided.

This is the command format:

    Find;BasePath=<BASEPATH>;FileName=<FILENAME>;Anchor=<ANCHOR>;TreeName=<TREENAME>;Regexp=<REGEXP>

Parameters `BasePath` and `FileName` are passed as-is to the AliEn [find
command](http://alien2.cern.ch/index.php?option=com_content&view=article&id=53&Itemid=99#Searching_for_files),
and are mandatory.

Parameters `Anchor`, `TreeName` and `Regexp` are optional.

Here's a detailed description of the parameters.

BasePath
:   Start search under the specified path on the AliEn file catalog.

    Jolly characters are supported: the asterisk (`*`) and the
    percentage sign (`%`) are interchangeable.

    Examples of valid values are:

        /alice/data/2010/LHC10h/000123456/*.*
        /alice/cern.ch/user/d/dummy/my_pp_production/%.%

FileName
:   File name to look for.

    Examples of valid values are: `root_archive.zip`, `aod_archive.zip`,
    `custom_archive.zip`, `AliAOD.root.

Anchor *(optional)*
:   In case `FileName` is a zip archive, the anchor is the name of a
    ROOT file inside the archive to point to.

    Examples of valid values are: `AliAOD.root`, `AliESDs.root`,
    `MyRootFile.root`.

    > Using the AliEn file catalog it is possible to point directly to a
    > ROOT file stored in an archive without using the anchor.
    >
    > There is however a substantial difference in how data is
    > retrieved, especially during staging: auxiliary ROOT files
    > *(friends)* are stored inside the archive along with the "main"
    > file, so that when you use the archive as `FileName` with the
    > proper `Anchor` you are still referencing to the same file, but
    > you are giving instructions of downloading the archive.
    >
    > Using the ROOT file name directly must be done in very special
    > cases (*i.e.*, to save space) and only when one is completely sure
    > that no external files in the archive are required for analysis.

TreeName *(optional)*
:   Name of each file's default tree.

    Examples of valid values are: `/aodTree`, `/esdTree`, `/myCustomTree`,
    `/TheDirectory/TheTree`.

Regexp *(optional)*
:   Additional extended regular expression applied after find command is
    run, to fine-grain search results.

    Only `alien://` paths matching the regular expression are
    considered, others are discarded.

    Examples of valid values are:

        /[0-9]{6}/[0-9]{3,4}
        \.root$

    > ROOT class
    > [TPMERegexp](http://root.cern.ch/root/html/TPMERegexp.html) is
    > used to perform regular expression matching.


Example of an AliEn raw find dataset string:

    Find;BasePath=/alice/data/2010/LHC10h/000139505/ESDs/pass1/*.*;FileName=root_archive.zip;Anchor=AliESDs.root

#### Data access modes

It is possible to append to the format string the `Mode` specifier that
affects the way URLs are generated.

    Mode=[local|remote|cache]

This parameter is optional and defaults to `local`. Description of each
possible value follows:

local
:   Local storage is checked for the presence of data you requested.
    Output URLs will be relative to your local storage. Also, locality
    information *(i.e., is your file staged?)* is filled.

    If you run a PROOF analysis on a dataset with this mode specified,
    only data marked as "staged" will be processed.

    This method is the preferred one, since it does not overload the
    remote storage, and it enables users to process partially-staged
    datasets, or partially-reconstructed runs, without the need to
    manually update static datasets.

    > This is the default if no mode is specified, and it is also the
    > most efficient one.
    >
    > Despite it might take some time (up to a couple of minutes to
    > locate ~4000 files), returned information is always reliable
    > (because it's dynamic) and speeds up analysis (because analysis
    > will always be run only on files having local copies).
    >
    > Moreover this information is cached for a configurable period of
    > time, so that subsequent calls to the same dataset will be faster.

remote
:   Only AliEn URLs are returned.

    A PROOF analysis run on a dataset with this mode specified will
    always obtain data from a remote storage, according to the AliEn
    file catalog.

    > Tasks run on remote data are usually much slower than using local
    > storage.

cache
:   URLs pointing to local copies of files are returned, but does not check
    whether the file is locally present or not.

    If local storage is configured for retrieving from AliEn files that
    are not available locally (which is the case of xrootd with vMSS),
    then data will be downloaded *while analysis is running*.

    It is called *cache mode* because it treats the local storage as a
    cache for the remote storage.

    > This mode is usually very slow on a busy analysis facility since
    > retrieving data in real time without any kind of scheduling is
    > inefficient. It also conflicts with the preferred method, which is
    > to stage data asynchronously using the [stager
    > daemon](http://afdsmgrd.googlecode.com/).

#### Force cache refresh

If the cached information for a certain AliEn file catalog query is wrong,
it is possible to force querying the catalog again by using the keyword
`ForceUpdate`:

    Data;ForceUpdate;Period=LHC10h;Variant=AOD086;Run=130831-130833;Pass=pass1

### Staging requests

Issuing staging requests and keeping track of them requires an auxiliary
database that can be read and updated by the [data stager
daemon](http://afdsmgrd.googlecode.com/).

Whenever a staging request is issued, a ROOT file containing the dataset
is saved in a special directory on the master's filesystem, monitored by
the file stager.

#### PROOF configuration

In the xproofd configuration file, there is a directive to specify the
directory used as repository for staging requests:

    xpd.stagereqrepo [dir:]/path/to/local/directory

> The literal `dir:` prefix is optional.

This directive is shared between PROOF and the stager daemon, so that the
same configuration file can be used for both.

Permissions on this directory must be kept open.

> Versions of the stager daemon prior to v1.0.7 do not support open
> permissions and the staging repository directive.

#### Request and monitor staging

Staging requests and monitoring can be done from within a PROOF session.

`gProof->RequestStagingDataSet("QueryString")`
:   Requests staging of the dataset specified via the query string.

    Staging request is honored if the stager daemon is running.

    > In order to avoid requesting to stage undesired data, it is
    > advisable to check in advance the results of your query string:
    >
    > `gProof->ShowDataSet("QueryString")`

`TProof->ShowStagingStatusDataSet("QueryString"[, "opts"])`
:   Shows progress status of a previously given staging request with
    data specified by the query string.

    Options are optional, and passed as-is to the `::Print()` method.

    > It is possible to show all the files marked as corrupted by the
    > daemon:
    >
    >     gProof->ShowStagingStatusDataSet("QueryString", "C")
    >
    > Or all the files successfully staged and not corrupted:
    >
    >     gProof->ShowStagingStatusDataSet("QueryString", "Sc")

`gProof->GetStagingStatusDataSet("QueryString")`
:   Gets a `TFileCollection` containing information on the staging
    request specified by the query string.

    Works exactly like `ShowStagingStatusDataSet()` but returns an
    object instead of displaying information on the screen.

`gProof->CancelStagingDataSet("QueryString")`
:   Removes a dataset from the list of staging requests. Datasets used
    as staging requests are usually removed automatically by the staging
    daemon if everything went right, so this command is used mostly to
    purge a completed staging request when it has some corrupted files.
