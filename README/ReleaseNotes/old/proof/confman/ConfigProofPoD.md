Setup a static PROOF cluster with PROOF on Demand
=================================================

Introduction
------------

Using PROOF on Demand is our current recommended way of running a PROOF
cluster. The usage of PoD is in particular helpful for the following
reasons:

-   **Sandboxing.** Each user get their own personal PROOF cluster,
    separated from the others: a problem occurring on one personal
    cluster does not affect the workflow of other users.

-   **Easier administration and self-servicing.** A user can restart their
    personal PROOF cluster in case of troubles without waiting for a
    system administrator's intervention.

-   **Efficient multiuser scheduling.** PROOF on Demand makes PROOF run on
    top of an existing resource management system, moving the problem of
    scheduling many concurrent users outside of PROOF.

This guide particularly refers to the setup of a static PROOF cluster
running on physical hosts: the recommended setup is in practice the same
as the ready-to-go Virtual Analysis Facility. If you want to use PROOF
on the clouds there is no configuration to go through.

Setup a resource management system
----------------------------------

Although PROOF on Demand can run on a cluster of nodes without using a
resource management system (using `pod-ssh`), it is recommended to setup a
dedicated one to benefit from the scheduling in a multiuser environment, or a
dedicated queue on an existing one.

As there's a variety of resource management systems, this guide does not cover
their setup. The RMS preconfigured for the Virtual Analysis Facility is
[HTCondor](http://research.cs.wisc.edu/htcondor/), which we recommend primarily
because it has dynamic addition of workers built in.

Configuration steps for all nodes
---------------------------------

### Setup CernVM-FS

[CernVM-FS](http://cernvm.cern.ch/portal/filesystem) should be installed
on all machines as the preferred method for software distribution.

> Configuration instructions for the latest CernVM-FS can be found
> [here](http://cernvm.cern.ch/portal/filesystem/techinformation).

A brief step-by-step procedure to install CernVM-FS is hereby described.

-   Download and install the latest stable version from
    [here](http://cernvm.cern.ch/portal/filesystem): pick one which is
    appropriate to your operating system. You need the `cvmfs` package,
    you *don't* need the `cvmfs-devel` or `cvmfs-server` ones.

-   As root user, run:

        # cvmfs_config setup

-   Start the `autofs` service: how to to this depends on your operating
    system.

    On Ubuntu using Upstart:

        # restart autofs

    On RHEL-based or older Ubuntus:

        # service autofs restart

-   Prepare a `/etc/cvmfs/default.local` file (create it if it does not
    exists) with the following configuration bits:

    ``` {.bash}
    CVMFS_HTTP_PROXY=http://your-proxy-server.domain.ch:3128,DIRECT
    CVMFS_REPOSITORIES=your-experiment.cern.ch,sft.cern.ch
    CVMFS_QUOTA_LIMIT=50000
    ```

    You need to properly specify your closest HTTP caching proxy:
    separate many of them via commas. The last fallback value, `DIRECT`,
    tells cvmfs to connect directly without using any proxy at all.

    Among the list of repositories (comma-separated), always specify
    `sft.cern.ch` and the one containing the software to your experiment
    (e.g., `cms.cern.ch`).

    The quota limit is, in Megabytes, the amount of local disk space to
    use as cache.

-   Check the configuration and repositories with:

        # cvmfs_config chksetup
        OK
        # cvmfs_config probe
        Probing /cvmfs/cms.cern.ch... OK
        Probing /cvmfs/sft.cern.ch... OK

> You might need special configurations for some custom software
> repositories! Special cases are not covered in this guide.

### Firewall configuration

[PROOF on Demand](http://pod.gsi.de/) is very flexible in handling
various cases of network topologies. The best solution would be to allow
all TCP communications between the cluster machines.

No other incoming communication is required from the outside.

Configuration steps for the head node only
------------------------------------------

### Setup HTTPS+SSH (sshcertauth) authentication

> Latest recommended sshcertauth version is 0.8.5.
>
> [Download](https://github.com/dberzano/sshcertauth/archive/v0.8.5.zip)
> and [read the
> instructions](http://newton.ph.unito.it/~berzano/w/doku.php?id=proof:sshcertauth).

If you want your users to connect to the PROOF cluster using their Grid
user certificate and private key you might be interested in installing
sshcertauth. Please refer to the [installation
guide](http://newton.ph.unito.it/~berzano/w/doku.php?id=proof:sshcertauth)
for further information.

### PROOF on Demand

> Latest recommended PROOF on Demand version is 3.12.
>
> **On CernVM-FS:** `/cvmfs/sft.cern.ch/lcg/external/PoD/3.12`
>
> **Source code:** [PoD download page](http://pod.gsi.de/download.html)
> and [Installation
> instructions](http://pod.gsi.de/doc/3.12/Installation.html)

[PROOF on Demand](http://pod.gsi.de/) is required on the head node and on the
user's client.

In case your experiment provides a version of PoD on CernVM-FS you can use
that one. Experiment-independent versions are available from the PH-SFT
cvmfs repository.

Only if you have specific reasons while you want to use a customly built
PoD version, download the source code and compile it using the
installation instructions.

Please note that [CMake](http://www.cmake.org/) and
[Boost](http://www.boost.org/) are required to build PoD.

-   After you have built PoD, install it with:

        make install

-   After installing PoD, run:

        pod-server getbins

    This has to be done only once and downloads the binary packages that
    will be dynamically transferred to the worker nodes as binary
    payload, and prevents us from installing PoD on each cluster node.

    It is important to do this step now, because in case PoD has been
    installed in a directory where the user has no write privileges, as
    in the case of system-wide installations, the user won't be able to
    download those required packages in the PoD binary directory.

> There is no need to "configure" PoD for your specific cluster: it is
> just enough to install it on your head node.
>
> PoD does not have any system-wide persistent daemon running or any
> system-wide configuration to be performed. Also, no part of PoD will
> be ever run as root.
>
> Do not worry about environment or software configuration at this time:
> there is no system configuration for that. All the environment for
> your software dependencies will be set via proper scripts from the PoD
> client.
>
> PoD client configuration and running is properly covered in the
> appropriate manual page.

### Firewall configuration

The head node only requires **TCP ports 22 (SSH) and 443 (HTTPS)** to accept
connections from the outside. Users will get an authentication "token"
from port 443 and all PROOF traffic will be automatically tunneled in a
SSH connection on port 22 by PoD.

In case you are not using the HTTPS+SSH token+authentication method, access to
the sole port 22 is all you need.
