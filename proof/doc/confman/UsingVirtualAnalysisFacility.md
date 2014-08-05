Using the Virtual Analysis Facility
===================================

Introduction
------------

The Virtual Analysis Facility can be easily used by having installed on
your client the following software:

-   [ROOT](http://root.cern.ch/)

-   [PROOF on Demand](http://pod.gsi.de/)

-   The VAF client *(see below)*: a convenience tool that sets up the
    environment for your experiment's software both on your client and
    on the PROOF worker nodes

> If you are the end user, you'll probably might skip the part that
> concerns how to configure the VAF client: your system administrator
> has probably and conveniently set it up for you.

The Virtual Analysis Facility client
------------------------------------

The Virtual Analysis Facility client takes care of setting the
environment for the end user required by your software's experiment. The
environment will both be set on the client and on each PROOF node.

Technically it is a Bash shell script which provides shortcuts for PROOF
on Demand commands and ensures local and remote environment consistency:
by executing it you enter a new clean environment where all your
software dependencies have already been set up.

Local and remote environment configuration is split into a series of
files, which give the possibility to:

-   have a system-wide, sysadmin-provided experiment configuration

-   execute user actions either *before* or *after* the execution of the
    system-wide script (for instance, choosing the preferred version of
    the experiment's software)

-   transfer a custom user **payload** on each PROOF worker (for instance,
    user's client-generated Grid credentials to make PROOF workers
    capable of accessing a remote authenticated storage)

Configuration files are searched for in two different locations:

-   a system-wide directory: `<client_install_dir>/etc`

-   user's home directory: `~/.vaf`

> A system-wide configuration file always has precedence over user's
> configuration. It is thus possible for the sysadmin to enforce a
> policy where some scripts cannot ever be overridden.

Thanks to this separation, users can maintain an uncluttered directory
with very simple configuration files that contain only what really needs
or is allowed to be customized: for instance, user might specify a single line
containing the needed ROOT version, while all the technicalities to set
up the environment are taken care of inside system-installed scripts,
leaving the user's configuration directory clean and uncluttered.

### Local environment configuration

All the local environment files are loaded at the time of the
client's startup following a certain order

-   `common.before`

-   `local.before`

-   `local.conf`

-   `$VafConf_LocalPodLocation/PoD_env.sh`

-   `common.after`

-   `local.after`

The `common.*` files are sourced both for the local and the remote
environment. This might be convenient to avoid repeating the same
configuration in different places.

Each file is looked for first in the system-wide directory and then in
the user's directory. If a configuration file does not exist, it is
silently skipped.

The `$VafConf_LocalPodLocation/PoD_env.sh` environment script, provided
with each PROOF on Demand installation, *must exist*: without this file,
the VAF client won't start.

### List of VAF-specific variables

There are some special variables that need to be set in one of the above
configuration files.

`$VafConf_LocalPodLocation`
:   Full path to the PoD installation on the client.

    > The `$VafConf_LocalPodLocation` variable must be set before the
    > `PoD_env.sh` script gets sourced, so set it either in
    > `common.before`, `local.before` or `local.conf`. Since PoD is
    > usually system-wide installed, its location is normally
    > system-wide set in either the `local.conf` file by the system
    > administrator.

`$VafConf_RemotePodLocation`
:   Full path to the PoD installation on the VAF master node.

    *Note: this variable should be set in the configuration files for
    the local environment despite it refers to a software present on the
    remote nodes.*

`$VafConf_PodRms` *(optional)*
:   Name of the Resource Management System used for submitting PoD jobs.
    Run `pod-submit -l` to see the possible values.

    If not set, defaults to `condor`.

`$VafConf_PodQueue` *(optional)*
:   Queue name where to submit PoD jobs.

    If no queue has been given, the default one configured on your RMS
    will be used.

### Remote environment configuration

All the PoD commands sent to the VAF master will live in the environment
loaded via using the following scripts.

Similarly to the local environment, configuration is split in different files
to allow for a system-wide configuration, which has precedence over
user's configuration in the home directory. If a script cannot be found,
it will be silently skipped.

-   `<output_of_payload>`

-   `common.before`

-   `remote.before`

-   `remote.conf`

-   `common.after`

-   `remote.after`

For an explanation on how to pass extra data to the workers safely
through the payload, see below.

### Payload: sending local files to the remote nodes

In many cases it is necessary to send some local data to the remote
workers: it is very common, for instance, to distribute a local Grid
authentication proxy on the remote workers to let them authenticate to
access a data storage.

The `payload` file must be an executable generating some output that
will be prepended to the remote environment preparation. Differently
than the other environment scripts, it is not executed: instead, it is
first run, then *the output it produces will be executed*.

Let's see a practical example to better understand how it works. We need
to send our Grid proxy to the master node.

This is our `payload` executable script:

``` {.bash}
#!/bin/bash
echo "echo '`cat /tmp/x509up_u$UID | base64 | tr -d '\r\n'`'" \
  "| base64 -d > /tmp/x509up_u\$UID"
```

This script will be executed locally, providing another "script line" as
output:

``` {.bash}
echo 'VGhpcyBpcyB0aGUgZmFrZSBjb250ZW50IG9mIG91ciBHcmlkIHByb3h5IGZpbGUuCg==' | base64 -d > /tmp/x509up_u$UID
```

This line will be prepended to the remote environment script and will be
executed before anything else on the remote node: it will effectively
decode the Base64 string back to the proxy file and write it into the
`/tmp` directory. Note also that the first `$UID` is not escaped and
will be substituted *locally* with your user ID *on your client
machine*, while the second one has the dollar escaped (`\$UID`) and will
be substituted *remotely* with your user ID *on the remote node*.

> It is worth noting that the remote environment scripts will be sent to
> the remote node using a secure connection (SSH), thus there is no
> concern in placing sensitive user data there.

Installing the Virtual Analysis Facility client
-----------------------------------------------

### Download the client from Git

The Virtual Analysis Facility client is available on
[GitHub](https://github.com/dberzano/virtual-analysis-facility):

``` {.bash}
git clone git://github.com/dberzano/virtual-analysis-facility.git /dest/dir
```

The client will be found in `/dest/dir/client/bin/vaf-enter`: it is
convenient to add it to the `$PATH` so that the users might simply start
it by typing `vaf-enter`.

### Install the experiment's configuration files system-wide

A system administrator might find convenient to install the experiment
environment scripts system-wide.

Configuration scripts for LHC experiments are shipped with the VAF
client and can be found in
`/dest/dir/client/config-samples/<experiment_name>`. To make them used
by default by the VAF client, place them in the `/dest/dir/etc`
directory like this:

``` {.bash}
rsync -a /dest/dir/client/config-samples/<experiment_name>/ /dest/dir/etc/
```

Remember that the trailing slash in the source directory name has a
meaning in `rsync` and must not be omitted.

> Remember that system-wide configuration files will always have
> precedence over user's configuration files, so *don't place there
> files that are supposed to be provided by the user!*

Entering the Virtual Analysis Facility environment
--------------------------------------------------

The Virtual Analysis Facility client is a wrapper around commands sent
to the remote host by means of PROOF on Demand's `pod-remote`. The VAF
client takes care of setting up passwordless SSH from your client node
to the VAF master.

### Getting the credentials

> You can skip this paragraph if the remote server wasn't configured for
> HTTPS+SSH authentication.

In our example we will assume that the remote server's name is
`cloud-gw-213.to.infn.it`: substitute it with your remote endpoint.

First, check that you have your Grid certificate and private key
installed both in your browser and in the `~/.globus` directory of your
client.

Point your browser to `https://cloud-gw-213.to.infn.it/auth/`: you'll
probably be asked for a certificate to choose for authentication. Pick
one and you'll be presented with the following web page:

![Web authentication with sshcertauth](img/sshcertauth-web.png)

The webpage clearly explains you what to do next.

### Customizing user's configuration

Before entering the VAF environment, you should customize the user's
configuration. How to do so depends on your experiment, but usually you
should essentially specify the version of the experiment's software you
need.

For instance, in the CMS use case, only one file is needed:
`~/.vaf/common.before`, which contains something like:

``` {.bash}
# Version of CMSSW (as reported by "scram list")
export VafCmsswVersion='CMSSW_5_3_9_sherpa2beta2'
```

### Entering the VAF environment

Open a terminal on your client machine (can be either your local
computer or a remote user interface) and type:

    vaf-enter <username>@cloud-gw-213.to.infn.it

You'll substitute `<username>` with the username that either your system
administrator or the web authentication (if you used it) provided you.

You'll be presented with a neat shell which looks like the following:

    Entering VAF environment: dberzano@cloud-gw-213.to.infn.it
    Remember: you are still in a shell on your local computer!
    pod://dberzano@cloud-gw-213.to.infn.it [~] >

This shell runs on your local computer and it has the environment
properly set up.

PoD and PROOF workflow
----------------------

> The following operations are valid inside the `vaf-enter` environment.

### Start your PoD server

With PROOF on Demand, each user has the control of its own personal
PROOF cluster. The first thing to do is to start the PoD server and the
PROOF master like this:

    vafctl --start

A successful output will be similar to:

    **    Starting remote PoD server on dberzano@cloud-gw-213.to.infn.it:/cvmfs/sft.cern.ch/lcg/external/PoD/3.12/x86_64-slc5-gcc41-python24-boost1.53
    **  Server is started. Use "pod-info -sd" to check the status of the server.

### Request and wait for workers

Now the server is started but you don't have any worker available. To
request for `<n>` workers, do:

    vafreq <n>

To check how many workers became available for use:

    pod-info -n

To continuously update the check (`Ctrl-C` to terminate):

    vafcount

Example of output:

    Updating every 5 seconds. Press Ctrl-C to stop monitoring...
    [20130411-172235] 0
    [20130411-172240] 0
    [20130411-172245] 12
    [20130411-172250] 12
    ...

To execute a command after a certain number of workers is available (in
the example we wait for 5 workers then start ROOT):

    vafwait 5 && root -l

> Workers take some time before becoming available. Also, it is possible
> that not all the requested workers will be satisfied.

### Start ROOT and use PROOF

When you are satisfied with the available number of active workers, you
may start your PROOF analysis. Start ROOT, and from its prompt connect
to PROOF like this:

    root [0] TProof::Open("pod://");

Example of output:

    Starting master: opening connection ...
    Starting master: OK
    Opening connections to workers: OK (12 workers)
    Setting up worker servers: OK (12 workers)
    PROOF set to parallel mode (12 workers)

### Stop or restart your PoD cluster

At the end of your session, remember to free the workers by stopping
your PoD server:

    vafctl --stop

> PoD will stop the PROOF master and the workers after detecting they've
> been idle for a certain amount of time anyway, but it is a good habit
> to stop it for yourself when you're finished using it, so that you are
> immediately freeing resources and let them be available for other
> users.

In case of a major PROOF failure (i.e., crash), you can simply restart
your personal PROOF cluster by running:

    vafctl --start

PoD will stop and restart the PROOF master. You'll need to request the
workers again at this point.
