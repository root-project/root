Deploying the Virtual Analysis Facility
=======================================

Introduction
------------

Thanks to CernVM and PROOF on Demand, it is possible to deploy a ready
to use Virtual Analysis Facility on your cloud (either public, private
or even your desktop computer).

On the server side, "configuring" the Virtual Analysis Facility is
simply a matter of starting a certain number of CernVM virtual machines
that will become part of your PROOF cluster. CernVM uses
contextualization to specialize each virtual machine to be either a head
node or a worker node.

The Virtual Analysis Facility comes with many preconfigured things:

-   a HTCondor cluster capable of running PROOF on Demand

-   certificate authentication

-   your experiment's software (if available on CernVM-FS)

Obtain the CernVM image and contextualization
---------------------------------------------

### Download the CernVM bare image

The Virtual Analysis Facility currently works with *CernVM Batch 2.7.1
64-bit*. This means that you need to have this CernVM image available
either on your local hard disk (in case of a desktop deployment) or in
your cloud's image repository.

> For convenience we provide the direct link for the working versions:
>
> -   [CernVM 2.7.1 batch 64-bit for
>     **KVM**](https://cernvm.cern.ch/releases/19/cernvm-batch-node-2.7.1-2-3-x86_64.hdd.gz)
>
> -   [CernVM 2.7.1 batch 64-bit for
>     **Xen**](https://cernvm.cern.ch/releases/19/cernvm-batch-node-2.7.1-2-3-x86_64.ext3.gz)
>
> Images are gzipped. In most cases you'll need to gunzip them before
> registering to your image repository.

### Create VM configuration profiles

CernVM images are base images supporting boot-time customization via
configuration profiles called "contexts". Context creation can be
performed through the [CernVM Online](https://cernvm-online.cern.ch/)
website. The site is immediately accessible if you have a CERN account.

Go to your [CernVM Online
Dashboard](https://cernvm-online.cern.ch/dashboard), click on the
**Create new context...** dropdown and select **Virtual Analysis Facility
node**.

There's only a few parameters to configure.

Context name
:   A name for your context (such as *VAF Master for ATLAS*). Any name
    will work.

Role
:   Use this to configure either a *master* or a *slave*.

VAF master (only available when configuring a slave)
:   IP address or FQDN of the Virtual Analysis Facility master.

Auth method
:   Choose between *ALICE LDAP* (useful only for ALICE users) or *Pool
    accounts* (good for authenticating all the other Grid users).

Num. pool accounts (only available when using pool accounts auth)
:   Number of pool accounts to create.

Proxy for CVMFS
:   An URL specifying the proxy server for CernVM-FS, such as
    `http://ca-proxy.cern.ch:3128/`. If you leave it empty, proxy will
    be automatically discovered.

HTCondor shared secret
:   VMs part of the same cluster should have the same value of this
    field. It is used to mutually authenticate VMs and it is used like a
    password.

Context password
:   Current profile will be saved on the [CernVM Online
    repository](http://cernvm-online.cern.ch/). If you don't want the
    information there to be publicly available to other users, type in
    a value for protecting the context with an encryption password.

You will have to create a profile for the **master** and the **slave**. Since
most of the configuration variables are the same (like the *HTCondor
shared secret*) you can create one, clone it and change only what's
needed to change.

Deploy it on the cloud
----------------------

Provided you have access to a certain cloud API, you'll need to
instantiate a certain number of CernVM batch images with proper
contextualization: one for the master, as many as you want as slaves.

CernVM supports contextualization through the "user data" field
supported by all cloud infrastructures.

Each cloud infrastructure has a different method of setting the "user
data". The following description will focus on:

-   [OpenNebula](http://opennebula.org/)

-   OpenStack (such as the [CERN Agile
    infrastructure](https://openstack.cern.ch/))

-   [Amazon EC2](http://aws.amazon.com/ec2/)-compatible interfaces via
    the open [Eucalyptus](http://www.eucalyptus.com/)
    [Euca2ools](http://www.eucalyptus.com/eucalyptus-cloud/tools): many popular
    clouds support such interface and tools

### Download the CernVM Online contextualizations

Go to the CernVM Online Dashboard page where you have previously
customized the contexts for your master and your slaves.

Click on the rightmost button on the line of the desired context and
select **Get rendered context** from the dropdown: save the output to a
text file (such as `my_vaf_context.txt`, the name we will use in the
examples that follow). This file will be subsequently passed as the so
called "user-data" file to the cloud API.

> Repeat the operation for both the master context and the slave
> context.

### OpenStack API: nova

Example of a CernVM instantiation using `nova`:

``` {.bash}
nova boot \
  --flavor m1.xlarge \
  --image cernvm-batch-node-2.6.0-4-1-x86_64 \
  --key-name my_default_keyparir \
  --user-data my_vaf_context.txt \
  Name-Of-My-New-VM
```

The `--user-data` option requires the context file we've just
downloaded.

### EC2 API: euca-tools

Example of a CernVM instantiation using `euca-tools`:

``` {.bash}
euca-run-instances \
  --instance-type m1.xlarge \
  --key my_default_keyparir \
  --user-data-file my_vaf_context.txt \
  cernvm-batch-node-2.6.0-4-1-x86_64
```

The `--user-data-file` option is the context file we've just downloaded.

### OpenNebula

An example VM definition follows:

``` {.ruby}
CONTEXT=[
  EC2_USER_DATA="<base64_encoded_string>",
]
CPU="6"
VCPU="6"
DISK=[
  IMAGE="cernvm-batch-node-2.6.0-4-1-x86_64",
  TARGET="vda" ]
MEMORY="16000"
NAME="CernVM-VAF-Node"
NIC=[
  NETWORK="My-OpenNebula-VNet" ]
OS=[
  ARCH="x86_64" ]
```

The `<base64_encoded_string>` requires the base64 version of the whole
downloaded context definition. You can obtain it by running:

    cat my_vaf_context.txt | base64 | tr -d '\n'

Network security groups
-----------------------

In order to make the Virtual Analysis Facility work properly, the
firewall of your infrastructure must be configured to allow some
connections.

Some ports need to allow "external" connections while other ports might
be safely opened to allow only connections from other nodes of the
Virtual Analysis Facility.

### Ports to open on all nodes

HTCondor ports
:   Allow **TCP and UDP range 9600-9700** only between nodes of the Virtual
    Analysis Facility.

Only HTCondor and PoD communication is needed between the nodes. No HTCondor
ports need to be opened to the world.

### Additional ports to open on the front end node

HTTPS
:   Allow **TCP 443** from all

SSH
:   Allow **TCP 22** from all

No other ports need to be opened from the outside. Your definition of
*allow from all* might vary.
