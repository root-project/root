# ROOT Mpi
\index{ROOT Mpi}
![ROOT MPI](pictures/rmpi1.png)


## Description
MPI is an standard of communication for HPC/Grid computing widely used in the scientific community. The idea is to do parallel computing with message passing in
a standart way.
This standart has several implementations developed by the academia of computer companies like [OpenMPI](https://www.open-mpi.org/), [MPICH](https://www.mpich.org/), [IBM](https://www.ibm.com/developerworks/downloads/im/mpi/index.html), [Intel](https://software.intel.com/en-us/intel-mpi-library), [Microsft](https://msdn.microsoft.com/en-us/library/bb524831(v=vs.85).aspx) etc..
It supports RMA (Remote Memory Access), shared memory,P2P(peer to perr) and collective communication and Checkpoint and Fault tolerance under develoment http://mpi-forum.org/mpi-40

ROOT Mpi is the integration of MPI and ROOT technologies in a framework for parallel computing. The <b>motivation</b> was to communicate ROOT objects through processes using serialization, to implement MPI with a better design for ROOT and create an interface that uses the new C++ features to write parallel code.
But the most important is to implement ROOT algorithms in parallel for HPC/Grid systems.


## Installation
To install ROOT MPI please read first.

- [https://root.cern.ch/building-root](https://root.cern.ch/building-root)
- [https://root.cern.ch/build-prerequisites](https://root.cern.ch/build-prerequisites)


### Compiling ROOT MPI:

**Prerequisities**
Install [OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/)

To compile ROOT MPI lets to create a compilation directory and to activate it use cmake -Dmpi=ON ..

``` {.sh}
mkdir compile
cd compile
cmake -Dmpi=ON ..
make -j 5
```

## ROOT Mpi Features

Feature       | Description                                                      | Status
 ------------ | :--------------------------------------------------------------- | ----- 
TCommunicator | Base class for all communcators, to perform p2p and collective   |<span style="color:green">**DONE**</span>
TIntraComminicator|Class to do communication in the same context                                                                                   |<span style="color:green">**DONE**</span>
TInterCommunicator|Class to perform communication between  different contexts.   |<span style="color:green">**DONE**</span>
TReuqest, TPrequest and TGrequest|Classes to handle non-blocking communication.|<span style="color:green">**DONE**</span>
TEnvironment | Class initialize/finalize communication system.                                                                                     |<span style="color:green">**DONE**</span>
TMpiMessage and TMpiMessageInfo|Classes for serialization in the communication                                                                     |<span style="color:green">**DONE**</span>
Error Handling                   | MPI errors are very hard to understant also for MPI experts, with the class TErrorHandler we improve the error handling with more beautiful and informative messages .|<span style="color:green">**DONE**</span>
TMpiFile and TMpiFileMerger| Classes manage distribute files.        <span style="color:blue"> **(SPECIAL CASE)**</span> see important notes below.|<span style="color:green">**DONE**</span>
TCartCommunicator|Cartesian Communicator (map processes in a mesh)                                                                                 |<span style="color:red">**TODO**</span>
TGraphComminicator|Graph communicator (map processes in a general graph topology)|<span style="color:red">**TODO**</span>
Checkpointing| Feature to save the state of the execution and restore it.                                                                          |<span style="color:red">**TODO**</span>
Fault Tolerance | Feature to continue execution in case of fault.                                                                                  |<span style="color:red">**TODO**</span>
Memory Window|Class to shared regions of memory with Remote Access Memory using one site communication|<span style="color:red">**NOT SUPPORTED**</span>



<span style="color:red">**Important NOTES**</span>

  * The message in the communication can be any serializable object supported by ROOT or raw datatypes.
  * Persistent communication is not supported because is not possible to know previously the size of the object in the serialization. (It will be implemented in the future for raw datatypes)
  * Fault tolerance and checkpoint needs the new standart that is not implemented yet in the mpi backends.
  * The class TMpiFile is not using the raw MPI C functions, instead a new design was created that is using TMemFile and TFile from ROOT.
  * Memory window is not supported becuase is not possible to know the size of the serialized object in both processes to get access to a memory region.


## rootmpi (command line tool)
Normally the mpi binaries are execute with the command mpirun or mpiexec, that lets to launch the binary code starting the communication system, but in ROOTMpi you can execute ROOT macros without compile it like is usual to run ROOT code.
rootmpi(executable) is a command line tool that helps you run binary code, root macros or compiling source code to generate binary code.
Internally to run a macro it launch mpirun with root interpreter, that allows to run the multiple processes in a distrubute system interpreting the code.

<center>
![Execution Diagram](pictures/rmpi2.png){:height="30%" width="90%"}
</center>

<b>Usage for Macro:</b> rootmpi (mpirun options) (root/cling options) (macro file.C )

<b>Options:</b>

 * --help-mpirun show mpi options for execution 

### Some mpirun options for (OpenMPI) :

 * -H, -host, --host (host1,host2,...,hostN) :List of hosts on which to invoke processes.
 * -machinefile,-hostfile file:File with hosts (see machine file example below)
 * -path (path) : (path) that will be used when attempting to locate the requested executables.
 * -np #: Run this many copies of the program on the given nodes.(number of process to use)

You can find full list of option for OpenMpi or Mpich at:

 * [https://www.open-mpi.org/doc/v2.1/man1/mpirun.1.php](https://www.open-mpi.org/doc/v2.1/man1/mpirun.1.php)
 * [http://www.mpich.org/static/docs/latest/www1/mpiexec.html](http://www.mpich.org/static/docs/latest/www1/mpiexec.html)

### Example of machine file

``` {.sh}
# This is an example hostfile.
# (taken from https://www.open-mpi.org/faq/?category=running#mpirun-hostfile)
# Comments begin with #
#
# The following node is a single processor machine:
node.example.com

# The following node is a dual-processor machine:
node.example.com slots=2

# The following node is a quad-processor machine, and we absolutely
# want to disallow over-subscribing it:
node.example.com slots=4 max-slots=4
```

### Options Cling/ROOT:

 * -b : run in batch mode without graphics
 * -n : do not execute logon and logoff macros as specified in .rootrc
 * -q : exit after processing command line macro files
 * -l : do not show splash screen
 * -x : exit on exception
 * -memstat : run with memory usage monitoring

## Hello world
This is a basic example that just print the host name and the rank id.
save it like hello.C
``` {.cpp}
using namespace ROOT::Mpi;
void hello()
{
   TEnvironment env; // environment to start communication system
   std::cout << "Hello from process " << COMM_WORLD.GetRank() 
             << " of " << COMM_WORLD.GetSize() << " in host "
             << env.GetProcessorName() << std::endl;
}
```

To execute it just use rootmpi command line tools
``` {.sh}
rootmpi -H wn0,wn1 -np 2 tutorials/mpi/hello.C
```
the output mus be something like 
``` {.sh}
```


## ROOT Mpi Basics
ROOT Mpi can to communicate processes between multiple nodes using messages,
according to the different schemas the communiction, peer to peer or collectives
like gather, scatter, broadcast etc..
The data is encapsulated internally using TMpiMessage objects and the communication can be blocking(that blocks the execution while the message comes)
or non-blocking(that can to receive the messages the messages asonchronously without blocks the execution).

Every message have a tag id that allow to send or receive to the same process multiple messages and identify them with the id. if you id is wrong in the communication your communication can be blocked forever because you message never will be delivered, or you message will have wrong datatype producing an error handle by ROOTMpi in the execution time.

All the communication is made in the same context of communication using the 
the TIntraCommunicator object ROOT::Mpi::COMM_WORLD, this global object
allows to use the multiple schemas of communication throw its methods.

### Blocking communictaion
In the blocking communication the execution is "synchronous" and it stop while the message is received,if the tag id is wrong in the message the execution will be blocked forever.

#### Peer to Peer communication
Peer to peer communication is the most basic communication operation,



#### Collective communication

#### Broadcast

#### Gather

