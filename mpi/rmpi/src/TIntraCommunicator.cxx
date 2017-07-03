#include <Mpi/TIntraCommunicator.h>
#include <Mpi/TInterCommunicator.h>
#include <Mpi/TInfo.h>
#include <Mpi/TPort.h>

using namespace ROOT::Mpi;

//______________________________________________________________________________
TIntraCommunicator::TIntraCommunicator(const MPI_Comm &comm) : TCommunicator(comm)
{
}

//______________________________________________________________________________
TIntraCommunicator::TIntraCommunicator(const TIntraCommunicator &comm) : TCommunicator(comm.fComm)
{
}

//______________________________________________________________________________
/**
 * Duplicates an existing communicator with all its cached information.
 * Duplicates  the  existing communicator comm with associated key values.
 * For each key value, the respective copy callback function determines the
 * attribute value associated with this key in the new communicator; one
 * particular action that a copy callback may take is to delete the attribute
 * from the  new communicator.
 * \return Returns a new communicator with the same group, any copied cached
 * information, but a new context (see Section 5.7.1 of the MPI-1 Standard,
 * "Functionality").
 *
 */
TIntraCommunicator *TIntraCommunicator::Dup() const
{
   MPI_Comm dupcomm;
   ROOT_MPI_CHECK_CALL(MPI_Comm_dup, (fComm, &dupcomm), this);
   auto fDupComm = new TIntraCommunicator(dupcomm);
   return fDupComm;
}


//______________________________________________________________________________
/**
 * This  method creates a new communicator with communication group defined by
 * group and a new context.
 * The function sets the returned communicator to a new communicator that spans
 * all the processes that are in the group.  It sets the returned communicator
 * to ROOT::Mpi::COMM_NULL for processes that are not in the group.
 * Each process must call with a group argument that is a subgroup of the group
 * associated with comm; this could be ROOT::Mpi::GROUP_EMPTY. The  processes
 * may  specify different  values  for  the  group  argument.
 * If a process calls with a non-empty group, then all processes in that group
 * must call the function with the same group as argument, that is: the same
 * processes in the same order. Otherwise the call is erroneous.
 * \param group Group, which is a subset of the group of comm (handle).
 * \return New communicator (handle).
 */
TIntraCommunicator TIntraCommunicator::Create(const TGroup &group) const
{
   MPI_Comm ncomm;
   ROOT_MPI_CHECK_GROUP(group, this);
   ROOT_MPI_CHECK_CALL(MPI_Comm_create, (fComm, group, &ncomm), this);
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}

//______________________________________________________________________________
/**
 * This function partitions the group associated with comm into disjoint
 * subgroups, one for each value of color.
 * Each subgroup contains all processes of the same color.
 * Within each subgroup, the processes are ranked in the order defined by the
 * value of the argument key, with ties broken according to their rank  in  the
 * old  group.
 * A new communicator is created for each subgroup and returned in newcomm.
 * A process may supply the color value ROOT::Mpi::UNDEFINED, in which case
 * newcomm returns ROOT::Mpi::COMM_NULL. This is a collective call, but each
 * process is permitted to provide different values for color and key.
 *
 * When you call ROOT::Mpi::TInterCommunicator::Split on an inter-communicator,
 * the processes on the left with the same color as those on the right combine
 * to create a new  inter-communicator.
 * The  key argument describes the relative rank of processes on each side of
 * the inter-communicator.
 * The function returns ROOT::Mpi::COMM_NULL for  those colors that are
 * specified on only one side of the inter-communicator, or for those that
 * specify ROOT::Mpi::UNDEFINED as the color.
 *
 * A call to ROOT::Mpi::TIntraCommunicator::Create is equivalent to a call to
 * ROOT::Mpi::TIntraCommunicator::Split( color, key), where all members of group
 * provide color =  0  and  key = rank in group, and all processes that are not
 * members of group provide color = ROOT::Mpi::UNDEFINED.
 * The function ROOT::Mpi::TIntraCommunicator::Split allows more general
 * partitioning of a group into one or more subgroups with optional reordering.
 * The value of color must be nonnegative or ROOT::Mpi::UNDEFINED.
 * \param color Control of subset assignment (nonnegative integer).
 * \param key Control of rank assignment (integer).
 * \return New communicator (handle).
 */
TIntraCommunicator TIntraCommunicator::Split(Int_t color, Int_t key) const
{
   MPI_Comm ncomm;
   ROOT_MPI_CHECK_CALL(MPI_Comm_split, (fComm, color, key, &ncomm), this);
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}

//______________________________________________________________________________
/**
 * This  call  creates  an  intercommunicator.  It is collective over the union
 * of the local and remote groups. Processes should provide identical local_comm
 * and local_leader arguments within each group. Wildcards are not permitted for
 * remote_leader, local_leader, and tag.
 *
 * This call uses point-to-point communication with communicator peer_comm, and
 * with tag tag between the leaders. Thus, care must be taken that there be no
 * pending communication on peer_comm that could interfere with this
 * communication.
 *
 * If  multiple  ROOT::Mpi::TInterCommunicator::Create  are  being made, they
 * should use different tags (more precisely, they should ensure that the local
 * and remote leaders are using different tags for each
 * ROOT::Mpi::TInterCommunicator::Create).
 * \param local_leader Rank of local group leader in local communicator
 * (integer).
 * \param peer_comm "Peer" communicator; significant only at the local_leader
 * (handle).
 * \param remote_leader Rank of remote group leader in peer_comm; significant
 * only at the local_leader (integer).
 * \param tag Message tag used to identify new intercommunicator (integer).
 * \return  Created intercommunicator (handle).
 */
TInterCommunicator TIntraCommunicator::CreateIntercomm(Int_t local_leader, const TIntraCommunicator &peer_comm,
                                                       Int_t remote_leader, Int_t tag) const
{
   MPI_Comm ncomm;
   ROOT_MPI_CHECK_CALL(MPI_Intercomm_create, (fComm, local_leader, peer_comm.fComm, remote_leader, tag, &ncomm), this);
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}

//______________________________________________________________________________
/**
 * Establishes communication with a client. It is collective over the calling
 *communicator. It returns an intercommunicator that allows communication with
 *the client, after the client has connected with the
 *ROOT::Mpi:TIntraCommunicator::Accept function using the
 *ROOT::Mpi:TIntraCommunicator::Connect function.
 *\param port ROOT::Mpi::TPort object with the port configuration
 *\param root Rank in communicator of root node (integer).
 *\return ROOT::Mpi:TInterCommunicator object with client as remote group
 *(handle).
 */
TInterCommunicator TIntraCommunicator::Accept(const TPort &port, Int_t root) const
{
   MPI_Comm ncomm;
   ROOT_MPI_CHECK_CALL(MPI_Comm_accept, (port.GetPortName(), port.GetInfo(), root, fComm, &ncomm), this);
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}

//______________________________________________________________________________
/**
 * Establishes communication with a server specified by port.
 * It is collective over the calling communicator and returns an
 * intercommunicator in which the remote group participated in an
 * ROOT::Mpi:TIntraCommunicator::Accept.
 * The ROOT::Mpi:TIntraCommunicator::Connect call must only be called after the
 * ROOT::Mpi:TIntraCommunicator::Accept call has  been  made by the MPI job
 * acting as the server.
 *
 * If the named port does not exist (or has been closed),
 * ROOT::Mpi:TIntraCommunicator::Connect raises an error of class
 * ROOT::Mpi::ERR_PORT.
 *
 * MPI  provides no guarantee of fairness in servicing connection attempts. That
 * is, connection attempts are not necessarily satisfied in the order in which
 * they were initiated, and competition from other connection attempts may
 * prevent a particular connection attempt from being satisfied.
 *
 * The port parameter is the address of the server. It must be the same as the
 * name returned by ROOT::Mpi::TPort on the server.
 * \param port ROOT::Mpi::TPort object with the port configuration
 * \param root Rank in communicator of root node (integer).
 * \return ROOT::Mpi:TInterCommunicator object with client as remote group
 * (handle)
 */
TInterCommunicator TIntraCommunicator::Connect(const TPort &port, Int_t root) const
{
   MPI_Comm ncomm;
   ROOT_MPI_CHECK_CALL(MPI_Comm_connect, (port.GetPortName(), port.GetInfo(), root, fComm, &ncomm), this);
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}

//______________________________________________________________________________
/**
 *
 */
TInterCommunicator TIntraCommunicator::Spawn(const Char_t *command, const Char_t *argv[], Int_t maxprocs,
                                             const TInfo &info, Int_t root) const
{
   MPI_Comm ncomm;
   ROOT_MPI_CHECK_CALL(
      MPI_Comm_spawn,
      (command, const_cast<Char_t **>(argv), maxprocs, info, root, fComm, &ncomm, (Int_t *)MPI_ERRCODES_IGNORE), this);
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}

//______________________________________________________________________________
/**
 * ROOT::Mpi:TInterCommunicator::Spawn tries to start maxprocs identical copies
 of the MPI program specified by command, establishing communication with them
 and returning an intercommunicator. The spawned processes are referred to as
 children.
 * The children have their own ROOT::Mpi:COMM_WORLD, which is  separate  from
 that  of  the  parents.
 * ROOT::Mpi:TInterCommunicator::Spawn is  collective  over communicator, and
 also may not return until ROOT::Mpi::TEnvironment object has been created in
 the children. Similarly, ROOT::Mpi::TEnvironment object in the children may not
 return until all parents have called ROOT::Mpi::TIntraCommunicator::Spawn. In
 this sense, ROOT::Mpi::TIntraCommunicator::Spawn in the parents and MPI_Init in
 the children  form  a  collective  operation over  the union of parent and
 child processes.
 * The intercommunicator returned by ROOT::Mpi::TIntraCommunicator::Spawn
 contains the parent processes in the local group and the child  processes in
 the remote group.
 * The ordering of processes in the local and remote groups is the same as the
 as the ordering of the group of  the  comm  in  the parents and of
 ROOT::Mpi::COMM_WORLD of the children, respectively. This intercommunicator can
 be obtained in the children through the function
 ROOT::Mpi::TCommunicator::GetParent.
 *
 * The MPI standard allows an implementation to use the ROOT::Mpi::UNIVERSE_SIZE
 attribute of ROOT::Mpi::COMM_WORLD to specify the number of processes that will
 be active in a program.  Although this implementation of the MPI standard
 defines ROOT::Mpi::UNIVERSE_SIZE, it does not allow the user to set its value.
 If  you  try  to  set  the value of ROOT::Mpi::UNIVERSE_SIZE, you will get an
 error message.
 *
 * <b>The command Argument</b>
 *
 * The  command  argument is a string containing the name of a program to be
 spawned. The string is null-terminated in C.
 * MPI looks for the file first in the working directory of the spawning
 process.
 *
 * <b>The argv Argument</b>
 *
 * argv is an array of strings containing arguments that are passed to the
 program. The first element of argv is the first argument passed to command,
 not, as is conventional  in  some  contexts, the command itself. The argument
 list is terminated by NULL in C and C++ (note that it is the MPI application's
 responsibility to ensure that the last entry of the argv array is an empty
 string; the compiler will not automatically insert it).
 * The constant ROOT::Mpi::ARGV_NULL may be used in C, C++ to indicate an empty
 argument list. In C and C++, this constant is the same as NULL.
 *
 * In C, the ROOT::Mpi::TIntraCommunicator::Spawn argument argv differs from the
 argv argument of main in two respects. First, it is shifted by one element.
 Specifically,  argv[0]  of main   contains  the  name  of  the  program  (given
 by  command).  argv[1]  of  main corresponds to argv[0] in
 ROOT::Mpi::TIntraCommunicator::Spawn, argv[2] of main to argv[1] of
 ROOT::Mpi::TIntraCommunicator::Spawn, and so on. Second, argv of
 ROOT::Mpi::TIntraCommunicator::Spawn must be null-terminated, so that its
 length can be determined. Passing an argv of  ROOT::Mpi::ARGV_NULL  to
 ROOT::Mpi::TIntraCommunicator::Spawn results in main receiving argc of 1 and an
 argv whose element 0 is the name of the program.
 *
 * <b>The maxprocs Argument</b>
 *
 * MPI  tries  to spawn maxprocs processes. If it is unable to spawn maxprocs
 processes, it raises an error of class ROOT::Mpi::ERR_SPAWN. If MPI is able to
 spawn the specified number of processes, ROOT::Mpi::TIntraCommunicator::Spawn
 returns successfully and the number of spawned processes, m, is given by the
 size of  the  remote  group  of intercomm.
 *
 * A spawn call with the default behavior is called hard. A spawn call for which
 fewer than maxprocs processes may be returned is called soft.
 *
 * <b>The info Argument</b>
 *
 * The  info  argument  is  an  opaque  handle  of  type MPI_Info in C,
 MPI::Info in C++ and INTEGER in Fortran. It is a container for a number of
 user-specified (key,value) pairs. key and value are strings (null-terminated
 char* in C). Routines to create and manipulate the info  argument  are
 described in Section 4.10 of the MPI-2 standard.(or ROOT::Mpi::TInfo)
 *
 * For  the  SPAWN calls, info provides additional, implementation-dependent
 instructions to MPI and the runtime system on how to start processes. An
 application may pass ROOT::Mpi::INFO_NULL. Portable programs not requiring
 detailed control over process locations should use ROOT::Mpi::INFO_NULL.
 *
 * The following keys for info are recognized in Open MPI. (The reserved values
 mentioned in Section 5.3.4 of the MPI-2 standard are not implemented.)
 *
 <table>
 <tr><th>Key</th><th>Type</th><th>Description</th></tr>
 <tr><td>host </td><td>char *</td><td>Host on which the process should be
 spawned.</td> </tr>
 <tr><td> hostfile </td><td> char * </td><td> Hostfile containing the hosts on
 which
 the processes are to be spawned. </td> </tr>
 <tr><td> add-host </td><td> char * </td><td> Add the specified host to the list
 of
 hosts known to this job and use it for the associated process. This will be
 used similarly to the -host option.</td> </tr>
 <tr><td> add-hostfile </td><td> char* </td><td> Hostfile containing hosts to be
 added to
 the list of hosts known to this job and use it for the associated process. This
 will be used similarly to the -hostfile option. </td> </tr>
 <tr><td> wdir  </td><td> char *</td><td> Directory where the executable is
 located. If files are to be pre-positioned, then this location is the desired
 working directory at time of execution - if not specified, then it will
 automatically be set to ompi_preload_files_dest_dir. </td> </tr>
 <tr><td> ompi_prefix </td><td> char * </td><td> Same as the --prefix command
 line
 argument to mpirun \see rootmpi command tool. </td> </tr>
 <tr><td> ompi_preload_binary  </td><td> bool </td><td> If set to true,
 pre-position the
 specified executable onto the remote host. A destination directory must also be
 provided.</td> </tr>
 <tr><td> ompi_preload_files  </td><td> char * </td><td> A comma-separated list
 of files that are
 to be pre-positioned in addition to the executable.  Note that this option does
 not depend upon ompi_preload_binary - files can be moved to the target even if
 an executable is not moved.</td> </tr>
 <tr><td> ompi_stdin_target      </td><td> char * </td><td> Comma-delimited list
 of ranks to
 receive stdin when forwarded.</td> </tr>
 <tr><td> ompi_non_mpi  </td><td> bool </td><td> If set to true, launching a
 non-MPI
 application; the returned communicator will be MPI_COMM_NULL. Failure to set
 this flag when launching a non-MPI application will cause both the child and
 parent jobs to "hang".</td> </tr>
 <tr><td> ompi_param </td><td> char * </td><td> Pass an OMPI MCA parameter to
 the child
 job.  If that parameter already  exists in  the environment, the value will be
 overwritten by the provided value.</td> </tr>
 <tr><td> mapper </td><td> char * </td><td> Mapper to be used for this job.
 </td> </tr>
 <tr><td> map_by </td><td> char * </td><td> Mapping directive indicating how
 processes are to be mapped (slot,node, socket, etc.). </td> </tr>
 <tr><td> rank_by </td><td> char * </td><td> Ranking directive indicating how
 processes are to be ranked (slot,node, socket, etc.). </td> </tr>
 <tr><td> bind_to   </td><td> char * </td><td> Binding directive indicating how
 processes  are to be bound (core, slot, node, socket, etc.). </td> </tr>
 <tr><td> path </td><td> char * </td><td> List of directories to search for the
 executable </td> </tr>
 <tr><td> npernode </td><td> char * </td><td> Number of processes to spawn on
 each node  of the allocation </td> </tr>
 <tr><td> pernode </td><td> bool </td><td> Equivalent to npernode of 1 </td>
 </tr>
 <tr><td> ppr </td><td> char * </td><td> Spawn specified number of processes on
 each  of the identified object type </td> </tr>
 <tr><td> env </td><td> char * </td><td> Newline-delimited list of envars to be
 passed to the spawned procs </td> </tr>
 </table>
 *
 * bool info keys are actually strings but are evaluated as follows: if the
 string value is a number, it is converted to an integer and cast to a boolean
 (meaning  that  zero  integers  are  false  and non-zero values are true).  If
 the string value is (case-insensitive) "yes" or "true", the boolean is true.
 If the string value is (case-insensitive) "no" or "false", the boolean is
 false.  All other string values are unrecognized, and therefore false.
 *
 * <b>The root Argument</b>
 * All arguments before the root argument are examined only on the process whose
 rank in comm is equal to root. The value of these arguments on  other
 processes is ignored.
 * <b>The array_of_errcodes Argument</b>
 *
 * The  array_of_errcodes  is  an array of length maxprocs in which MPI reports
 the status of the processes that MPI was requested to start. If all maxprocs
 processes were spawned, array_of_errcodes is filled in with the value
 ROOT::Mpi::SUCCESS.
 * If any of the processes are not spawned, array_of_errcodes is  filled  in
 with the  value  ROOT::Mpi::ERR_SPAWN.  An application may pass
 ROOT::Mpi::ERRCODES_IGNORE if it is not interested in the error codes.
 * \param command Name of program to be spawned (string, significant only at
 root).
 * \param argv Arguments to command (array of strings, significant only at
 root).
 * \param maxprocs Maximum number of processes to start (integer, significant
 only at root).
 * \param info A set of key-value pairs telling the runtime system where and how
 to start the processes (handle, significant only at root).
 * \param root Rank of process in which previous arguments are examined
 (integer).
 * \param array_of_errcodes One code per process (array of integers).
 * \return ROOT::Mpi::TInterCommunicator between original group and the newly
 spawned group (handle).
 */
TInterCommunicator TIntraCommunicator::Spawn(const Char_t *command, const Char_t *argv[], Int_t maxprocs,
                                             const TInfo &info, Int_t root, Int_t array_of_errcodes[]) const
{
   MPI_Comm ncomm;
   ROOT_MPI_CHECK_CALL(MPI_Comm_spawn,
                       (command, const_cast<Char_t **>(argv), maxprocs, info, root, fComm, &ncomm, array_of_errcodes),
                       this);
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}

//______________________________________________________________________________
/**
 * Similar to ROOT::Mpi::TIntraCommunicator::Spawn but spawning multiple process
 * at same time. \see ROOT::Mpi::TIntraCommunicator::Spawn
 * \param count Number of commands (positive integer, significant to MPI only at
 * root -- see NOTES).
 * \param array_of_commands Programs to be executed (array of strings,
 * significant only at root).
 * \param array_of_argv Arguments for commands (array of array of strings,
 * significant only at root).
 * \param array_of_maxprocs Maximum number of processes to start for each
 * command (array of integers, significant only at root).
 * \param array_of_info Info objects telling the runtime system where and how to
 * start processes (array of handles, significant only at root).
 * \param root Rank of process in which previous arguments are examined
 * (integer).
 * \return ROOT::Mpi::TInterCommunicator between original group and the newly
 * spawned group (handle).
 */
TInterCommunicator TIntraCommunicator::SpawnMultiple(Int_t count, const Char_t *array_of_commands[],
                                                     const Char_t **array_of_argv[], const Int_t array_of_maxprocs[],
                                                     const TInfo array_of_info[], Int_t root)
{
   MPI_Comm ncomm;

   MPI_Info *array_of_mpi_info = new MPI_Info[count];
   for (Int_t i = 0; i < count; i++) {
      array_of_mpi_info[i] = array_of_info[i];
   }

   ROOT_MPI_CHECK_CALL(MPI_Comm_spawn_multiple,
                       (count, const_cast<Char_t **>(array_of_commands), const_cast<Char_t ***>(array_of_argv),
                        const_cast<Int_t *>(array_of_maxprocs), array_of_mpi_info, root, fComm, &ncomm,
                        (Int_t *)MPI_ERRCODES_IGNORE),
                       this);
   delete[] array_of_mpi_info;
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}

//______________________________________________________________________________
/**
 * Similar to ROOT::Mpi::TIntraCommunicator::Spawn but spawning multiple process
 * at same time. \see ROOT::Mpi::TIntraCommunicator::Spawn
 * \param count Number of commands (positive integer, significant to MPI only at
 * root -- see NOTES).
 * \param array_of_commands Programs to be executed (array of strings,
 * significant only at root).
 * \param array_of_argv Arguments for commands (array of array of strings,
 * significant only at root).
 * \param array_of_maxprocs Maximum number of processes to start for each
 * command (array of integers, significant only at root).
 * \param array_of_info Info objects telling the runtime system where and how to
 * start processes (array of handles, significant only at root).
 * \param array_of_errcodes One code per process (array of integers).
 * \param root Rank of process in which previous arguments are examined
 * (integer).
 * \return ROOT::Mpi::TInterCommunicator between original group and the newly
 * spawned group (handle).
 */
TInterCommunicator TIntraCommunicator::SpawnMultiple(Int_t count, const Char_t *array_of_commands[],
                                                     const Char_t **array_of_argv[], const Int_t array_of_maxprocs[],
                                                     const TInfo array_of_info[], Int_t root, Int_t array_of_errcodes[])
{
   MPI_Comm ncomm;

   MPI_Info *array_of_mpi_info = new MPI_Info[count];
   for (Int_t i = 0; i < count; i++) {
      array_of_mpi_info[i] = array_of_info[i];
   }

   ROOT_MPI_CHECK_CALL(MPI_Comm_spawn_multiple,
                       (count, const_cast<Char_t **>(array_of_commands), const_cast<Char_t ***>(array_of_argv),
                        const_cast<Int_t *>(array_of_maxprocs), array_of_mpi_info, root, fComm, &ncomm,
                        array_of_errcodes),
                       this);
   delete[] array_of_mpi_info;
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}
