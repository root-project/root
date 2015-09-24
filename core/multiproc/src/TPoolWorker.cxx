#include "TPoolWorker.h"

//////////////////////////////////////////////////////////////////////////
///
/// \class TPoolWorker
///
/// This class works together with TPool to allow the execution of
/// functions in server processes. Depending on the exact task that the
/// worker is required to execute, a different version of the class
/// can be called.
///
/// ### TPoolWorker<F, T, R>
/// The most general case, used by TPool::MapReduce(F func, T& args, R redfunc).
/// This worker is build with:
/// * a function of signature F (the one to be executed)
/// * a collection of arguments of type T on which to apply the function
/// * a reduce function with signature R to be used to squash many
/// returned values together.
///
/// ### Partial specializations
/// A few partial specializations are provided for less general cases:
/// * TPoolWorker<F, T, void> handles the case of a function that takes
/// one argument and does not perform reduce operations
/// (TPool::Map(F func, T& args)).
/// * TPoolWorker<F, void, R> handles the case of a function that takes
/// no arguments, to be executed a specified amount of times, which
/// returned values are squashed together (reduced)
/// (TPool::Map(F func, unsigned nTimes, R redfunc))
/// * TPoolWorker<F, void, void> handles the case of a function that takes
/// no arguments and whose arguments are not "reduced"
/// (TPool::Map(F func, unsigned nTimes))
///
/// Since all the important data are passed to TPoolWorker at construction
/// time, the kind of messages that client and workers have to exchange
/// are usually very simple.
///
//////////////////////////////////////////////////////////////////////////
