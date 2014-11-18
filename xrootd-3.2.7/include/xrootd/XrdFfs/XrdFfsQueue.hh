/******************************************************************************/
/* XrdFfsQueue.hh  functions to run independent tasks in queue                */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/* Author: Wei Yang (SLAC National Accelerator Laboratory, 2009)              */
/*         Contract DE-AC02-76-SFO0515 with the Department of Energy          */
/******************************************************************************/

#ifdef __cplusplus
  extern "C" {
#endif

#include <stdlib.h>
#include <pthread.h>

struct XrdFfsQueueTasks {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    short done;
    void* (*func)(void*);
    void **args;

    unsigned int id;
    struct XrdFfsQueueTasks *next;
    struct XrdFfsQueueTasks *prev;
};

struct XrdFfsQueueTasks* XrdFfsQueue_create_task(void* (*func)(void*), void **args, short initstat);
void XrdFfsQueue_free_task(struct XrdFfsQueueTasks *task);
void XrdFfsQueue_wait_task(struct XrdFfsQueueTasks *task);
unsigned int XrdFfsQueue_count_tasks();

int XrdFfsQueue_create_workers(int n);
int XrdFfsQueue_remove_workers(int n);
int XrdFfsQueue_count_workers();
 
#ifdef __cplusplus
  }
#endif

