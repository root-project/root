/******************************************************************************/
/* XrdFfsQueue.cc  functions to run independent tasks in queue                */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/* Author: Wei Yang (SLAC National Accelerator Laboratory, 2009)              */
/*         Contract DE-AC02-76-SFO0515 with the Department of Energy          */
/******************************************************************************/

#include "XrdFfs/XrdFfsQueue.hh"

/* queue operation */
 
#ifdef __cplusplus
  extern "C" {
#endif

struct XrdFfsQueueTasks *XrdFfsQueueTaskque_head = NULL;
struct XrdFfsQueueTasks *XrdFfsQueueTaskque_tail = NULL;
unsigned int XrdFfsQueueNext_task_id = 0;
pthread_mutex_t XrdFfsQueueTaskque_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t XrdFfsQueueTaskque_cond = PTHREAD_COND_INITIALIZER;

void XrdFfsQueue_enqueue(struct XrdFfsQueueTasks *task)
{
    pthread_mutex_lock(&XrdFfsQueueTaskque_mutex);

    task->id = XrdFfsQueueNext_task_id + 1;
    XrdFfsQueueNext_task_id = task->id;
    if (XrdFfsQueueTaskque_tail == NULL) 
    {
        XrdFfsQueueTaskque_head = task;
        XrdFfsQueueTaskque_tail = task;
        task->next = NULL;
        pthread_cond_broadcast(&XrdFfsQueueTaskque_cond);
    }
    else
    {
        task->prev = XrdFfsQueueTaskque_tail;
        task->next = NULL;
        XrdFfsQueueTaskque_tail->next = task;
        XrdFfsQueueTaskque_tail = task;
    }

    pthread_mutex_unlock(&XrdFfsQueueTaskque_mutex);
    return;
}

struct XrdFfsQueueTasks *XrdFfsQueue_dequeue()
{
    struct XrdFfsQueueTasks *head;
    while (pthread_mutex_lock(&XrdFfsQueueTaskque_mutex) == 0)
        if (XrdFfsQueueTaskque_head == NULL)
        {
            pthread_cond_wait(&XrdFfsQueueTaskque_cond, &XrdFfsQueueTaskque_mutex);
            pthread_mutex_unlock(&XrdFfsQueueTaskque_mutex);
        }
        else
            break;

    head = XrdFfsQueueTaskque_head;
    XrdFfsQueueTaskque_head = XrdFfsQueueTaskque_head->next;

    head->next = NULL;
    head->prev = NULL;        

    if (XrdFfsQueueTaskque_head == NULL)
        XrdFfsQueueTaskque_tail = NULL;

    pthread_mutex_unlock(&XrdFfsQueueTaskque_mutex);
    return head;
}

/* create, wait and free(delete) a task */

struct XrdFfsQueueTasks* XrdFfsQueue_create_task(void* (*func)(void*), void **args, short initstat)
{
    struct XrdFfsQueueTasks *task = (struct XrdFfsQueueTasks*) malloc(sizeof(struct XrdFfsQueueTasks));
    task->func = func;
    task->args = args;
    task->done = ( (initstat == -1)? -1 : 0); /* -1 means this task is meant to kill a worker thread */

    pthread_mutex_init(&task->mutex, NULL);
    pthread_cond_init(&task->cond, NULL);

    XrdFfsQueue_enqueue(task);
    return task;
}

void XrdFfsQueue_free_task(struct XrdFfsQueueTasks *task) 
{
    pthread_mutex_destroy(&task->mutex);
    pthread_cond_destroy(&task->cond);
    task->func = NULL;
    task->args = NULL;
    task->next = NULL;
    task->prev = NULL;
    free(task);
    task = NULL;
}

void XrdFfsQueue_wait_task(struct XrdFfsQueueTasks *task)
{
    pthread_mutex_lock(&task->mutex);
    if (task->done != 1)
        pthread_cond_wait(&task->cond, &task->mutex);
    pthread_mutex_unlock(&task->mutex);
}

unsigned int XrdFfsQueue_count_tasks()
{
    unsigned int que_len = 0;
    pthread_mutex_lock(&XrdFfsQueueTaskque_mutex);
    if (XrdFfsQueueTaskque_head != NULL && XrdFfsQueueTaskque_tail != NULL)
        if (XrdFfsQueueTaskque_tail->id > XrdFfsQueueTaskque_head->id)
            que_len = XrdFfsQueueTaskque_tail->id - XrdFfsQueueTaskque_head->id;
        else
            que_len = (unsigned int)2147483647 - (XrdFfsQueueTaskque_head->id - XrdFfsQueueTaskque_tail->id) + 1;
    pthread_mutex_unlock(&XrdFfsQueueTaskque_mutex);
    return que_len;
}

/* workers */

void *XrdFfsQueue_worker(void* x)
{
    struct XrdFfsQueueTasks *task;
    short quit = 0;

    loop:
    task = XrdFfsQueue_dequeue();

    if (task->done == -1) // terminate this worker thread
        quit = 1;

    pthread_mutex_lock(&task->mutex);
#ifdef QUEDEBUG
    printf("worker %d on task %d\n", wid, task->id);
#endif
    if (!quit)
        (task->func)(task->args);
 
    task->done = 1;
    pthread_cond_signal(&task->cond);
    pthread_mutex_unlock(&task->mutex);
    if (quit)
    {
#ifdef QUEDEBUG
        printf("worker %d is leaving\n", wid);
#endif
        free(x);
//        pthread_exit(NULL);
        return(NULL);
    }
    else
        goto loop;
}

pthread_mutex_t XrdFfsQueueWorker_mutex;
unsigned short XrdFfsQueueNworkers = 0;
unsigned int XrdFfsQueueWorker_id = 0;

int XrdFfsQueue_create_workers(int n)
{
    int i, rc, *id;
    pthread_t *thread;
    pthread_attr_t attr;
    size_t stacksize = 2*1024*1024;

    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, stacksize);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    
    pthread_mutex_lock(&XrdFfsQueueWorker_mutex);
    for (i = 0; i < n; i++)
    {
        id = (int*) malloc(sizeof(int));
        *id = XrdFfsQueueWorker_id++;
        thread = (pthread_t*) malloc(sizeof(pthread_t));
        if (thread == NULL) 
        {
            XrdFfsQueueWorker_id--;
            break;
        }
        rc = pthread_create(thread, &attr, XrdFfsQueue_worker, id);
        if (rc != 0) 
        {
            XrdFfsQueueWorker_id--;
            break;
        }
        pthread_detach(*thread);
        free(thread);
    }
    pthread_attr_destroy(&attr);
    XrdFfsQueueNworkers += i;
    pthread_mutex_unlock(&XrdFfsQueueWorker_mutex);
    return i;
}

int XrdFfsQueue_remove_workers(int n)
{
    int i;
    struct XrdFfsQueueTasks *task;

    pthread_mutex_lock(&XrdFfsQueueWorker_mutex);
    if (XrdFfsQueueNworkers == 0)
        n = 0;
    else if (n > XrdFfsQueueNworkers)
    {
        n = XrdFfsQueueNworkers;
        XrdFfsQueueNworkers = 0;
    }
    else
        XrdFfsQueueNworkers -= n;
    for (i = 0; i < n; i++)
    {
        task = XrdFfsQueue_create_task(NULL, NULL, -1);
        XrdFfsQueue_wait_task(task);
        XrdFfsQueue_free_task(task);
    }
    pthread_mutex_unlock(&XrdFfsQueueWorker_mutex);
    return n;
}

int XrdFfsQueue_count_workers()
{
    int i;
    pthread_mutex_lock(&XrdFfsQueueWorker_mutex);
    i = XrdFfsQueueNworkers;
    pthread_mutex_unlock(&XrdFfsQueueWorker_mutex);
    return i;
}


/* Test program below
   ==================

struct jobargs {
    int i;
    int XrdFfsQueueWorker_id;
};

void* job1(void *arg)
{
     int i = ((struct jobargs*)arg)->i;
//     int wid = ((struct jobargs*)arg)->XrdFfsQueueWorker_id;

//     if (i == 10 || i == 20 || i == 30 || i == 40)
//        sleep(2);
     printf("hello from job1 ( %d )\n", i);
}

int main()
{
    int i;

    XrdFfsQueue_create_workers(20);
#define N 500
    struct XrdFfsQueueTasks *myjob1[N];
    struct jobargs myarg1[N];

    sleep(1);
    printf("1st round ...\n");
    for (i = 0; i < N; i++)
    {
        myarg1[i].i = i;
        myjob1[i] = XrdFfsQueue_create_task((void*) &job1, (void*) &myarg1[i], 0);
    }
    for (i = 0; i < N; i++)
    {
        XrdFfsQueue_wait_task(myjob1[i]);
        XrdFfsQueue_free_task(myjob1[i]);
    }

    printf("there are %d workers after 1st round\n", XrdFfsQueue_count_workers());
    printf("remove %d workers\n", XrdFfsQueue_remove_workers(8));
    printf("add 1 worker\n");
    XrdFfsQueue_create_workers(10);

    sleep(2);
    printf("2nd round ...\n");

    for (i = 0; i < N; i++)
    {
        myarg1[i].i = i;
        myjob1[i] = XrdFfsQueue_create_task((void*) &job1, (void*) &myarg1[i], 0);
    }
    for (i = 0; i < N; i++)
    {
        XrdFfsQueue_wait_task(myjob1[i]);
        XrdFfsQueue_free_task(myjob1[i]);
    }

    XrdFfsQueue_remove_workers(XrdFfsQueue_count_workers());
    printf("bye ...\n");
    return 0; 
}

*/ 

#ifdef __cplusplus
  }
#endif
