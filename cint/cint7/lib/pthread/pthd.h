/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file lib/pthread/pthd.h
 ************************************************************************
 * Description:
 *  pthread API
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// Linux2.0 egcs   : g++ -lpthread pthread1.cxx
// HP-UX aCC       : aCC -D_POSIX_C_SOURCE=199506L -lpthread pthread1.cxx

#ifndef G__PTHREADDLL_H
#define G__PTHREADDLL_H

#ifndef __MAKECINT__

#include <pthread.h>

#else /* __MAKECINT__ */

/************************************************************************
 * Types
 *	pthread_t		Used to identify a thread.		*
 *	pthread_attr_t		Used to identify a thread attributes obj*
 *	pthread_mutex_t		Used for mutexes.			*
 *	pthread_mutexattr_t	Used to identify a mutex attributes obj	*
 *	pthread_cond_t		Used for condition variables.		*
 *	pthread_condattr_t	Used to identify a condition attribute	*
 *	pthread_key_t		Used for thread-specific data keys.	*
 *	pthread_once_t		Used for dynamic package initialization.*
 ************************************************************************/
typedef int	pthread_t;
typedef int	pthread_attr_t;

struct pthread_cond ;
typedef struct pthread_cond	pthread_cond_t;
typedef int	pthread_condattr_t;

struct pthread_mutex;
typedef	struct pthread_mutex	pthread_mutex_t;
typedef int	pthread_mutexattr_t;

typedef int	pthread_key_t;

typedef struct 
{
#if 0
	char			po_executing;
	char			po_completed;
	pthread_mutex_t		po_mutex;
	pthread_cond_t		po_executed;
#endif
} 
pthread_once_t;

struct timespec;

/************************************************************************
 * Functions
 ************************************************************************/
int pthread_create(pthread_t *thread,const pthread_attr_t *attr
		  ,void *(*start_routine)(void*),void *arg);
void pthread_exit(void *value_ptr);
int pthread_detach(pthread_t thread);

int pthread_join(pthread_t thread, void **value_ptr);

int pthread_cond_wait(pthread_cond_t *cond,pthread_mutex_t *mutex);
#if 0
int pthread_cond_timewait(pthread_cond_t *cond,pthread_mutex_t *mutex
                         ,const struct timespec *abstime);
#endif

int pthread_cond_signal(pthread_cond_t *cond);
int pthread_cond_broadcast(pthread_cond_t *cond);

#endif /* __MAKECINT__ */

#endif
