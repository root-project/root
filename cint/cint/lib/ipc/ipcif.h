/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file lib/ipc/ipcif.h
 ************************************************************************
 * Description:
 *  Create Inter Process Communication API
 ************************************************************************
 * Copyright(c) 1995~2002  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

/* Please read README file in this directory */

#ifndef G__IPCDLL_H
#define G__IPCDLL_H

#ifndef __MAKECINT__

/**************************************************************************
 * Include system header files
 **************************************************************************/
#include <time.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#if defined(G__APPLE) || defined(__APPLE__)
/* union semun is defined by including <sys/sem.h> */
#elif defined(__GNU_LIBRARY__) && !defined(_SEM_SEMUN_UNDEFINED)
/* union semun is defined by including <sys/sem.h> */
#else
/* according to X/OPEN we have to define it ourselves */
#if !defined(__FreeBSD__) && !defined(__KCC) && !defined(__sgi)
union semun {
  int val;                    /* value for SETVAL */
  struct semid_ds *buf;       /* buffer for IPC_STAT, IPC_SET */
  unsigned short int *array;  /* array for GETALL, SETALL */
  struct seminfo *__buf;      /* buffer for IPC_INFO */
};
#endif
#endif 

/* sys/msg.h does not exist on macosx */
#if defined(G__APPLE) || defined(__APPLE__)
#else
#include <sys/msg.h>
#endif

#else /* __MAKECINT__ */

#include <time.h>
#include <sys/types.h>

struct ipc_parm;
struct ipc_perm;
struct shmid_ds;
struct semid_ds;
struct msqid_ds;

union semun;

struct sembuf;
struct msgbuf; /* does not exist in RH7.0 */

/**************************************************************************
 * convert a pathname and a project id to a System V IPC Key
 **************************************************************************/
/* Flags under Linux2.0 */
#define IPC_CREAT       01000
#define IPC_EXCL        02000
#define IPC_NOWAIT      04000

#define IPC_RMID        0
#define IPC_SET         1
#define IPC_STAT        2
#if !defined(G__HPUX)
#define IPC_INFO        3
#endif

/* typedef int key_t; */

key_t ftok(char *pathname, char proj);


/**************************************************************************
 * Shared Memory API
 **************************************************************************/
#define SHM_R           0400
#define SHM_W           0200

#define SHM_RDONLY      010000
#define SHM_RND         020000
#if !defined(G__HPUX)
#define SHM_REMAP       040000
#endif

#if defined(G__GNUC)
#define SHM_LOCK        11
#define SHM_UNLOCK      12
#elif defined(G__HPUX)
#define SHM_LOCK        3
#define SHM_UNLOCK      4
#else
#define SHM_LOCK        11
#define SHM_UNLOCK      12
#endif

#define SHM_STAT        13
#define SHM_INFO        14

struct ipc_parm 
#if 0
{
  key_t key;
  ushort uid;
  ushort gid;
  ushort cuid;
  ushort cgid;
  ushort mode;
  ushort seq;
}
#endif
;

struct shmid_ds {
  struct ipc_perm shm_perm;
  int shm_segsz;
  time_t shm_atime;
  time_t shm_dtime;
  time_t shm_ctime;
  unsigned short shm_cpid;
  unsigned short shm_lpid;
  short shm_nattch;
};

int shmget(key_t key, int size, int shmflg);
char *shmat(int shmid,char* shmaddr, int shmflg);
int shmdt(char* shmaddr);
int shmctl(int shmid,int cmd,struct shmid_ds *buf);


/**************************************************************************
 * Semaphoe API
 **************************************************************************/
#define GETALL     6
#define SETVAL     8
#define SETALL     9

struct semid_ds;

#if defined(G__GNUC)
union semun {
  int val;
  struct semid_ds *buf;
  unsigned short *array;
};
#elif defined(G__HPUX)
/* nothing ? */
#else
union semun {
  int val;
  struct semid_ds *buf;
  unsigned short *array;
};
#endif

struct sembuf {
  ushort sem_num; /* semaphore number */
  short sem_op;   /* semaphore operation */
  short sem_flg;  /* operation flags */
};

int semget(key_t key, int nsems,int semflg);
#if defined(G__GNUC)
int semctl(int semid,int semnum,int cmd,union semun arg);
#elif defined(G__HPUX)
int semctl(int semid,int semnum,int cmd,void* x);
#else
int semctl(int semid,int semnum,int cmd,union semun arg);
#endif
int semop(int semid,struct sembuf *sops,unsigned int nsops);


/**************************************************************************
 * Message API
 **************************************************************************/
struct msgbuf {
  long mtype;
  char mtext[80];  /* This is dummy */
};

struct msqid_ds;

#if !(defined(G__APPLE) || defined(__APPLE__))
int msgget(key_t key,int msgflg);
int *msgsnd(int msgid,struct msgbuf *msgp,int msgsz,int msgflg);
int msgrcv(int msgid,struct msgbuf *msgp,int msgsz,long msgtyp,int msgflg);
int msgctl(int msgid, int cmd,struct msqid_ds *buf);
#endif


#pragma link off struct msgbuf;
#endif /* __MAKECINT__ */

#endif /* G__IPC_H */

