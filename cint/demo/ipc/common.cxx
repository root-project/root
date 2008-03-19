/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// common.cxx

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <sys/msg.h>

const char *keyfile = "./keyfile";
const char projid = 'Q';
const char projid2 = 'q';
const int semsize = 5;

key_t mykey;
key_t mykey2;

// shared memory
int myshmid;
void *shmbuffer;

// semaphore
int mysemid;
struct sembuf lock[1]   = { { 1, -1, 0 } };
struct sembuf unlock[1] = { { 1,  1, 0 } };

// message
int mymsg;
int mymsg2;

/////////////////////////////////////////////////////////////////////////
// Initialize IPC
/////////////////////////////////////////////////////////////////////////
void initsend() {
  // Prepare keys
  mykey=ftok(keyfile,projid);
  printf("mykey=%x\n",mykey);
  mykey2=ftok(keyfile,projid2);
  printf("mykey2=%x\n",mykey2);

  // Shared Memory
  myshmid = shmget(mykey,semsize*sizeof(int),SHM_R|SHM_W|IPC_CREAT);
  printf("myshmid=%x\n",myshmid);
  if(-1 == myshmid) {
    fprintf(stderr,"shmget failed\n");
    exit(1);
  }

  shmbuffer = shmat(myshmid,0,0);
  printf("shmbuffer=%x\n",shmbuffer);
  if((void*)0xffffffff==shmbuffer) {
    fprintf(stderr,"shmat failed\n");
    exit(1);
  }

#ifndef NOSEM
  // Semaphoe
  mysemid = semget(mykey,2,SHM_R|SHM_W|IPC_CREAT);
  printf("semid=%x\n",mysemid);
  if(-1 == mysemid) {
    fprintf(stderr,"semget failed\n");
  }

  //int stat = semctl(mysemid,0,SETALL,(void*)1);
  //printf("semctl = %d\n",stat);

  // set semval = 1 ,  same operation as unlock
  int stat = semop(mysemid,unlock,1);
  printf("semstat = %d\n",stat);
#endif

  // Message
  mymsg = msgget(mykey,SHM_R|SHM_W|IPC_CREAT);
  printf("mymsg = %d\n",mymsg);
  mymsg2 = msgget(mykey2,SHM_R|SHM_W|IPC_CREAT);
  printf("mymsg2 = %d\n",mymsg2);
}

////////////////////////////////////////////////////////////////////////
void initrecv() {
  mykey=ftok(keyfile,projid);
  printf("mykey=%x\n",mykey);
  mykey2=ftok(keyfile,projid2);
  printf("mykey2=%x\n",mykey2);

  // Shared Memory
  while(-1==(myshmid=shmget(mykey,semsize*sizeof(int),SHM_R|SHM_W))) {
    printf("myshmid=%x\n",myshmid);
    sleep(1);
  }

  shmbuffer = shmat(myshmid,0,0);
  printf("shmbuffer=%x\n",shmbuffer);
  if((void*)0xffffffff==shmbuffer) {
    fprintf(stderr,"shmat failed\n");
    exit(1);
  }

#ifndef NOSEM
  // Semaphoe
  mysemid = semget(mykey,1,SHM_R|SHM_W);
  printf("mysemid=%x\n",mysemid);
  if(-1 == mysemid) {
    fprintf(stderr,"semget failed\n");
  }
#endif

  // Message
  mymsg = msgget(mykey,SHM_R|SHM_W);
  printf("mymsg = %d\n",mymsg);
  mymsg2 = msgget(mykey2,SHM_R|SHM_W);
  printf("mymsg2 = %d\n",mymsg2);
}

/////////////////////////////////////////////////////////////////////////
// Send and Receive information vir shared memory
/////////////////////////////////////////////////////////////////////////
void send(int x=2) {
#ifndef NOSEM
  // semaphore lock
  int stat = semop(mysemid,lock,1);
  printf("semstat = %d\n",stat);
#endif

  // shared memory
  int *ary = (int*)shmbuffer;
  for(int i=0;i<semsize;i++) {
    ary[i] = i*x;
    sleep(1);
    printf("send %d\n",ary[i]);
  }

#ifndef NOSEM
  // semaphore unlock
  stat = semop(mysemid,unlock,1);
  printf("semstat = %d\n",stat);
#endif
}

////////////////////////////////////////////////////////////////////////
void recv() {
#ifndef NOSEM
  // semaphore lock
  int stat = semop(mysemid,lock,1);
  printf("semstat = %d\n",stat);
#endif

  // shared memory
  int *ary = (int*)shmbuffer;
  for(int i=0;i<semsize;i++) {
    printf("recv %d\n",ary[i]);
    sleep(1);
  }

#ifndef NOSEM
  // semaphore unlock
  stat = semop(mysemid,unlock,1);
  printf("semstat = %d\n",stat);
#endif
}

/////////////////////////////////////////////////////////////////////////
// Send and Receive message
/////////////////////////////////////////////////////////////////////////
struct msgbuff { long mtype; char mtext[80]; };
void sendmsg(char* m) {
  int stat;
  struct msgbuff b;
  b.mtype = 1;
  strcpy(b.mtext,m);
  stat = msgsnd(mymsg,(msgbuf*)&b,strlen(b.mtext)+1,0);
  //printf("msgsnd = %d\n",stat);
}

/////////////////////////////////////////////////////////////////////////
void recvmsg() {
  int stat;
  struct msgbuff  b;
  stat = msgrcv(mymsg,(msgbuf*)&b,80,0,0);
  printf("msgrcv = %d  %s\n",stat,b.mtext);
}

/////////////////////////////////////////////////////////////////////////
void sendmsg2(char* m) {
  int stat;
  struct msgbuff b;
  b.mtype = 1;
  strcpy(b.mtext,m);
  stat = msgsnd(mymsg2,(msgbuf*)&b,strlen(b.mtext)+1,0);
  //printf("msgsnd = %d\n",stat);
}

/////////////////////////////////////////////////////////////////////////
void recvmsg2() {
  int stat;
  struct msgbuff  b;
  stat = msgrcv(mymsg2,(msgbuf*)&b,80,0,0);
  printf("msgrcv = %d  %s\n",stat,b.mtext);
}

/////////////////////////////////////////////////////////////////////////
// Terminate IPC
/////////////////////////////////////////////////////////////////////////
void finishsend() {
  // shared memory
  shmdt(shmbuffer);
  shmctl(myshmid,IPC_RMID,0);
  
#ifndef NOSEM
  // semaphore
#ifdef G__HPUX
  semctl(mysemid,0,IPC_RMID,0);
#else
  semun arg;
  semctl(mysemid,0,IPC_RMID,arg);
#endif
#endif

  // message
  msgctl(mymsg,IPC_RMID,0);
}

////////////////////////////////////////////////////////////////////////
void finishrecv() {
  // shared memory
  shmdt(shmbuffer);
}

