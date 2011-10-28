/* fpont 12/99 */
/* pont.net    */
/* udpserver.c */

/* Converted to echo client/server with select() (timeout option).
   See testTUDPSocket.C */
/* Compile with: gcc udpserver.c -o udpserver 
/* 3/30/05 John Schultz */

#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <stdio.h>
#include <unistd.h> /* close() */
#include <string.h> /* memset() */

#define LOCAL_SERVER_PORT 1500
#define MAX_MSG 100

int main(int argc, char *argv[]) {
  
  int sd, rc, n, flags;
  unsigned cliLen;
  struct sockaddr_in cliAddr, servAddr;
  char msg[MAX_MSG];

  /* socket creation */
  sd=socket(AF_INET, SOCK_DGRAM, 0);
  if(sd<0) {
    printf("%s: cannot open socket \n",argv[0]);
    exit(1);
  }

  /* bind local server port */
  servAddr.sin_family = AF_INET;
  servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
  servAddr.sin_port = htons(LOCAL_SERVER_PORT);
  rc = bind (sd, (struct sockaddr *) &servAddr,sizeof(servAddr));
  if(rc<0) {
    printf("%s: cannot bind port number %d \n", 
	   argv[0], LOCAL_SERVER_PORT);
    exit(1);
  }

  printf("%s: waiting for data on port UDP %u\n", 
	   argv[0],LOCAL_SERVER_PORT);

/* BEGIN jcs 3/30/05 */

  flags = 0;

/* END jcs 3/30/05 */

  /* server infinite loop */
  while(1) {
    
    /* init buffer */
    memset(msg,0x0,MAX_MSG);

    /* receive message */
    cliLen = sizeof(cliAddr);
    n = recvfrom(sd, msg, MAX_MSG, flags,
		 (struct sockaddr *) &cliAddr, &cliLen);

    if(n<0) {
      printf("%s: cannot receive data \n",argv[0]);
      continue;
    }
      
    /* print received message */
    printf("%s: from %s:UDP%u : %s \n", 
	   argv[0],inet_ntoa(cliAddr.sin_addr),
	   ntohs(cliAddr.sin_port),msg);

/* BEGIN jcs 3/30/05 */

    sleep(1);
    sendto(sd,msg,n,flags,(struct sockaddr *)&cliAddr,cliLen);

/* END jcs 3/30/05 */
    
  }/* end of server infinite loop */

return 0;

}
