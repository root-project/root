#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <memory.h>
#include <pwd.h>
#include <sys/time.h>
#include <sys/timeb.h>
#include <sys/types.h>
#include <sys/stat.h>

/********************************************************************/
main(int argc, char **argv)
{
  char cSource[64];
  int iBufferSize, iInterval,iFile,iRead,iLoop=0,ii,iBuffers=0,iSize;
  int *piBuffer;
  FILE *fp;
  if(argc == 1)
    {
      printf("-c file size[MB] buffersize[KB]\n");
      printf("-r file buffersize[KB]\n");
      printf("-t file buffersize[KB] 1..10\n");
      exit(0);
    }
  /*-------------------------------------------------------------------------------*/
  if(strstr(argv[1],"-c") != 0)
    {
      strcpy(cSource,argv[2]);
      iSize=atol(argv[3])*1000;
      iBufferSize=atol(argv[4]);
      printf("Create file %s, size %d M, buffer %d K\n",cSource,iSize/1000,iBufferSize);

      piBuffer=(int *)malloc(iBufferSize*1000);
      *piBuffer=0;
      *(piBuffer+1)=iBufferSize*1000;
      fp=fopen(cSource,"w");
      for(ii=0;ii<iSize/iBufferSize;ii++)
	{
	  (*piBuffer)++;
	  fwrite(piBuffer,iBufferSize*1000,1,fp);
	}
      fclose(fp);
      exit;
    }
  /*-------------------------------------------------------------------------------*/
  if(strstr(argv[1],"-r") != 0)
    {
      strcpy(cSource,argv[2]);
      iBufferSize=atol(argv[3])*1000;
      printf("Read total file %s, buffer %3d [K]",cSource,iBufferSize/1000);fflush(stdout);

      piBuffer=(int *)malloc(iBufferSize);
      iBuffers=-1;
      iFile=open(cSource,O_RDONLY);
      iSize=lseek(iFile,0,SEEK_END);
      lseek(iFile,0,SEEK_SET);
      ii=1;
      while(ii) {ii=read(iFile,piBuffer,iBufferSize);iBuffers++;}
      close(iFile);
      printf("%6d read: ",iBuffers);fflush(stdout);
      exit;
    }
  /*-------------------------------------------------------------------------------*/
  if(strstr(argv[1],"-s") != 0)
    {
      strcpy(cSource,argv[2]);
      iBufferSize=atol(argv[3])*1000;
      iInterval=atol(argv[4]);

      piBuffer=(int *)malloc(iBufferSize);
      iFile=open(cSource,O_RDONLY);
      iSize=lseek(iFile,0,SEEK_END);
      printf("\nSeq.   %s %4d [M], buffer %3d [K], take %2d of 10. ",
            cSource,iSize/1000000,iBufferSize/1000,iInterval);fflush(stdout);

      while(1)
	{
	  lseek(iFile,iLoop,SEEK_SET);// set to next bufferstream (10 buffers)
          for(ii=0;ii<iInterval;ii++)
	    {
	      iRead=read(iFile,piBuffer,iBufferSize);// read sequential buffers
	      if(iRead != iBufferSize) break;
	      iBuffers++;
	    }
	  if(iRead != iBufferSize) break;
          iLoop += iBufferSize*10;
	}
      free(piBuffer);
      close(iFile);
      printf("%6d read: ",iBuffers);fflush(stdout);
      exit;
    }
  /*-------------------------------------------------------------------------------*/
  if(strstr(argv[1],"-t") != 0)
    {
      strcpy(cSource,argv[2]);
      iBufferSize=atol(argv[3])*1000;
      iInterval=atol(argv[4]);

      piBuffer=(int *)malloc(iBufferSize);
      iFile=open(cSource,O_RDONLY);
      iSize=lseek(iFile,0,SEEK_END);
      printf("\nNoseq. %s %4d [M], buffer %3d [K], take %2d of 10. ",
            cSource,iSize/1000000,iBufferSize/1000,iInterval);fflush(stdout);

      for(ii=0;ii<iInterval;ii++)
	{
	  iLoop=ii*iBufferSize;
	  while(1)
	    {
	      lseek(iFile,iLoop,SEEK_SET); // set to next buffer
	      iRead=read(iFile,piBuffer,iBufferSize);
	      if(iRead != iBufferSize) break;
	      iBuffers++;
	      iLoop += iBufferSize*10;
	    }
	}
      free(piBuffer);
      close(iFile);
      printf("%6d read: ",iBuffers);fflush(stdout);
    }
}
