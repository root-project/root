#include <iostream>
#include <errno.h>
#include <sys/wait.h>
#include <signal.h>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <TTree.h>
#include <TFile.h>
#include <TMath.h>
#include <TString.h>
#include "pt_data.h"
#include <time.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <TGraphErrors.h>
#include <TROOT.h>
#include <TError.h>
#include <TCanvas.h>
#include <TFrame.h>
#include <TAxis.h>
#include <TH1.h>
#include <TH2.h>
//#define PT_DEBUG
using namespace std;

void saveGraph(Int_t n,double *x,double * y,double * e,const char *ytitle,char * imageName,char * progName);

int main(int argc, char** argv)
{
  const int ExpTime=200; // 2*ExpTime values (not including outliers) stored in tree int status; 
  pid_t pid;
  stringstream sstm;
  const char* fifoName;
  char *nameEnv,*path;
  
#ifdef PT_DEBUG 
  ofstream fout("pt_monitor.txt",ios::app);
  if (!fout) cout << "Cannot open output file pt_monitor.txt\n";
  for (Int_t i=0;i<argc;i++) fout<<argv[i]<<" ";
#endif

  ++argv; // skip "pt_collector", previous argv[1] becomes argv[0] etc
  --argc;

  // build fifo name
  path = NULL;
  path = getcwd(NULL, 0);
  sstm << "PT_FIFONAME=" << path << "/pt_fifo_" << getpid();
  nameEnv = new char[sstm.str().length() + 1];
  strcpy(nameEnv, sstm.str().c_str());
  putenv(nameEnv);
  fifoName = getenv("PT_FIFONAME");
  mkfifo(fifoName, 0666);     

  pid=fork();
  if (pid == 0 )   
    {
      sstm.str("");
      sstm << "LD_PRELOAD=ptpreload.so";
      nameEnv = new char[sstm.str().length() + 1];
      strcpy(nameEnv, sstm.str().c_str());
      putenv(nameEnv);
      execvp(argv[0],argv);    
    } else
    {
      long arr[4];
      int fd,status;
      string string1(argv[argc-1]);
      char *fileName,*progName; 
      double utime,stime;  
      struct rusage usage;
      time_t rawtime;
      TFile* f;
      TTree *t;
      ULong64_t hashVal;
      pt_data *event,*data;
      size_t pos;
      Int_t i,nevent;

      fd=open(fifoName, O_RDONLY); 
      if( fd < 0 ) printf( "Error opening file in PP: %s\n", strerror( errno ) );
 
      // read child performance information
      wait(&status);
#ifdef PT_DEBUG
      fout<<"Exit status after root "<<WEXITSTATUS(status)<<" "<<status<<"\n";
#endif     
      if (status!=0){
	unlink(fifoName);
	return status; // test failed ------------------> do something
      }
      // get memory
      read(fd,&arr,4*sizeof(long));
      unlink(fifoName); 
      if (arr[3]!=699692586){
	cout<<"Error in recording dynamic memory usage for test "<<argv[argc-1]<<"\n"; 
	return 2;
      }
      // get cpu time
      getrusage(RUSAGE_CHILDREN,&usage);      
      utime=usage.ru_utime.tv_sec+(double)usage.ru_utime.tv_usec/1000000;
      stime=usage.ru_stime.tv_sec+(double)usage.ru_stime.tv_usec/1000000;

      // build program name
      string string2(path);
      pos=string2.find("roottest/",0); 
      if (string::npos != pos) string2=string2.substr(pos+9); 
      sstm.str("");
      sstm<< string2 << "/" << string1;
      progName= new char[sstm.str().length() + 1];
      strcpy(progName, sstm.str().c_str());

      // build file name
      pos=string1.find("/",0); // '/' must not be contained in file name
      while(string::npos != pos){
	string1.erase(pos,1);
	pos =string1.find("/",pos+1);   
      }     
      hashVal = TString::Hash(string1.c_str(),string1.length());
      pos = string1.find("."); // cut file name after first '.'
      if (string::npos != pos) string1=string1.substr(0,pos);
      sstm.str("");
      sstm<< "pt_" << string1 << hashVal <<".root";
      fileName = new char[sstm.str().length() + 1];
      strcpy(fileName, sstm.str().c_str());
      
      time (&rawtime);
      event =new pt_data();   
      event->testname=progName;
      event->testtime=ctime(&rawtime);
      event->heapalloc=(float)arr[2]/(float)1000; // in kilobyte
      event->heappeak=(float)arr[1]/(float)1000;
      event->heapleak=(float)arr[0]/(float)1000;
      event->cputime=utime+stime; // in seconds
      event->outlier=0; // if > 0, then outlier
      event->testnumber=0; // increasing integer value, for exponential deletion
      event->meanTime=event->cputime;
      event->varTime=0;
      event->squareTime=event->cputime*event->cputime;
      event->meanHeappeak=event->heappeak;    
      event->varHeappeak=0;
      event->squareHeappeak=event->heappeak*event->heappeak;
      event->meanHeapleak=event->heapleak;    
      event->varHeapleak=0;
      event->squareHeapleak=event->heapleak*event->heapleak;
      event->meanHeapalloc=event->heapalloc;    
      event->varHeapalloc=0;
      event->squareHeapalloc=event->heapalloc*event->heapalloc;
      event->z1=0;
      event->z2=0;
      event->z3=0;
      event->z4=0;
      event->svn=gROOT->GetSvnRevision();

      f=new TFile(fileName, "UPDATE");
      if (f==0) printf("CANNOT create file %s\n",fileName); 
      f->GetObject("PerftrackTree",t);  
      if (t==0) {
#ifdef PT_DEBUG
	fout<<"New Tree \n";
#endif
	t = new TTree("PerftrackTree","Performance monitoring tree");	
	t->Branch("event", &event);
      }
      else{
	double memav,timeav,memstd,timestd,z,allocstd,allocmean,leakstd,leakmean;
	int timetest,heaptest,alloctest,leaktest,k;
	timetest=heaptest=alloctest=leaktest=0;
	data=new pt_data();
	t->SetBranchAddress("event", &data);
	nevent = t->GetEntries();
	
	k=nevent-1;
	while (k>=0){ 
	  t->GetEvent(k);
	  if (data->outlier==0){ 
	    event->testnumber=data->testnumber+1;
	    break;
	  }
	  k--;
	}

	if (event->testnumber % ExpTime == 0){ // exponential cut
	  TTree *nt = t->CloneTree(0);
	  Int_t j;
	  j=k=0;
	  memav=timeav=memstd=timestd=allocstd=allocmean=leakstd=leakmean=0;
	  for (i=0;i<nevent;i++){
	    t->GetEntry(i);
	    if (data->outlier>0){ 
	      nt->Fill();
	      continue;
	    }
	    if (j % 2 == 1){
	      k++; 
	      timeav=(k-1)*timeav+data->cputime;
	      timestd=(k-1)*timestd+data->cputime*data->cputime; // square of sum
	      memav=(k-1)*memav+data->heappeak; 
	      memstd=(k-1)*memstd+data->heappeak*data->heappeak;
	      allocmean=(k-1)*allocmean+data->heapalloc;
	      allocstd=(k-1)*allocstd+data->heapalloc*data->heapalloc;	    
	      leakmean=(k-1)*leakmean+data->heapleak;
	      leakstd=(k-1)*leakstd+data->heapleak*data->heapleak;
	      
	      if ((k*timestd-timeav*timeav)/(double)(k*(k-1))<=0) data->varTime=0;
	      else data->varTime=sqrt((k*timestd-timeav*timeav)/(double)(k*(k-1)));
	      timeav=timeav/(double)k;
	      data->meanTime=timeav;
	      timestd=timestd/(double)k;
	      data->squareTime=timestd;
	      if ((k*memstd-memav*memav)/(double)(k*(k-1))<=0) data->varHeappeak=0;
	      else data->varHeappeak=sqrt((k*memstd-memav*memav)/(double)(k*(k-1)));
	      memav=memav/(double)k;
	      data->meanHeappeak=memav; 
	      memstd=memstd/(double)k;
	      data->squareHeappeak=memstd;
	      if ((k*allocstd-allocmean*allocmean)/(double)(k*(k-1))<=0) data->varHeapalloc=0; 
	      else data->varHeapalloc=sqrt((k*allocstd-allocmean*allocmean)/(double)(k*(k-1)));
	      allocmean=allocmean/(double)k;
	      data->meanHeapalloc=allocmean;
	      allocstd=allocstd/(double)k;
	      data->squareHeapalloc=allocstd;	   
	      if ((k*leakstd-leakmean*leakmean)/(double)(k*(k-1))<=0) data->varHeapleak=0; 
	      else data->varHeapleak=sqrt((k*leakstd-leakmean*leakmean)/(double)(k*(k-1)));
	      leakmean=leakmean/(double)k;
	      data->meanHeapleak=leakmean;
	      leakstd=leakstd/(double)k;
	      data->squareHeapleak=leakstd; 
	      nt->Fill(); 
	    }
	    j++;
	  }
	  t=nt->CloneTree();
	}
	nevent = t->GetEntries();
#ifdef PT_DEBUG
	fout<<"Entries already there "<<nevent<<"\n";	
	for (i=0;i<nevent;i++){
	  t->GetEvent(i);
	  fout <<" "<<data->testnumber<<" "<<data->testtime<< " " << data->testname << " cputime "  << data->cputime << " heappeak  " << data->heappeak <<  " heapalloc " << data->heapalloc << " heapleak  "<<data->heapleak<<" --  mean cputime "<<data->meanTime<<" std cputime "<<data->varTime<<", mean heap peak  "<<data->meanHeappeak<<" std heap peak "<<data->varHeappeak<<", mean heap alloc  "<<data->meanHeapalloc<<" std heap alloc  "<<data->varHeapalloc<<" mean heap leak "<<data->meanHeapleak<<" std heap leak "<<data->varHeapleak<<"  --  outlier "<<data->outlier<<" z1 "<<data->z1<<" z2 "<<data->z2<<" z3 "<<data->z3<<" z4 "<<data->z4<<"\n";
	}
#endif
	
	k=nevent-1;
	while (k>=0){ 
	  t->GetEvent(k);
	  if (data->outlier==0){ 
	    timeav=data->meanTime;
	    timestd=data->squareTime;// to compute variance  	    
	    memav=data->meanHeappeak; 
	    memstd=data->squareHeappeak;
	    allocmean=data->meanHeapalloc;
	    allocstd=data->squareHeapalloc;	    
	    leakmean=data->meanHeapleak;
	    leakstd=data->squareHeapleak;
	    break;
	  }
	  k--;
	}

	if (nevent>0){    	  
	  // calculate moving mean and stddev
	  timeav=nevent*timeav+event->cputime;	  
	  timestd=nevent*timestd+event->cputime*event->cputime;
	  event->squareTime=timestd/(double)(nevent+1);
	  timestd=((nevent+1)*timestd-timeav*timeav)/(double)((nevent+1)*nevent); // variance
	  timeav=timeav/(double)(nevent+1); // mean
	  if (timestd<0) timestd=0;	  
	  timestd=sqrt(timestd); // std dev

	  memav=nevent*memav+event->heappeak;
	  memstd=nevent*memstd+event->heappeak*event->heappeak;
	  event->squareHeappeak=memstd/(double)(nevent+1);
	  memstd=((nevent+1)*memstd-memav*memav)/(double)((nevent+1)*nevent);	  
	  memav=memav/(double)(nevent+1);
	  if (memstd<0) memstd=0;
	  memstd=sqrt(memstd);

	  allocmean=nevent*allocmean+event->heapalloc;
	  allocstd=nevent*allocstd+event->heapalloc*event->heapalloc;
	  event->squareHeapalloc=allocstd/(double)(nevent+1);
	  allocstd=((nevent+1)*allocstd-allocmean*allocmean)/(double)((nevent+1)*nevent);	  
	  allocmean=allocmean/(double)(nevent+1);
	  if (allocstd<0) allocstd=0;
	  allocstd=sqrt(allocstd);

	  leakmean=nevent*leakmean+event->heapleak;
	  leakstd=nevent*leakstd+event->heapleak*event->heapleak;
	  event->squareHeapleak=leakstd/(double)(nevent+1);
	  leakstd=((nevent+1)*leakstd-leakmean*leakmean)/(double)((nevent+1)*nevent); 	  
	  leakmean=leakmean/(double)(nevent+1);
	  if (leakstd<0) leakstd=0;
	  leakstd=sqrt(leakstd);
#ifdef PT_DEBUG
	  fout<<(nevent+1)<<" entries\n";
	  fout<<"CPU time:  MEANS  "<<timeav<<"    STD "<<timestd<<"\n";
	  fout<<"Heappeak:  MEANS  "<<memav<<"     STD "<<memstd<<"\n";    
	  fout<<"Heapalloc: MEANS  "<<allocmean<<" STD "<<allocstd<<"\n";
	  fout<<"Heapleak:  MEANS  "<<leakmean<<"  STD "<<leakstd<<"\n";
#endif
	  if (nevent>1){
	    // test for outliers
	    z=0;
	    if (timestd>0 && event->cputime>timeav){ 
	      z=(event->cputime-timeav)/(timestd+0.05);    
	      if (z>5.0) timetest=1;
	    }
	    event->z1=z;	    
	    z=0;		    
	    if (memstd>0 && event->heappeak>memav){ 
	      z=(event->heappeak-memav)/(memstd+0.1); // z-value	     
	      if (z>4.5) heaptest=1;
	    }
	    event->z2=z;	    
	    z=0;
	    if (allocstd>0 && event->heapalloc>allocmean){ 
	      z=(event->heapalloc-allocmean)/(allocstd+0.1); // z-value    
	      if (z>5.0) alloctest=1;
	    }
	    event->z3=z;	    
	    z=0;
	    if (leakstd>0 && event->heapleak>leakmean){ 
	      z=(event->heapleak-leakmean)/(leakstd+0.1); // z-value   
	      if (z>3.5) leaktest=1;
	    }
	    event->z4=z;

#ifdef PT_DEBUG
	    fout<<"Tests time: z=="<<z<<"? "<<timetest<<"\n";
	    fout<<"Tests heappeak: z=="<<z<<"? "<<heaptest<<"\n";
	    fout<<"Tests heapalloc: z=="<<z<<"? "<<alloctest<<"\n";
	    fout<<"Tests heapleak: z=="<<z<<"? "<<leaktest<<"\n";
#endif
	    event->outlier=timetest+heaptest+leaktest+alloctest;
	  } //endif (nevent>3)	   
	} //endif (nevent>2)  

	if (event->outlier>0){ 
	  cout<<"Performance decrease for test "<<progName<<" in file "<<fileName<<endl;
	  cout<<"CPU-time (in seconds): changed from "<<data->meanTime<<" to "<<event->cputime<<endl;
	  cout<<"Dynamical memory (in kilobyte): peak memory usage changed from "<<data->meanHeappeak<<" to "<<event->heappeak<<", size allocated memory changed from "<<data->meanHeapalloc<<" to "<<event->heapalloc<<", size memory leaks changed from "<<data->meanHeapleak<<" to "<<event->heapleak<<endl;
	  timeav=data->meanTime;
	  timestd=data->squareTime;// take old values	    
	  memav=data->meanHeappeak; 
	  memstd=data->squareHeappeak;
	  allocmean=data->meanHeapalloc;
	  allocstd=data->squareHeapalloc;	    
	  leakmean=data->meanHeapleak;
	  leakstd=data->squareHeapleak;
	}
	event->meanTime=timeav;
	event->varTime=timestd;
	event->meanHeappeak=memav;
	event->varHeappeak=memstd;
	event->meanHeapalloc=allocmean;
	event->varHeapalloc=allocstd;
	event->meanHeapleak=leakmean;
	event->varHeapleak=leakstd;     
	t->SetBranchAddress("event", &event);
	
      } //endelse

#ifdef PT_DEBUG
      fout <<" "<<"WRITE CURRENT "<<event->testnumber<<" "<<event->testtime<< " "<<event->testname <<" cputime "<<event->cputime<<" heappeak  "<<event->heappeak<<" heapalloc "<<event->heapalloc<<" heapleak  "<<event->heapleak<<" --  mean heap  "<<event->meanHeappeak<<" std heap  "<<event->varHeappeak<<" mean cputime "<<event->meanTime<<" std cputime "<<event->varTime<<" outlier "<<event->outlier<<"\n";
      fout<<"Exit status now "<<WEXITSTATUS(status)<<" "<<status<<"\n\n";
      fout.close();
#endif
      t->Fill();      
      f->Write("",TObject::kOverwrite);          
           
      gErrorIgnoreLevel=3000;     
     
      double *x,*etime,*eheappeak,*eheapleak,*eheapalloc,*ytime,*yheappeak,*yheapalloc,*yheapleak;
      char *imageName; 
      const char* ytitle;

      nevent = t->GetEntries();
      t->SetBranchAddress("event", &data);
      x=new double[nevent];
      ytime=new double[nevent];
      yheappeak=new double[nevent];
      yheapalloc=new double[nevent];
      yheapleak=new double[nevent];
      etime=new double[nevent];
      eheappeak=new double[nevent]; 
      eheapalloc=new double[nevent]; 
      eheapleak=new double[nevent];
      for (i=0;i<nevent;i++){
	t->GetEvent(i);
	x[i]=(double)data->svn;
	ytime[i]=data->cputime;
	yheappeak[i]=data->heappeak;
	yheapleak[i]=data->heapleak;
	yheapalloc[i]=data->heapalloc;
	etime[i]=data->varTime;
	eheappeak[i]=data->varHeappeak;
	eheapalloc[i]=data->varHeapalloc;
	eheapleak[i]=data->varHeapleak; 
      }
      
      sstm.str("");
      sstm<< "pt_" << string1 << hashVal << "_cputime.gif";
      imageName = new char[sstm.str().length() + 1];
      strcpy(imageName, sstm.str().c_str());           
      ytitle="CPU time [s]";
      saveGraph(nevent,x,ytime,etime,ytitle,imageName,progName);

      sstm.str("");
      sstm<< "pt_" << string1 << hashVal << "_heappeak.gif";
      imageName = new char[sstm.str().length() + 1];
      strcpy(imageName, sstm.str().c_str());
      ytitle="Peak heap usage [kB]";
      saveGraph(nevent,x,yheappeak,eheappeak,ytitle,imageName,progName);

      sstm.str("");
      sstm<< "pt_" << string1 << hashVal << "_heapalloc.gif";
      imageName = new char[sstm.str().length() + 1];
      strcpy(imageName, sstm.str().c_str()); 
      ytitle="Size of total allocated memory [kB]";
      saveGraph(nevent,x,yheapalloc,eheapalloc,ytitle,imageName,progName);

      sstm.str("");
      sstm<< "pt_" << string1 << hashVal << "_heapleak.gif";
      imageName = new char[sstm.str().length() + 1];
      strcpy(imageName, sstm.str().c_str());
      ytitle="Size of memory leaks [kB]";
      saveGraph(nevent,x,yheapleak,eheapleak,ytitle,imageName,progName);

      delete x;
      delete ytime;
      delete yheappeak;
      delete yheapalloc;
      delete yheapleak;
      delete etime;
      delete eheappeak;
      delete eheapalloc;
      delete eheapleak;

      delete data;
      if (event->outlier>0){ 
	delete event;
	delete t;
	return 1;
      }
      delete event;    
      delete t;
    }
  return 0;
} 


void saveGraph(Int_t n,double *x,double * y,double * e,const char *ytitle,char * imageName,char * progName)
{ 
  char * title;
  stringstream sstm;
  TCanvas *c1;
  TGraphErrors *g;
  sstm<< "Distribution of "<<ytitle<<" for test '" << progName << "'.";
  title = new char[sstm.str().length() + 1];
  strcpy(title, sstm.str().c_str()); ;
  c1 = new TCanvas("c1","Performance Monitoring Plots",200,80,1500,1000); 
  c1->SetFillColor(32);
  c1->SetGrid();
  g = new TGraphErrors(n,x,y,0,e);
  g->SetTitle(title);
  g->SetMarkerColor(4);
  g->SetMarkerStyle(20); // 7 probably faster (not scalable)
  g->SetLineColor(2);
  g->GetXaxis()->SetTitle("SVN revision");
  g->GetXaxis()->SetTitleOffset(1.0);
  g->GetYaxis()->SetTitle(ytitle);
  g->GetYaxis()->SetTitleOffset(1.5);  
  g->Draw("AP");
  c1->SaveAs(imageName);
  delete c1;
  delete g;
}
