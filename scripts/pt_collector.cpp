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
#include <TAxis.h>
//#define PT_DEBUG
using namespace std;

void saveGraph(Int_t n,double *x,double * y,double * e,const char *ytitle,char * imageName,char * progName,double v);
TTree* createTree(TFile *f,pt_data *event);
TTree* detectOutliers(TFile *f,TTree *t,pt_data *event);
TTree* deleteEntries(TTree *t);

int main(int argc, char** argv)
{
  pid_t pid;
  stringstream sstm;
  const char* fifoName;
  char *nameEnv,*path;
  const int ExpTime=200; // 2*ExpTime values (not including outliers) stored in tree int status; 
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
      char *fileName,*progName,*imageName;
      const char* ytitle; 
      double utime,stime,*x,*etime,*eheappeak,*eheapleak,*eheapalloc,*ytime,*yheappeak,*yheapalloc,*yheapleak;  
      struct rusage usage;
      time_t rawtime;     
      ULong64_t hashVal;size_t pos;
      pt_data *event,*data;
      TFile* f;
      TTree *t;
      Int_t i,nevent;

      fd=open(fifoName, O_RDONLY); 
      if( fd < 0 ) printf( "Error opening file in PP: %s\n", strerror( errno ) );
 
      // read child performance information
      wait(&status);
#ifdef PT_DEBUG
      fout<<"\nExit status after root "<<WEXITSTATUS(status)<<" "<<status<<"\n";
#endif     
      if (status!=0){
	unlink(fifoName);
	return status; // test failed
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
	pos =string1.find("/",pos);   
      }     
      hashVal = TString::Hash(string1.c_str(),string1.length());
      pos = string1.find("."); // cut file name after first '.'
      if (string::npos != pos) string1=string1.substr(0,pos);
      sstm.str("");
      sstm<< "pt_" << string1 << hashVal <<".root";
      fileName = new char[sstm.str().length() + 1];
      strcpy(fileName, sstm.str().c_str());
      
      time (&rawtime);
      data=new pt_data();   
      event =new pt_data();
      event->testname=progName;
      event->testtime=ctime(&rawtime);
      event->heapalloc=(float)arr[2]/(float)1000; // in kilobyte
      event->heappeak=(float)arr[1]/(float)1000;
      event->heapleak=(float)arr[0]/(float)1000;
      event->cputime=utime+stime; // in seconds
      event->outlier=0; // if > 0, then outlier
      event->testnumber=1; // increasing integer value, for moving mean/variance
      event->counter=1; // increasing integer value, for exponential deletion
      event->meanTime=event->cputime;
      event->varTime=0;
      event->squareTime=event->cputime*event->cputime; //for moving variance
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
   
      // detect outliers & save data
      int outlier=0;
      f=new TFile(fileName, "UPDATE");
      if (f==0) printf("CANNOT create file %s\n",fileName); 
      f->GetObject("PerftrackTree",t);
      
      if (t==0) t=createTree(f,event);
      else{ 
	t->SetBranchAddress("event", &data);
	i=(t->GetEntries())-1;
	while (i>=0){ 
	  t->GetEvent(i);
	  if (data->outlier==0){ 
	    event->counter=data->counter+1;
	    break;
	  }
	  i--;
	}
	if (event->counter % ExpTime == 0){ 
#ifdef PT_DEBUG
	  fout<<"Counter "<<event->counter<<": Delete entries\n";
#endif
	  t=deleteEntries(t);
	}
	t=detectOutliers(f,t,event);
      } 
      //f->GetObject("PerftrackTree",t);

#ifdef PT_DEBUG
      nevent = t->GetEntries();
      t->SetBranchAddress("event", &data);
      fout<<"Entries there "<<nevent<<"\n";	
      for (i=0;i<nevent;i++){
	t->GetEvent(i);	  
	fout <<"value "<<data->counter<<" "<<data->testnumber<<" "<<data->testtime<< " " << data->testname << " cputime "  << data->cputime << " heappeak  " << data->heappeak <<  " heapalloc " << data->heapalloc << " heapleak  "<<data->heapleak<<" --  mean cputime "<<data->meanTime<<" std cputime "<<data->varTime<<", mean heap peak  "<<data->meanHeappeak<<" std heap peak "<<data->varHeappeak<<", mean heap alloc  "<<data->meanHeapalloc<<" std heap alloc  "<<data->varHeapalloc<<" mean heap leak "<<data->meanHeapleak<<" std heap leak "<<data->varHeapleak<<"  --  outlier "<<data->outlier<<" z1 "<<data->z1<<" z2 "<<data->z2<<" z3 "<<data->z3<<" z4 "<<data->z4<<"\n";
      }  
      fout<<"\n";
      fout.close();
#endif
      
      if (event->outlier>0){
	cout<<"Performance decrease for test "<<progName<<" in file "<<fileName<<endl;
	cout<<"CPU-time: changed from "<<data->meanTime<<" s to "<<event->cputime<<" s\n";
	cout<<"Dynamical memory: peak memory usage changed from "<<data->meanHeappeak<<" kB to "<<event->heappeak<<" kB\n";
	cout<<"Dynamical memory: size allocated memory changed from "<<data->meanHeapalloc<<" kB to "<<event->heapalloc<<" kB\n";
	cout<<"Dynamical memory: size memory leaks changed from "<<data->meanHeapleak<<" kB to "<<event->heapleak<<" kB\n";
	outlier=1;
      }

      // paint graphs
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
	x[i]=(double)data->svn; // svn
	ytime[i]=data->cputime;
	yheappeak[i]=data->heappeak;
	yheapleak[i]=data->heapleak;
	yheapalloc[i]=data->heapalloc;
	etime[i]=data->varTime;
	eheappeak[i]=data->varHeappeak;
	eheapalloc[i]=data->varHeapalloc;
	eheapleak[i]=data->varHeapleak; 
      }
      gErrorIgnoreLevel=3000; 
      sstm.str("");
      sstm<< "pt_" << string1 << hashVal << "_cputime.gif";
      imageName = new char[sstm.str().length() + 1];
      strcpy(imageName, sstm.str().c_str());           
      ytitle="CPU time [s]";
      saveGraph(nevent,x,ytime,etime,ytitle,imageName,progName,event->cputime);

      sstm.str("");
      sstm<< "pt_" << string1 << hashVal << "_heappeak.gif";
      imageName = new char[sstm.str().length() + 1];
      strcpy(imageName, sstm.str().c_str());
      ytitle="Peak heap usage [kB]";
      saveGraph(nevent,x,yheappeak,eheappeak,ytitle,imageName,progName,event->heappeak);

      sstm.str("");
      sstm<< "pt_" << string1 << hashVal << "_heapalloc.gif";
      imageName = new char[sstm.str().length() + 1];
      strcpy(imageName, sstm.str().c_str()); 
      ytitle="Size of total allocated memory [kB]";
      saveGraph(nevent,x,yheapalloc,eheapalloc,ytitle,imageName,progName,event->heapalloc);

      sstm.str("");
      sstm<< "pt_" << string1 << hashVal << "_heapleak.gif";
      imageName = new char[sstm.str().length() + 1];
      strcpy(imageName, sstm.str().c_str());
      ytitle="Size of memory leaks [kB]";
      saveGraph(nevent,x,yheapleak,eheapleak,ytitle,imageName,progName,event->heapleak);
      gErrorIgnoreLevel=0; 

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
      delete event;
      delete t;
      delete f;

      return outlier;
    } 
}


TTree* detectOutliers(TFile *f,TTree *t,pt_data *event)
{
  double memav,timeav,memstd,timestd,z,allocstd,allocmean,leakstd,leakmean;
  int timetest,heaptest,alloctest,leaktest,k;
  Int_t nevent,n;    
  pt_data *data;
  data=new pt_data();
  t->SetBranchAddress("event", &data); 
  nevent = t->GetEntries();
  timetest=heaptest=alloctest=leaktest=0;
  k=nevent-1;
  while (k>=0){ 
    t->GetEvent(k);
    if (data->outlier==0){ 
      n=data->testnumber+1;
      event->testnumber=data->testnumber+1;
      break;
    }
    k--;
  }

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
    
  if (n>1){    	  
    // calculate moving mean and stddev
    timeav=(n-1)*timeav+event->cputime;	  
    timestd=(n-1)*timestd+event->cputime*event->cputime;
    event->squareTime=timestd/(double)n;
    timestd=(n*timestd-timeav*timeav)/(double)(n*(n-1)); // variance
    timeav=timeav/(double)n; // mean
    if (timestd<0) timestd=0;	  
    timestd=sqrt(timestd); // std dev

    memav=(n-1)*memav+event->heappeak;
    memstd=(n-1)*memstd+event->heappeak*event->heappeak;
    event->squareHeappeak=memstd/(double)n;
    memstd=(n*memstd-memav*memav)/(double)(n*(n-1));	  
    memav=memav/(double)n;
    if (memstd<0) memstd=0;
    memstd=sqrt(memstd);

    allocmean=(n-1)*allocmean+event->heapalloc;
    allocstd=(n-1)*allocstd+event->heapalloc*event->heapalloc;
    event->squareHeapalloc=allocstd/(double)n;
    allocstd=(n*allocstd-allocmean*allocmean)/(double)(n*(n-1));	  
    allocmean=allocmean/(double)n;
    if (allocstd<0) allocstd=0;
    allocstd=sqrt(allocstd);

    leakmean=(n-1)*leakmean+event->heapleak;
    leakstd=(n-1)*leakstd+event->heapleak*event->heapleak;
    event->squareHeapleak=leakstd/(double)n;
    leakstd=(n*leakstd-leakmean*leakmean)/(double)(n*(n-1)); 	  
    leakmean=leakmean/(double)n;
    if (leakstd<0) leakstd=0;
    leakstd=sqrt(leakstd);

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

    event->outlier=timetest+heaptest+leaktest+alloctest;
  } //endif (n>1)	    

  event->meanTime=timeav;
  event->varTime=timestd;
  event->meanHeappeak=memav;
  event->varHeappeak=memstd;
  event->meanHeapalloc=allocmean;
  event->varHeapalloc=allocstd;
  event->meanHeapleak=leakmean;
  event->varHeapleak=leakstd;     
  t->SetBranchAddress("event", &event);	
  t->Fill();      
  f->Write("",TObject::kOverwrite);          	
  return t;   
}

TTree* createTree(TFile *f,pt_data *event)
{
  TTree *t = new TTree("PerftrackTree","Performance monitoring tree");	
  t->Branch("event", &event);
  t->Fill();      
  f->Write("",TObject::kOverwrite); 
  return t;
}

TTree* deleteEntries(TTree *t)
{

  Int_t i,j,k,nevent;
  double memav,timeav,memstd,timestd,allocstd,allocmean,leakstd,leakmean;
  pt_data *data;
  data=new pt_data();
  t->SetBranchAddress("event", &data);
  TTree *newT = t->CloneTree(0);
  j=k=1;
  nevent = t->GetEntries();
  memav=timeav=memstd=timestd=allocstd=allocmean=leakstd=leakmean=0;
  for (i=0;i<nevent;i++){
    t->GetEntry(i);
    if (data->outlier>0){ 
      newT->Fill();
      continue;
    }
    if (j % 2 == 0){ 
      timeav=(k-1)*timeav+data->cputime;
      timestd=(k-1)*timestd+data->cputime*data->cputime; // square of sum
      memav=(k-1)*memav+data->heappeak; 
      memstd=(k-1)*memstd+data->heappeak*data->heappeak;
      allocmean=(k-1)*allocmean+data->heapalloc;
      allocstd=(k-1)*allocstd+data->heapalloc*data->heapalloc;	    
      leakmean=(k-1)*leakmean+data->heapleak;
      leakstd=(k-1)*leakstd+data->heapleak*data->heapleak;
	      
      if ((k*timestd-timeav*timeav)/(double)(k*(k-1))<=0 || k==1) data->varTime=0;
      else data->varTime=sqrt((k*timestd-timeav*timeav)/(double)(k*(k-1)));
      timeav=timeav/(double)k;
      timestd=timestd/(double)k;
      data->meanTime=timeav;
      data->squareTime=timestd;

      if ((k*memstd-memav*memav)/(double)(k*(k-1))<=0 || k==1) data->varHeappeak=0;
      else data->varHeappeak=sqrt((k*memstd-memav*memav)/(double)(k*(k-1)));
      memav=memav/(double)k;	
      memstd=memstd/(double)k;
      data->meanHeappeak=memav; 
      data->squareHeappeak=memstd;

      if ((k*allocstd-allocmean*allocmean)/(double)(k*(k-1))<=0 || k==1) data->varHeapalloc=0; 
      else data->varHeapalloc=sqrt((k*allocstd-allocmean*allocmean)/(double)(k*(k-1)));
      allocmean=allocmean/(double)k;	
      allocstd=allocstd/(double)k;
      data->squareHeapalloc=allocstd;
      data->meanHeapalloc=allocmean;
	   
      if ((k*leakstd-leakmean*leakmean)/(double)(k*(k-1))<=0 || k==1) data->varHeapleak=0; 
      else data->varHeapleak=sqrt((k*leakstd-leakmean*leakmean)/(double)(k*(k-1)));
      leakmean=leakmean/(double)k;	
      leakstd=leakstd/(double)k;
      data->squareHeapleak=leakstd;
      data->meanHeapleak=leakmean; 

      data->testnumber=k;
	
      newT->Fill();      
      k++; 
    }
    j++;
  }
  //t = newT->CloneTree();
  return newT;
}

void saveGraph(Int_t n,double *x,double * y,double * e,const char *ytitle,char * imageName,char * progName,double v)
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
  
  /*cout<<"Min "<<g->GetXMin()<<" Max "<<g->GetMaximum()<<endl; 
    if (g->GetMinimum()>v*0.99) g->SetMinimum(v*0.99);
    if (g->GetMaximum()<v*1.01) g->SetMaximum(v*1.01);
    cout<<"Min "<<g->GetMinimum()<<" Max "<<g->GetMaximum()<<endl; */
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
