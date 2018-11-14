% ROOT Version 6.18 Release Notes
% 2018-11-14
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.18/00 is scheduled for release in May, 2019.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Bertrand Bellenot, CERN/SFT,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/SFT,\
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Sergey Linev, GSI,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Axel Naumann, CERN/SFT,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Matevz Tadel, UCSD/CMS,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/Atlas,

## Deprecation and Removal

### THttpServer classes

Following deprecated methods were removed:

   * Bool_t THttpServer::SubmitHttp(THttpCallArg *arg, Bool_t can_run_immediately = kFALSE, Bool_t ownership = kFALSE);
   * Bool_t THttpServer::ExecuteHttp(THttpCallArg *arg)
   * Bool_t TRootSniffer::Produce(const char *path, const char *file, const char *options, void *&ptr, Long_t &length, TString &str);
   * TString THttpCallArg::GetPostDataAsString();
   * void THttpCallArg::FillHttpHeader(TString &buf, const char *header = nullptr);
   * void THttpCallArg::SetBinData(void *data, Long_t length);
   
   
Methods should be replaced by equivalent methods with other signature:

   * Bool_t THttpServer::SubmitHttp(std::shared_ptr<THttpCallArg> arg, Bool_t can_run_immediately = kFALSE);
   * Bool_t THttpServer::ExecuteHttp(std::shared_ptr<THttpCallArg> arg);
   * Bool_t TRootSniffer::Produce(const std::string &path, const std::string &file, const std::string &options, std::string &res);   
   * const void *THttpCallArg::GetPostData() const;
   * Long_t THttpCallArg::GetPostDataLength() const;
   * std::string THttpCallArg::FillHttpHeader(const char *header = nullptr);
   * void THttpCallArg::SetContent(std::string &&cont);


## Core Libraries


## I/O Libraries


## TTree Libraries


## Histogram Libraries


## Math Libraries


## RooFit Libraries


## 2D Graphics Libraries


## 3D Graphics Libraries


## Geometry Libraries


## Database Libraries


## Networking Libraries


## GUI Libraries


## Montecarlo Libraries


## PROOF Libraries


## Language Bindings


## JavaScript ROOT


## Tutorials


## Class Reference Guide


## Build, Configuration and Testing Infrastructure


