################################################################################
##                        Makefile.am for xrootd
##
## (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University
##       All Rights Reserved. See XrdInfo.cc for complete License Terms
##   Produced by Andrew Hanushevsky for Stanford University under contract
##              DE-AC03-76-SFO0515 with the Department of Energy
##
##
## Version info: $Id: Makefile.am,v 1.20 2010/08/05 15:26:29 feichtinger Exp $
## Checked in by $Author: feichtinger $
################################################################################


noinst_HEADERS = XrdVersion.hh XProtocol/YProtocol.hh XProtocol/XProtocol.hh \
                 XProtocol/XPtypes.hh

#######################################################################
## Option conditionals
## Note: specifying optional directories using automake conditionals
##       guarantees that the 'make dist' command still yields the complete
##       tarball  
if ENABLE_PWD
XRDSUT = XrdSut
XRDSECPWD = XrdSecpwd
XRDCRYPTO = XrdCrypto
endif

if ENABLE_OPENSSL
XRDSECPWD = XrdSecpwd
XRDSECSSS = XrdSecsss
endif

if ENABLE_KRB4
XRDSECKRB4 = XrdSeckrb4
endif

if ENABLE_KRB5
XRDSECKRB5 = XrdSeckrb5
endif

if ENABLE_GSI
XRDSECGSI = XrdSecgsi
endif

if ENABLE_MON
XRDMON = XrdMon
endif


if ENABLE_POSIX
XRDPOSIX = XrdPosix
XRDPSS = XrdPss
XRDCNS = XrdCns
endif

if ENABLE_APPS
XRDAPPS=XrdApps
endif


#######################################################################

SUBDIRS = XrdSys \
          XrdOuc \
          XrdNet \
          XrdSec \
          $(XRDSUT) \
          $(XRDCRYPTO) \
          $(XRDSECPWD) \
          $(XRDSECKRB4) \
          $(XRDSECKRB5) \
          $(XRDSECGSI) \
          $(XRDSECSSS) \
          XrdSecunix \
          XrdSfs \
          XrdAcc \
          XrdBwm \
          XrdOss \
          Xrd \
          XrdFrm \
          XrdCms \
          XrdOfs \
          XrdRootd \
          XrdXrootd \
          XrdClient \
          $(XRDPOSIX) \
          $(XRDAPPS) \
          $(XRDPSS) \
          $(XRDCNS) \
          $(XRDMON)




