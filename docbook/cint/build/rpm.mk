.PHONY : rpm rpminstall

ECHO = echo


rpmdirs = RPM RPM/BUILD RPM/SOURCES RPM/SRPMS RPM/RPMS RPM/SPECS
$(rpmdirs) :
	mkdir $@

origtgz_rpm = $(shell pwd)/RPM/SOURCES/cint-${G__CFG_CINTVERSION}.tar.gz
$(origtgz_rpm) : $(rpmdirs)
	@echo Generating original source tarball: $(origtgz_rpm)
	@tmp=`mktemp -d /tmp/cint_rpm.XXXXXX`; \
	  cp -r . $${tmp}/cint-${G__CFG_CINTVERSION}; \
	  cd      $${tmp}/cint-${G__CFG_CINTVERSION} ; \
	  make distclean; rm -rf debian RPM rpm.spec; \
	  cd ..; tar cvzf $@ cint-${G__CFG_CINTVERSION}; \

# find out, which package provides g++
gpp_pkg :=
ifeq ($(shell rpm -q --whatprovides `which g++ 2>/dev/null` >/dev/null 2>&1; echo $$?),0)
  gpp_pkg := $(strip $(gpp_pkg))
  gpp_pkg := $(shell rpm -q --whatprovides `which g++ 2>/dev/null`)
  gpp_pkg := $(shell echo $(gpp_pkg) | sed 's/-[^-]*-[^-]*$$//g') # strip off version numbers
endif

# find out, which package provides G__CFG_READLINELIB
readline_pkg :=
ifeq ($(shell rpm -q --whatprovides $(G__CFG_READLINELIB) >/dev/null 2>&1; echo $$?),0)
  readline_pkg := $(strip $(readline_pkg))
  readline_pkg := $(shell rpm -q --whatprovides $(G__CFG_READLINELIB))
  readline_pkg := $(shell echo $(readline_pkg) | sed 's/-[^-]*-[^-]*$$//g') # strip off version numbers
endif

requires := $(gpp_pkg)
ifneq ($(readline_pkg),)
  ifeq ($(requires),)
    requires += $(readline_pkg)
  else
    requires += ,$(readline_pkg)
  endif
endif
ifneq ($(requires),)
  requires := Requires: $(requires)
endif

build_requires := $(gpp_pkg)
ifneq ($(readline_pkg),)
  ifeq ($(build_requires),)
    build_requires += $(readline_pkg)
  else
    build_requires += ,$(readline_pkg)
  endif
endif
ifneq ($(build_requires),)
  build_requires := BuildRequires: $(build_requires)
endif

confflags := $(shell cat config.status)
confflags := $(subst     --with-prefix, ,$(confflags))
confflags := $(patsubst  --prefix=%, ,$(confflags))
confflags += --with-prefix --prefix=/usr
CONFIGCMD := ./configure $(confflags)

cintspec = RPM/SPECS/cint.spec
$(cintspec) : Makefile $(rpmdirs)
	@$(ECHO) Creating $@
	@$(ECHO) %define _topdir $(shell pwd)/RPM > $@
	@$(ECHO) Summary: CINT C/C++ interpreter >> $@
	@$(ECHO) Url: http://root.cern.ch/twiki/bin/view/ROOT/CINT >>$@
	@$(ECHO) Name: cint >> $@
	@$(ECHO) Version: $(G__CFG_CINTVERSION) >> $@
	@$(ECHO) Release: 1 >> $@
	@$(ECHO) $(requires)       >> $@
	@$(ECHO) $(build_requires) >> $@
	@$(ECHO) Source0: %{name}-%{version}.tar.gz >> $@
	@$(ECHO) License: X11/MIT >> $@
	@$(ECHO) Group: Development >> $@
	@$(ECHO) BuildRoot: %{_builddir}/%{name}-root >> $@
	@$(ECHO) %description >> $@
	@$(ECHO) 'CINT is a C/C++ interpreter aimed at processing C/C++ scripts.' >> $@
	@$(ECHO) 'Scripts are programs performing specific tasks. Generally '>> $@
	@$(ECHO) 'execution time is not critical, but rapid development is.' >> $@
	@$(ECHO) 'Using an interpreter the compile and link cycle is dramatically' >> $@
	@$(ECHO) 'reduced facilitating rapid development. CINT makes C/C++ ' >> $@
	@$(ECHO) 'programming enjoyable even for part-time programmers.' >> $@
	@$(ECHO) 'CINT is written in C++ itself (slightly less than 400,000'>> $@
	@$(ECHO) 'lines of code). It is used in production by several companies'>> $@
	@$(ECHO) 'in the banking, integrated devices, and even gaming environment,'>> $@
	@$(ECHO) 'and of course by ROOT, making it the default interpreter for'>>$@
	@$(ECHO) 'a large number of high energy physicists all over the world' >> $@
	@$(ECHO) %prep >> $@
	@$(ECHO) %setup -q >> $@
	@$(ECHO) %build >> $@
	@$(ECHO) $(CONFIGCMD) >> $@
	@$(ECHO) make >> $@
	@$(ECHO) %install >> $@
	@$(ECHO) 'rm -rf $$RPM_BUILD_ROOT' >> $@
	@$(ECHO) 'make DESTDIR=$$RPM_BUILD_ROOT install' >> $@
	@$(ECHO) %clean >> $@
	@$(ECHO) 'rm -rf $$RPM_BUILD_ROOT' >> $@
	@$(ECHO) %post >>$@
	@$(ECHO) 'ldconfig' >>$@
#	@$(ECHO) %preun >>$@
#	@$(ECHO) '' >> $@
	@$(ECHO) %files >> $@
	@$(ECHO) "%defattr(-,root,root)"  >>$@
	@$(ECHO) '$(G__CFG_BINDIR)'  >>$@
	@$(ECHO) '$(G__CFG_MANDIR)/man1'  >>$@
	@$(ECHO) '$(G__CFG_LIBDIR)'  >>$@
	@$(ECHO) '$(G__CFG_INCLUDEDIRCINT)'  >>$@
	@$(ECHO) '$(G__CFG_DATADIRCINT)'  >>$@


build_arch = $(shell rpmbuild --showrc | awk '$$1=="build" && $$2=="arch" && $$3==":" { print $$4; exit; }')
RPM=RPM/RPMS/$(build_arch)/cint-${G__CFG_CINTVERSION}-1.$(build_arch).rpm

rpm : $(RPM)
rpminstall : rpm
	rpm -q cint > /dev/null 2>&1 && rpm -U $(RPM) || rpm -i $(RPM) 


$(RPM) : $(cintspec) $(origtgz_rpm) 
	rpmbuild  -ba $<
	@$(ECHO) '###################################################'
	@$(ECHO) '#                                                 #' 
	@$(ECHO) '# Package files created in RPM/RPMS and RPM/SRPMS #'
	@$(ECHO) '#                                                 #' 
	@$(ECHO) '###################################################'
