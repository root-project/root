# Makefile to create all the files debian/* necessary for
# debian package creation, and the package itself.

# Read the configure flags from config.status. Create a new
# file debian/confflags with the same flags. This file can
# be edited by users who later want to get the deb-src 
# package, and rebuild it with different configure flags.
# The --with-prefix and --prefix=/usr flags are removed
# from debian/confflags so that users do not believe
# that they can modify it (these flags are added in debian/control)

.PHONY : deb debinstall

ECHO = /bin/echo

cint_signature_file=~/.cint-signature
cint_signature=`cat $(cint_signature_file)`
$(cint_signature_file) :
	@$(ECHO) 'We need your name and email address, to store them'; \
	  $(ECHO) 'in the created package. The package also needs to be'; \
	  $(ECHO) 'signed, so make sure that you have a gpg key corresponding'; \
	  $(ECHO) 'to the name/email, that you will specify below'; \
	  $(ECHO) '(gpg --gen-key is the command which generates a key)'; \
	  $(ECHO) ; \
	  name_guess=`awk -F: '$$1=="'"$$USER"'" {print $$5}' /etc/passwd | sed 's/,.*//g'`; \
	  $(ECHO) -n "What is your name [$${name_guess}]: "; \
	  read name; \
	  if [ "$name" = "" ] ; then name="$$name_guess"; fi; \
	  $(ECHO) -n $$name > $@; \
	  $(ECHO) -n 'What is your email address: '; \
	  read email; \
	  $(ECHO) " <$$email>" >> $@; \
	  $(ECHO) ; \
	  $(ECHO) "Great! The file '$@' will store this info. Edit this file" ;\
	  $(ECHO) "If you want to make any changes" ; \
	  $(ECHO) ; \
	  sleep 5s

deb_arch = $(shell dpkg-architecture -qDEB_HOST_ARCH)
pkgfilename = cint_$(G__CFG_CINTVERSION)-1_$(deb_arch).deb 

pkg  = debian/package-files/$(pkgfilename)

CONFIGCMD := ./configure
#--host=$$(DEB_HOST_GNU_TYPE) --build=$$(DEB_BUILD_GNU_TYPE)

# make deb          -- creates the .deb package in the parent directory
deb : $(pkg)

# make debinstall   -- installs the created .deb package file to your system
debinstall : $(pkg)
	dpkg -i $(pkg)

# the original source tgz must be in the parent directory:
origtgz_deb = $(shell pwd | sed 's|/[^/]*$$||g')/cint_${G__CFG_CINTVERSION}.orig.tar.gz

# This is the default value for the -i option of dpkg-source, + some other extensions
# (.png, .o, .exe, .a, 

ignore = '(?:^|/).*~$$|(?:^|/)\.\#.*$$|(?:^|/)\..*\.swp$$|(?:^|/),,.*(?:$$|/.*$$)|(?:^|/)(?:DEADJOE|\.cvsignore|\.arch-inventory|\.bzrignore|\.gitignore)$$|(?:^|/)(?:CVS|RCS|\.deps|\{arch\}|\.arch-ids|\.svn|\.hg|_darcs|\.git|\.shelf|\.bzr(?:\.backup|tags)?)(?:$$|/.*$$)|.*\.png$$|.*\.o$$|.*\.exe$$|.*\.a$$|/config\..*$$|.*\.dvi$$|.*\.gz$$|.*\.deb$$|.*\.rpm$$|.*build-stamp$$|.*\.ps$$|rmkdepend|mktypes|.*\.dll$$|.*\.d'

$(pkg) : $(origtgz_deb) debian/control debian/copyright debian/cint.dirs debian/cint.docs debian/changelog debian/rules debian/compat debian/cint.postinst debian/cint.prerm debian/confflags debian/README.txt
	@if [ "x$(G__CFG_PREFIX)" = "x" ] ; then \
	  echo configure was not called with the --with-prefix --prefix=/usr options; \
	  echo therefore we clean up the source directory, and reconfigure; \
	  make distclean; \
	fi
	@$(ECHO) '###  Creating the package files'
	rm -f config.status
#	make distclean
	@wd=`pwd | sed 's|^.*/||g'`; \
	if [ "$$wd" != cint-${G__CFG_CINTVERSION} ] ; then \
	  cd ..; \
	  if [ -e cint-${G__CFG_CINTVERSION} ] ; then \
	    if [ -h cint-${G__CFG_CINTVERSION} ] ; then \
	      rm -f cint-${G__CFG_CINTVERSION}; \
	    else \
	      echo ; \
	      echo 'cint-${G__CFG_CINTVERSION} already exists. I need this to build the' ;\
	      echo 'deb package. (This would be a symlink to the current directory)' ;\
	      echo 'Should I remove it? [y/n]'; \
	      read answer; \
	      if [ "$$answer" != "y" ] ; then exit 1; fi; \
	      rm -f cint-${G__CFG_CINTVERSION}; \
	    fi; \
	  fi; \
	  ln -s $$wd cint-${G__CFG_CINTVERSION}; \
	fi; cd cint-${G__CFG_CINTVERSION}; dpkg-buildpackage -rfakeroot -W -i$(ignore)
	@if [ ! -d debian/package-files ] ; then mkdir debian/package-files; fi
	@mv ../$(pkgfilename) $@
	@mv ../cint_$(G__CFG_CINTVERSION)-1_$(deb_arch).changes debian/package-files/
	@mv ../cint_$(G__CFG_CINTVERSION)-1.dsc                 debian/package-files/
	@mv ../cint_$(G__CFG_CINTVERSION)-1.diff.gz             debian/package-files/
	@$(ECHO) '###################################################'
	@$(ECHO) '#                                                 #' 
	@$(ECHO) '#  Package files created in debian/package-files  #'
	@$(ECHO) '#                                                 #' 
	@$(ECHO) '###################################################'



$(origtgz_deb) : 
	@$(ECHO) '###  Creating original source tarball: $(origtgz_deb)'
	@ make distclean;
	@ pardir=`pwd | sed 's|/[^/]*$$||g'` ; \
	  tmp=`mktemp -d /tmp/cint_deb.XXXXXX`; \
	  cp -r . $${tmp}/cint-${G__CFG_CINTVERSION}; \
	  cd $${tmp}/cint-${G__CFG_CINTVERSION} ; \
	  rm -rf debian; \
	  cd ..; tar cvzf $@ cint-${G__CFG_CINTVERSION}; \
	  rm -rf $${tmp}

debian/cint.prerm : build/deb.mk
	@$(ECHO) '#! /bin/sh' > $@
	@$(ECHO) '# prerm script for root-cint' >> $@
	@$(ECHO) 'set -e' >> $@
	@$(ECHO) 'exit 0' >> $@
	@chmod +x $@

debian/cint.postinst : build/deb.mk
	@$(ECHO) '#! /bin/sh' > $@
	@$(ECHO) 'set -e' >> $@
	@$(ECHO) 'case "$$1" in' >> $@
	@$(ECHO) 'configure)' >> $@
	@$(ECHO) '# Nothing to be done here' >> $@
	@$(ECHO) ';;' >> $@
	@$(ECHO) 'abort-upgrade|abort-remove|abort-deconfigure)' >> $@
	@$(ECHO) '# Nothing to be done here' >> $@
	@$(ECHO) ';;' >> $@
	@$(ECHO) '*)' >> $@
	@$(ECHO) 'echo "postinst called with unknown argument \"$$1\"" >&2' >> $@
	@$(ECHO) 'exit 0' >> $@
	@$(ECHO) ';;' >> $@
	@$(ECHO) 'esac' >> $@
	@$(ECHO) 'exit 0' >> $@
	@chmod +x $@

debian/compat : build/deb.mk
	@if [ ! -d debian ] ; then mkdir debian; fi
	@$(ECHO) '###  Creating $@'
	@$(ECHO) 5 > $@

# try to cleverly collect the packages needed for the build, and for the usage
build_needed_packages=$(shell unalias -a; dpkg -S `which g++` | awk -F: '{a[$$1]=1} END{c=0; for(i in a) { if(c!=0) print ", "; print i; ++c; }}') 
build_needed_packages += , fakeroot, libreadline-dev

needed_packages=$(shell unalias -a; dpkg -S `which g++` |  awk -F: '{a[$$1]=1} END{c=0; for(i in a) { if(c!=0) print ", "; print i; ++c; }}')
needed_packages += , libreadline-dev

debian/control : $(cint_signature_file) build/deb.mk
	@if [ ! -d debian ] ; then mkdir debian; fi
	@$(ECHO) '###  Creating $@'
	@$(ECHO) 'Source: cint' > $@
	@$(ECHO) 'Section: development' >> $@
	@$(ECHO) 'Priority: optional' >> $@
	@$(ECHO) "Maintainer: $(cint_signature)" >> $@
	@$(ECHO) 'Build-Depends: debhelper (>= 5), $(build_needed_packages)' >> $@
	@$(ECHO) 'Standards-Version: 3.7.2' >> $@
	@$(ECHO) '' >> $@
	@$(ECHO) 'Package: cint' >> $@
	@$(ECHO) 'Architecture: any' >> $@
	@$(ECHO) 'Depends: $${shlibs:Depends}, $(needed_packages)' >> $@
	@$(ECHO) 'Description: CINT C/C++ Interpreter' >> $@

debian/copyright : $(cint_signature_file) build/deb.mk
	@if [ ! -d debian ] ; then mkdir debian; fi
	@$(ECHO) '###  Creating $@'
	@$(ECHO) "This package was created by $(cint_signature) on" > $@
	@$(ECHO) "`date -R`" >> $@
	@$(ECHO) '' >> $@
	@$(ECHO) 'Downloadable from: http://root.cern.ch/twiki/bin/view/ROOT/CINT' >> $@
	@$(ECHO) '' >> $@
	@$(ECHO) 'Upstream Author: Masaharu Goto & others' >> $@
	@$(ECHO) '' >> $@
	@$(ECHO) 'Copyright: 2006 Masaharu Goto (gotom@hanno.jp)' >> $@
	@$(ECHO) '' >> $@
	@$(ECHO) 'License: X11/MIT' >> $@
	@$(ECHO) '' >> $@
	@$(ECHO) 'The Debian packaging is (C) 2006, $(cint_signature) and' >> $@
	@$(ECHO) "is licensed under the GPL, see '/usr/share/common-licenses/GPL'." >> $@

debian/cint.dirs : build/deb.mk
	@if [ ! -d debian ] ; then mkdir debian; fi
	@$(ECHO) '###  Creating $@'
	@$(ECHO) usr/bin > $@
	@$(ECHO) usr/include >> $@
	@$(ECHO) usr/lib >> $@
	@$(ECHO) usr/share/cint >> $@
	@$(ECHO) usr/share/man/man1 >> $@

debian/cint.docs : build/deb.mk
	@if [ ! -d debian ] ; then mkdir debian; fi
	@$(ECHO) '###  Creating $@'
	@$(ECHO) README.txt > $@
	@$(ECHO) RELNOTE.txt   >> $@
	@$(ECHO) FAQ.txt   >> $@
	@$(ECHO) COPYING   >> $@

debian/changelog : $(cint_signature_file) build/deb.mk
	@if [ ! -d debian ] ; then mkdir debian; fi
	@$(ECHO) '###  Creating $@'
	@$(ECHO) 'cint (${G__CFG_CINTVERSION}-1) unstable; urgency=low' > $@
	@$(ECHO) '' >> $@
	@$(ECHO) '  * New upstream release' >> $@
	@$(ECHO) '' >> $@
	@$(ECHO) " -- $(cint_signature)  `date -R`" >> $@

debian/rules : build/deb.mk
	@if [ ! -d debian ] ; then mkdir debian; fi
	@$(ECHO) '###  Creating $@'
	@$(ECHO) '#!'`which make`' -f' > $@
	@$(ECHO) '# -*- makefile -*-' >> $@
	@$(ECHO) '' >> $@
	@$(ECHO) '# Uncomment this to turn on verbose mode.' >> $@
	@$(ECHO) '#export DH_VERBOSE=1' >> $@
	@$(ECHO) '' >> $@
	@$(ECHO) '# These are used for cross-compiling and for saving the configure script' >> $@
	@$(ECHO) '# from having to guess our platform (since we know it already)' >> $@
	@$(ECHO) 'DEB_HOST_GNU_TYPE   ?= $$(shell dpkg-architecture -qDEB_HOST_GNU_TYPE)' >> $@
	@$(ECHO) 'DEB_BUILD_GNU_TYPE  ?= $$(shell dpkg-architecture -qDEB_BUILD_GNU_TYPE)' >> $@
	@$(ECHO) '#CFLAGS = -Wall' >> $@
	@$(ECHO) 'ifneq (,$$(findstring noopt,$$(DEB_BUILD_OPTIONS)))' >> $@
	@$(ECHO) '        CFLAGS += -O0' >> $@
	@$(ECHO) 'else' >> $@
	@$(ECHO) '        CFLAGS += -O2' >> $@
	@$(ECHO) 'endif' >> $@
	@$(ECHO) '' >> $@
	@$(ECHO) 'include debian/confflags' >> $@
	@$(ECHO) '# Make sure the user rebuilding this package has not added ' >> $@
	@$(ECHO) '# --with-prefix --prefix=/some/other/location flags ' >> $@
	@$(ECHO) 'confflags := $$(subst   --with-prefix, ,$$(confflags))' >> $@
	@$(ECHO) 'confflags := $$(patsubst   --prefix=%, ,$$(confflags))' >> $@
	@$(ECHO) 'config.status: configure' >> $@
	@$(ECHO) -e '\tdh_testdir' >> $@
	@$(ECHO) -e '\t$(CONFIGCMD) $$(confflags) --with-prefix --prefix=/usr' >> $@
	@$(ECHO) '' >> $@
	@$(ECHO) 'build: build-stamp' >> $@
	@$(ECHO) 'build-stamp:  config.status' >> $@
	@$(ECHO) -e '\tdh_testdir' >> $@
	@$(ECHO) -e '\t$$(MAKE)' >> $@
	@$(ECHO) -e '\ttouch $$@' >> $@
	@$(ECHO) '' >> $@
	@$(ECHO) 'clean:' >> $@
	@$(ECHO) -e '\tdh_testdir' >> $@
	@$(ECHO) -e '\tdh_testroot' >> $@
	@$(ECHO) -e '\trm -f build-stamp' >> $@
#	@$(ECHO) -e '\t-$$(MAKE) distclean' >> $@
	@$(ECHO) 'ifneq "$$(wildcard /usr/share/misc/config.sub)" ""' >> $@
	@$(ECHO) -e '\tcp -f /usr/share/misc/config.sub config.sub' >> $@
	@$(ECHO) 'endif' >> $@
	@$(ECHO) 'ifneq "$$(wildcard /usr/share/misc/config.guess)" ""' >> $@
	@$(ECHO) -e '\tcp -f /usr/share/misc/config.guess config.guess' >> $@
	@$(ECHO) 'endif' >> $@
	@$(ECHO) -e '\tdh_clean' >> $@
	@$(ECHO) '' >> $@
	@$(ECHO) 'install: build' >> $@
	@$(ECHO) -e '\tdh_testdir' >> $@
	@$(ECHO) -e '\tdh_testroot' >> $@
	@$(ECHO) -e '\tdh_clean -k' >> $@
	@$(ECHO) -e '\tdh_installdirs' >> $@
	@$(ECHO) -e '\t$$(MAKE) install DESTDIR=$$(CURDIR)/debian/cint' >> $@
	@$(ECHO) '' >> $@
	@$(ECHO) 'binary-indep: build install' >> $@
	@$(ECHO) 'binary-arch: build install' >> $@
	@$(ECHO) -e '\tdh_testdir' >> $@
	@$(ECHO) -e '\tdh_testroot' >> $@
	@$(ECHO) -e '\tdh_installchangelogs' >> $@
	@$(ECHO) -e '\tdh_link' >> $@
	@$(ECHO) -e '\tdh_compress' >> $@
	@$(ECHO) -e '\tdh_fixperms' >> $@
	@$(ECHO) -e '\tdh_installdeb' >> $@
	@$(ECHO) -e '\tdh_shlibdeps' >> $@
	@$(ECHO) -e '\tdh_gencontrol' >> $@
	@$(ECHO) -e '\tdh_md5sums' >> $@
	@$(ECHO) -e '\tdh_builddeb' >> $@
	@$(ECHO) '' >> $@
	@$(ECHO) 'binary: binary-indep binary-arch' >> $@
	@$(ECHO) '.PHONY: build clean binary-indep binary-arch binary install' >> $@
	@chmod +x $@

confflags := $(shell cat config.status)
confflags := $(subst     --with-prefix, ,$(confflags))
confflags := $(patsubst  --prefix=%, ,$(confflags))
debian/confflags : build/deb.mk config.status
	@$(ECHO) '# This file contains the configure flags, which were used' >> $@
	@$(ECHO) '# to build the debian package. Feel free to modify them, and' >> $@
	@$(ECHO) '# then recompile the package by debuild -us -uc' >> $@
	@$(ECHO) '' >> $@
	@$(ECHO) 'confflags := $(confflags)' >> $@

debian/README.txt : build/deb.mk
	@$(ECHO) 'CINT has a built-in debian packaging scheme by calling "make deb"' >> $@
	@$(ECHO) 'If you read these lines, then this debian package was most probably' >> $@
	@$(ECHO) 'created by this scheme (or otherwise somebody faked this file... :-)' >> $@
	@$(ECHO) '' >> $@
	@$(ECHO) 'The idea is: ' >> $@
	@$(ECHO) '1) Download the official CINT source code, which has built-in' >> $@
	@$(ECHO) '   possibility to build a debian package' >> $@
	@$(ECHO) '2) Run ./configure [flags], with the flags you want. ' >> $@
	@$(ECHO) '   The flags are written to config.status so that the packaging' >> $@
	@$(ECHO) '   can make use of it, and build the package with these flags' >> $@
	@$(ECHO) '3) Run "make deb", which creates the debian/ subdirectory with' >> $@
	@$(ECHO) '   all the necessary control files, etc. There will be also a' >> $@
	@$(ECHO) '   replica of the confflags file, so that the debian source package' >> $@
	@$(ECHO) '   also contains the configuration flags in an easily editable' >> $@
	@$(ECHO) '   way, so that further rebuilding volunteers can easily change them' >> $@
