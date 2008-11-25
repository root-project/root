## Process this file with automake to produce Makefile.in
# 
# Author: Paul_Kunz@slac.stanford.edu
#

# The following is set, otherwise it would have to follow GNU conventions.
AUTOMAKE_OPTIONS = foreign

# A trick to add options to the aclocal command.
ACLOCAL = aclocal -I config/m4

## The subdirectories.  
SUBDIRS =					\
	inc					\
	src					\
	test					\
	doc					


#EXTRA_DIST = BuildFile makefile.msc

# The following is for platforms where automake dependencies don't work.
depend:
	for dir in $(SUBDIRS); \
	do ( test -d $$dir && cd $$dir && $(GMAKE) depend ) done

## Make the documentation in the build directory
docs:
	( cd doc; $(MAKE) docs )

# extra clean targets for Sun CC compiler and generated distribution file
clean-local:
	-rm -rf SunWS_cache
	-rm -rf *.tar.gz

