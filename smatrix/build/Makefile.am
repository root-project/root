# @(#)root/mathcore:$Id$
# Author: L. Moneta    2005  
#
## Process this file with automake to produce Makefile.in .
#
#
# The following is set, otherwise it would follow GNU conventions.
AUTOMAKE_OPTIONS = foreign



# A trick to add options to the aclocal command.
ACLOCAL = aclocal -I config/m4

SUBDIRS =   inc doc test
#tests 

EXTRA_DIST = autogen config





## Make the documentation in the build directory
docs:
	( cd doc; $(MAKE) docs )
