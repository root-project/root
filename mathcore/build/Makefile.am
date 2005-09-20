# @(#)root/mathcore:$Name:  $:$Id: Makefile.am,v 1.1 2005/09/18 17:33:47 brun Exp $
# Author: L. Moneta    2005  
#
## Process this file with automake to produce Makefile.in .
#
#
# The following is set, otherwise it would follow GNU conventions.
AUTOMAKE_OPTIONS = foreign



# A trick to add options to the aclocal command.
ACLOCAL = aclocal -I config/m4

SUBDIRS =   src inc doc test
#tests 

EXTRA_DIST = autogen config





## Make the documentation in the build directory
docs:
	( cd doc; $(MAKE) docs )
