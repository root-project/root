## Process this file with automake to produce Makefile.in .
#
# Author: Paul_Kunz@slac.stanford.edu
#

# The following is set; otherwise, it would follow GNU conventions.
# AUTOMAKE_OPTIONS = foreign


SUBDIRS =			\
		MnSim		\
		MnTutorial	

# For platforms where automake dependency generation doesn't work.
depend: 
	for dir in $(SUBDIRS); \
	do ( test -d $$dir && cd $$dir && $(GMAKE) depend ) done


