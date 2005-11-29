# 
# $Id: Makefile.am,v 1.4 2004/11/12 16:29:26 pfkeb Exp $
#
# Author: Paul_Kunz@slac.stanford.edu
#

# The following is set; otherwise, it would have to follow GNU
# conventions.
AUTOMAKE_OPTIONS = foreign

EXTRA_DIST =					\
	Doxyfile.in

docs:
	if test -d html; then \
	(cd html; rm -rf *.html ) \
	fi
	doxygen Doxyfile
	mv Minuit.tag html/.

clean-local:
	if test -d html; then \
	(cd html; rm -rf *.html *.gif *.css *.dot *.png formula.repository; \
	rm -rf *.md5 *.map *.tag installdox ) \
	fi
	if test -d latex; then \
	( cd latex; rm -rf *tex *.eps *.sty ) \
	fi

