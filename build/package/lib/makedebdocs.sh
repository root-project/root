#!/bin/sh -e 
#
# $Id: makedebdocs.sh,v 1.1 2001/04/23 14:11:47 rdm Exp $
#
# Writes a documentation file entry to debian/root-<pkg>.docs
#
tgtdir=$1 ; shift 
cmndir=$1 ; shift 
prefix=$1 ; shift
etcdir=$1 ; shift
docdir=$1 ; shift 
pkg=$1

# Make sure we get a fresh file 
rm -f ${tgtdir}/${pkg}.docs

if test ! -f ${tgtdir}/${pkg}.README.Debian ; then 
    cat > ${tgtdir}/${pkg}.README.Debian <<EOF
Debian GNU/[Linux|Hurd] package $pkg
--------------------------------------------

There's no documentation distributed with this package, but the 
package root-doc contains a test suite and a directory of tutorial 
scripts; look in the directory $docdir.  
Also, packages that provides programs also provides a short man(1)
page for those programs.   

Extensive documentation, mailing list archive, and so on is available
from the ROOT web-site at:

  http://root.cern.ch 

 -- Christian Holm <cholm@nbi.dk>  Wed,  9 Jan 2002 04:08:19 +0100
EOF
fi 
#  # See if file exists in common directory 
#  if [ -f $cmndir/$pkg.docs ] ; then 
#      # Prepend each line with a '/', and ignore comment lines
#      grep -v "^#" $cmndir/$pkg.docs | \
#  	sed -e "s,@etcdir@,${etcdir},g" \
#  	    -e "s,@prefix@,${prefix},g" \
#  	    -e "s,@docdir@,${docdir},g" \
#  		> ${tgtdir}/${pkg}.docs
#  fi 

#
# $Log: makedebdocs.sh,v $
# Revision 1.1  2001/04/23 14:11:47  rdm
# part of the debian and redhat build system.
#
#
