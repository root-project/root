#!/bin/sh -e 
#
# $Id: makedebcontrol.sh,v 1.1 2001/04/23 14:11:47 rdm Exp $
#
# Writes a control file entry debian/control 
#
tgtdir=$1 ; shift 
debdir=$1 ; shift
cmndir=$1 ; shift 
pkg=$1

# Check if skeleton and description file exist 
if [ ! -f $cmndir/$pkg.dscr ] || [ ! -f $debdir/$pkg.control.in ] ; then 
    echo "$0: couldn't find one and/or both of" 
    echo "   $cmndir/$pkg.dscr"
    echo "   $debdir/$pkg.control.in"
    echo "giving up" 
    exit 4
fi 

# Find the short description 
short=`sed -n 's/^short: \(.*\)$/\1/p' < ${cmndir}/${pkg}.dscr` 
if [ "x$short" = "x" ] ; then 
    echo "$0: short description empty - giving up" 
    exit 4
fi

# Find the long description
#
long=`sed -e '/^#.*$/d' \
          -e '/^short:.*$/d' \
          -e '/^long:.*/d' \
          -e 's/^[ \t]*$/./' \
          -e 's/^\(.*\)$/ \1/' < ${cmndir}/${pkg}.dscr`
if [ "x$long" = "x" ] ; then 
    echo "$0: long description empty - giving up" 
    exit 4
fi

# Insert the short description 
sed -e "s|@short@|$short|"  < $debdir/$pkg.control.in > ${cmndir}/tmp

# Now prepare to insert long description 
#
# first split the file at mark '@long@'
csplit -q -f ${cmndir}/tmp. -k ${cmndir}/tmp "/@long@/"

# cat first part to new file 
cat ${cmndir}/tmp.00 > ${cmndir}/tmp

# then output the long description 
echo "$long" >> ${cmndir}/tmp

# If this is not the bin package,  give a refernce to that 
if [ "x$pkg" != "xroot-bin" ] ; then 
  echo " ." >> ${cmndir}/tmp
  echo " See also root-bin package documentation for more information" >> \
	${cmndir}/tmp
fi

# Insert the general description 
if [ -f ${cmndir}/general.dscr ] ; then 
  # put an 'empty' line
  echo " ." >> ${cmndir}/tmp

  # put the general documentation in the end 
  sed -e '/^#.*$/d' \
      -e '/^general:.*/d' \
      -e 's/^[ \t]*$/./' \
      -e 's/^\(.*\)/ \1/' < ${cmndir}/general.dscr >> ${cmndir}/tmp 
fi

# and finally cat the last part of the file to new file 
cat ${cmndir}/tmp.01 >> ${cmndir}/tmp 

# remove temporary split files
rm ${cmndir}/tmp.00 ${cmndir}/tmp.01

# remove the remainder of the split 
sed "/^@long@.*$/d" < ${cmndir}/tmp > ${cmndir}/control.tmp

# list the entry to standard out 
cat ${cmndir}/control.tmp >> ${tgtdir}/control 

# Put a new line
echo "" >> ${tgtdir}/control 

# remove temporary files 
rm ${cmndir}/tmp ${cmndir}/control.tmp

#
# $Log: makedebcontrol.sh,v $
# Revision 1.1  2001/04/23 14:11:47  rdm
# part of the debian and redhat build system.
#
#
