#!/bin/sh -e 
#
# $Id$
#
# Make the debian packaging directory 
#

if [ $# -lt 1 ] ; then 
    echo "$0: I need to know the pacakging system"
    exit 1
fi

# save packaging system in logical variable 
pkgsys=$1

# Get the setup 
. build/package/lib/common.sh $1

for i in ${pkgs} ; do 
    case $i in 
    "root-ttf")    use_ttf="yes";; 
    "root-zebra")  use_zebra="yes";; 
    "root-gl")     use_gl="yes";; 
    "root-mysql")  use_mysql="yes";; 
    "root-pythia") use_pythia="yes";; 
    "root-star")   use_star="yes";; 
    "root-shift")  use_shift="yes";; 
    *) ;;
    esac
done 

# write begining 
echo -e "\t./configure ${arch} \\"
echo -e "\t\t--etcdir=${etcdir} \\"
echo -e "\t\t--prefix=${prefix} \\"
echo -e "\t\t--docdir=${docdir} \\"
echo -e "\t\t--mandir=${mandir} \\"
echo -e "\t\t--cintincdir=${cintdir} \\"

# Write ttf line 
if [ "x${use_ttf}" = "xyes" ] ; then 
    echo -e "\t\t--enable-ttf \\"
    echo -e "\t\t--with-ttf-fontdir=/usr/share/fonts/truetype \\"
else 
    echo -e "\t\t--disable-ttf \\"
fi

# Write ttf cernlib conversion line 
if [ "x${use_zebra}" = "xyes" ] ; then 
    echo -e "\t\t--enable-cern \\"
else 
    echo -e "\t\t--disable-cern \\"
fi

# write OpenGL line 
if [ "x${use_gl}" = "xyes" ] ; then 
    echo -e "\t\t--enable-opengl \\"
else 
    echo -e "\t\t--disable-opengl \\"
fi

# write MySQL line 
if [ "x${use_mysql}" = "xyes" ] ; then 
    echo -e "\t\t--enable-mysql \\"
else 
    echo -e "\t\t--disable-mysql \\"
fi

# write Pyhtia line 
if [ "x${use_pythia}" = "xyes" ] ; then 
    echo -e "\t\t--enable-pythia \\"
    echo -e "\t\t--enable-pythia6 \\"
else 
    echo -e "\t\t--disable-pythia \\"
    echo -e "\t\t--disable-pythia6 \\"
fi    

# write STAR line 
if [ "x${use_star}" = "xyes" ] ; then 
    echo -e "\t\t--enable-star \\"
else 
    echo -e "\t\t--disable-star \\"
fi

# write SHIFT line 
if [ "x${use_shift}" = "xyes" ] ; then 
    echo -e "\t\t--enable-rfio \\"
else 
    echo -e "\t\t--disable-rfio \\"
fi

# these are always disabled
echo -e "\t\t--disable-afs \\"
echo -e "\t\t--disable-srp \\"

# these are always enabled
echo -e "\t\t--enable-thread \\"
echo -e "\t\t--enable-shared \\"
echo -e "\t\t--enable-soversion \\"

# always look for additional xpms in this directory 
echo -e "\t\t--with-sys-iconpath=/usr/share/pixmaps"



#
# $Log$
#
