#!/bin/sh -e 
#
# $Id: makerpmspec.sh,v 1.7 2005/03/21 21:34:37 rdm Exp $
#
# Make the rpm spec file in ../root.spec
#
#
### echo %%% Some general variables 
chmod a+x build/package/lib/*
tgtdir=rpm

### echo %%% Packages ordered by preference
pkglist=`./configure --pkglist					\
		  --enable-table				\
		  --enable-ruby					\
		  --enable-qt					\
		  --enable-pythia				\
		  --enable-xrootd				\
		  --enable-shared				\
		  --enable-soversion				\
		  --enable-explicitlink				\
		  --disable-rpath				\
		  --disable-afs					\
		  --disable-srp					\
		  --disable-builtin-freetype			\
		  --disable-builtin-afterimage			\
		| sed -n -e 's/packages: //p' -e 's/ttf-root-installer//'` 
builddepends=`build/package/lib/makebuilddepend.sh rpm $pkglist`
dpkglist="`echo $pkglist | sed -e 's/ *ttf-root[-a-z]* *//g' -e 's/ /, /g'`, root-ttf"

# ROOT version 
version=`cat build/version_number | tr '/' '.'`

### echo %%% make sure we've got a fresh file 
rm -f root.spec

### echo %%% Write header stuff 
sed -e "s/@version@/${version}/" \
    -e "s/@builddepends@/${builddepends}/" \
    -e "s/@pkglist@/${dpkglist}/" \
    < build/package/rpm/spec.in > root.spec

# Write out sub-package information 
for p in $pkglist ; do 
    if test "x$p" = "xttf-root-installer" ; then continue ; fi
    echo "Adding package $p to spec file"
    short=`sed -n 's/Description: //p' < build/package/common/$p.control`
    long=`sed -n -e 's/^ \./ /' -e 's/^ //p' < build/package/common/$p.control`
    cat >> root.spec <<-EOF
	# -----------------------------------------------
	# Package $p
	%package -n $p
	Summary: $short
	Group: Applications/Physics
	EOF
    case $p in 
	ttf-root*) 
	    echo "Provides: root-ttf" 				>> root.spec 
	    ;;
	*xrootd|*rootd)
	    echo "Provides: root-file-server" 			>> root.spec 
	    ;;
	*minuit|*fumili)
	    echo "Provides: root-fitter"			>> root.spec
	    ;;
    esac
    case $p in 
	root-bin) 
	    echo "Requires: root-fitter, libroot-dev" 		>> root.spec 
	    ;;
	libroot) 
	    echo "Requires: root-ttf"				>> root.spec 
	    ;; 
	*rootd) 
	    echo "Prefix: %_prefix" >> root.spec 
	    ;; 
    esac
    cat >> root.spec <<-EOF
	%description -n $p
	$long

	%files -n $p -f rpm/$p.install

	EOF

    for s in post postun pre preun ; do 
	if test -f build/package/rpm/$p.$s ; then 
	    echo "%$s -n $p" 			>> root.spec
	    cat build/package/rpm/$p.$s		>> root.spec
echo "" >> root.spec
	fi
done 
echo "" >> root.spec
done 
cat >> root.spec <<EOF
#
# End of ROOT spec file
#
EOF

#
# EOF
#
