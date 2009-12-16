#!/bin/sh -e 
#
# $Id: makerpmspec.sh,v 1.15 2006/08/24 13:49:53 rdm Exp $
#
# Make the rpm spec file in ../root.spec
#
#
### echo %%% Some general variables 
chmod a+x build/package/lib/*
tgtdir=rpm

# Configure arguments 
confargs="--enable-cintex			\
    --disable-clarens				\
    --enable-explicitlink			\
    --enable-gdml				\
    --enable-gsl-shared				\
    --disable-fftw3				\
    --enable-ldap				\
    --disable-qt				\
    --disable-qtgsi				\
    --enable-mathcore				\
    --enable-mathmore				\
    --enable-minuit2				\
    --enable-mysql				\
    --disable-peac				\
    --enable-pgsql				\
    --enable-odbc				\
    --enable-reflex				\
    --enable-roofit				\
    --enable-ruby				\
    --enable-shadowpw				\
    --enable-shared				\
    --enable-soversion				\
    --enable-table				\
    --disable-rpath				\
    --disable-afs				\
    --disable-srp				\
    --enable-builtin-ftgl			\
    --disable-builtin-freetype			\
    --disable-builtin-pcre			\
    --disable-builtin-zlib			\
    --disable-alien				\
    --disable-chirp				\
    --disable-dcache				\
    --disable-g4root				\
    --disable-gfal				\
    --disable-globus				\
    --disable-monalisa				\
    --disable-oracle				\
    --disable-pythia6				\
    --disable-rfio				\
    --fail-on-missing				\
    --enable-unuran				\
    --enable-xrootd				\
    --disable-sapdb				\
    --enable-cern"

# Configure arguments 
# confargs="--enable-cintex				\
# 	  --enable-explicitlink				\
#     	  --enable-fftw3				\
# 	  --enable-mathmore				\
#           --enable-mathcore                          	\
#           --enable-minuit2                           	\
# 	  --enable-odbc					\
# 	  --enable-pythia6				\
# 	  --enable-qt					\
# 	  --enable-qtgsi				\
#     	  --enable-reflex				\
#     	  --enable-roofit				\
# 	  --enable-ruby					\
# 	  --enable-shared				\
# 	  --enable-shadowpw				\
# 	  --enable-soversion				\
#     	  --enable-table				\
# 	  --enable-xrootd				\
# 	  --disable-rpath				\
# 	  --disable-afs					\
# 	  --disable-srp					\
# 	  --disable-builtin-freetype			\
# 	  --disable-builtin-pcre			\
# 	  --disable-builtin-zlib"

test "x$QTDIR" = "x" || unset QTDIR
test "x$QTLIB" = "x" || unset QTLIB
test "x$QTINC" = "x" || unset QTINC

### echo %%% Packages ordered by preference
pkglist=`./configure --pkglist	${confargs} \
		| sed -n -e 's/packages: //p'` 
if test $? -ne 0 ; then 
    echo "Failed to get package list from configure, try" 
    echo "  ./configure --pkglist	${confargs}" \
	| tr '\n' ' ' | sed 's/ */ /'
    exit 1
fi
pkglist=`echo $pkglist | sed 's/libroot\([-a-zA-Z0-9]*\)/libroot\1 libroot\1-dev/g'`
pkglist="libroot-static $pkglist"
echo $pkglist
for i in $pkglist ; do 
    echo "Adding package $i to spec" 
done
builddepends=`build/package/lib/makebuilddepend.sh rpm $pkglist`
if test $? -ne 0 ;then 
    echo "Failed to make build dependencies" 
    exit 1
fi
dpkglist="`echo $pkglist | sed -e 's/ *ttf-root[-a-z]* *//g' -e 's/ /, /g'`, root-ttf"

# ROOT version 
version=`cat build/version_number | tr '/' '.'`
major=`echo $version | cut -f1 -d.`
sovers=`echo $version | cut -f1,2 -d.`
### echo %%% make sure we've got a fresh file 
rm -f root.spec
csplit -s -f root.spec. build/package/rpm/spec.in '/@builddepends@/'
cat root.spec.00     >  root.spec.in
echo "$builddepends" >> root.spec.in
sed '/@builddepends/d' < root.spec.01 >> root.spec.in
rm -f root.spec.00 root.spec.01
### echo %%% Write header stuff 
sed -e "s/@version@/${version}/"   \
    -e "s/@sovers@/${sovers}/"     \
    -e "s/@pkglist@/${dpkglist}/"  \
    -e "s/@confargs@/${confargs}/" \
    < root.spec.in > root.spec
rm -f root.spec.in

# Write out sub-package information 
for p in $pkglist ; do 
    # if test "x$p" = "xttf-root-installer" ; then continue ; fi
    case $p in 
	root-common)     pp=$p       ; c=libroot-core ;;
	libroot-static)  pp=$p       ; c=libroot-core ;;
	libroot*-dev)    pp=$p       ; c=`echo $p | sed 's/-dev//'`;;  
	libroot*)        pp=$p$major ; c=$p ;; 
	*-net-xrootd)    pp=$p	     ; c=root-system-xrootd ;;
	*)               pp=$p       ; c=$p ;; 
    esac 
    echo "Adding package $p ($pp) to spec file" 
    cat >> root.spec <<-EOF
	# -----------------------------------------------
	# Package $pp
	EOF
    sed -n -e "/Package: $p/,/^ / { s/^Package: $p\(@libvers@\)*/%package -n $pp/p; s/^Description:/Summary:/p ; /^ /q}" < build/package/common/$c.control >>  root.spec
    # 	# -----------------------------------------------
    # 	# Package $pp
    # 	%package -n $pp
    # 	Summary: $short
    # 	Group: Applications/Physics
    # 	EOF
    echo "Group: Applications/Physics" >> root.spec
    case $pp in 
	ttf-root*) 
	    echo "Provides: root-ttf" 				>> root.spec 
	    ;;
	*net-xrootd)
            echo "Provides: libXrdClient.so"			>> root.spec
            echo "Provides: libXrdCrypto.so"			>> root.spec
            echo "Provides: libXrdSut.so"			>> root.spec
	    ;;
	*xrootd)
	    echo "Provides: root-file-server" 			>> root.spec 
	    ;;
	*rootd)
	    echo "Provides: root-file-server" 			>> root.spec 
	    ;;
	*minuit*|*fumili)
	    echo "Provides: root-fitter"			>> root.spec
	    echo "Provides: $p"					>> root.spec
            ;;
	libroot*)
	    echo "Provides: $p"					>> root.spec
	    ;;
    esac
    case $p in 
	root-bin) 
	    echo "Requires: root-fitter"   		        >> root.spec 
	    ;;
	libroot) 
	    echo "Requires: root-ttf, root-system-common"	>> root.spec 
	    ;; 
	*rootd) 
	    echo "Prefix: %_prefix" >> root.spec 
	    ;; 
    esac
    sed -n "/Package: $p/,/^$/ { s/^Description:.*/%description -n $pp/p ; s/^ //p; /^$/q }"  < build/package/common/$c.control >>  root.spec
    case $p in 
	libroot-static) files=rpm/${p}.install ;;
	lib*-dev) 	files=rpm/${p}.install ;; 
	lib*)     	files=rpm/${p}${sovers}.install ;;
	*) 	  	files=rpm/${p}.install ;; 
    esac

    cat >> root.spec <<-EOF
	%files -n $pp -f ${files}
	%defattr(-,root,root)

	EOF

    for s in post postun pre preun ; do 
	if test -f build/package/rpm/$p.$s ; then 
	    echo "%$s -n $pp" 			>> root.spec
	    cat build/package/rpm/$p.$s		>> root.spec
	    echo "" >> root.spec
	fi
    done 
    echo "" >> root.spec
done 
# Add changelog 
cat >> root.spec <<EOF
# ____________________________________________________________________
%changelog 

EOF
sed -e '/Local variables:/d' 			\
    -e '/mode: debian-changelog/d' 		\
    -e '/End:/d' 				\
    <  build/package/debian/changelog 	 	\
    | build/package/lib/makerpmcl.pl		\
    >> root.spec
cat >> root.spec <<EOF
#
# End of ROOT spec file
#
EOF

l=`find math/unuran/src/ -name "*.tar.gz" 2>/dev/null`
if test "x$l" = "x" ; then 
    svn update math/unuran/src 
    rm -rf math/unuran/src/unuran-*-root
fi

#
# EOF
#
