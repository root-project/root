#!/bin/bash

# This script creates the file Doxyfile_INPUT which defines the list of directories to be
# analysed by doxygen. To only build a subset of the documentation it is enough to comment
# the unwanted folders.
#  Author: Olivier Couet <olivier.couet@cern.ch> CERN

# This line is mandatory. Do not comment it
echo "INPUT = ./mainpage.md                    \\" > Doxyfile_INPUT

# echo "        ../../core/base/                 \\" >> Doxyfile_INPUT
# echo "        ../../core/dictgen/              \\" >> Doxyfile_INPUT
# echo "        ../../core/cont/                 \\" >> Doxyfile_INPUT
# echo "        ../../core/foundation/           \\" >> Doxyfile_INPUT
# echo "        ../../core/gui/                  \\" >> Doxyfile_INPUT
# echo "        ../../core/macosx/               \\" >> Doxyfile_INPUT
# echo "        ../../core/meta/                 \\" >> Doxyfile_INPUT
# echo "        ../../core/metacling/            \\" >> Doxyfile_INPUT
# echo "        ../../core/clingutils/           \\" >> Doxyfile_INPUT
# echo "        ../../core/multiproc/            \\" >> Doxyfile_INPUT
# echo "        ../../core/rint/                 \\" >> Doxyfile_INPUT
# echo "        ../../core/thread/               \\" >> Doxyfile_INPUT
# echo "        ../../core/unix/                 \\" >> Doxyfile_INPUT
# echo "        ../../core/winnt/                \\" >> Doxyfile_INPUT
# echo "        ../../core/imt/                  \\" >> Doxyfile_INPUT
# echo "        ../../core/zip/inc/Compression.h \\" >> Doxyfile_INPUT
# echo "        ../../geom/                      \\" >> Doxyfile_INPUT
# echo "        ../../graf2d/asimage/            \\" >> Doxyfile_INPUT
# echo "        ../../graf2d/cocoa/              \\" >> Doxyfile_INPUT
# echo "        ../../graf2d/fitsio/             \\" >> Doxyfile_INPUT
# echo "        ../../graf2d/gpad/               \\" >> Doxyfile_INPUT
# echo "        ../../graf2d/gpadv7/             \\" >> Doxyfile_INPUT
# echo "        ../../graf2d/graf/               \\" >> Doxyfile_INPUT
# echo "        ../../graf2d/gviz/               \\" >> Doxyfile_INPUT
# echo "        ../../graf2d/postscript/         \\" >> Doxyfile_INPUT
# echo "        ../../graf2d/quartz/             \\" >> Doxyfile_INPUT
# echo "        ../../graf2d/win32gdk/           \\" >> Doxyfile_INPUT
# echo "        ../../graf2d/x11/                \\" >> Doxyfile_INPUT
# echo "        ../../graf2d/x11ttf/             \\" >> Doxyfile_INPUT
# echo "        ../../graf3d/eve/                \\" >> Doxyfile_INPUT
# echo "        ../../graf3d/eve7/               \\" >> Doxyfile_INPUT
# echo "        ../../graf3d/g3d/                \\" >> Doxyfile_INPUT
# echo "        ../../graf3d/gl/                 \\" >> Doxyfile_INPUT
# echo "        ../../graf3d/gviz3d/             \\" >> Doxyfile_INPUT
# echo "        ../../gui/                       \\" >> Doxyfile_INPUT
# echo "        ../../hist/                      \\" >> Doxyfile_INPUT
# echo "        ../../html/                      \\" >> Doxyfile_INPUT
echo "        ../../io/doc/TFile               \\" >> Doxyfile_INPUT
echo "        ../../io/dcache/                 \\" >> Doxyfile_INPUT
echo "        ../../io/gfal/                   \\" >> Doxyfile_INPUT
echo "        ../../io/io/                     \\" >> Doxyfile_INPUT
echo "        ../../io/sql/                    \\" >> Doxyfile_INPUT
echo "        ../../io/xml/                    \\" >> Doxyfile_INPUT
echo "        ../../io/xmlparser/              \\" >> Doxyfile_INPUT
echo "        ../../main/src/hadd.cxx          \\" >> Doxyfile_INPUT
# echo "        ../../math/                      \\" >> Doxyfile_INPUT
# echo "        ../../montecarlo/                \\" >> Doxyfile_INPUT
# echo "        ../../net/alien/                 \\" >> Doxyfile_INPUT
# echo "        ../../net/auth/                  \\" >> Doxyfile_INPUT
# echo "        ../../net/davix/                 \\" >> Doxyfile_INPUT
# echo "        ../../net/http/                  \\" >> Doxyfile_INPUT
# echo "        ../../net/monalisa/              \\" >> Doxyfile_INPUT
# echo "        ../../net/net/                   \\" >> Doxyfile_INPUT
# echo "        ../../net/netx/                  \\" >> Doxyfile_INPUT
# echo "        ../../net/netxng/                \\" >> Doxyfile_INPUT
# echo "        ../../proof/                     \\" >> Doxyfile_INPUT
# echo "        ../../tmva/                      \\" >> Doxyfile_INPUT
# echo "        ../../roofit/                    \\" >> Doxyfile_INPUT
# echo "        ../../tree/                      \\" >> Doxyfile_INPUT
# echo "        ../../sql/                       \\" >> Doxyfile_INPUT
# echo "        ../../tutorial/                  \\" >> Doxyfile_INPUT
# echo "        ../../bindings/tpython/          \\" >> Doxyfile_INPUT
# echo "        ../../bindings/pyroot/           \\" >> Doxyfile_INPUT
# echo "        ../../bindings/                  \\" >> Doxyfile_INPUT

# echo "        ../../core/clib/                 \\" >> Doxyfile_INPUT
# echo "        ../../core/lzma/                 \\" >> Doxyfile_INPUT
# echo "        ../../core/newdelete/            \\" >> Doxyfile_INPUT
# echo "        ../../core/textinput/            \\" >> Doxyfile_INPUT
# echo "        ../../graf2d/mathtext/           \\" >> Doxyfile_INPUT
# echo "        ../../graf3d/ftgl/               \\" >> Doxyfile_INPUT
# echo "        ../../graf3d/glew/               \\" >> Doxyfile_INPUT
# echo "        ../../graf3d/x3d/                \\" >> Doxyfile_INPUT
# echo "        ../../net/rootd/                 \\" >> Doxyfile_INPUT
# echo "        ../../net/rpdutils/              \\" >> Doxyfile_INPUT


# Add to the list of files to be analyzed the .pyzdoc files created by extract_docstrings.py
# and print_roofit_pyz_doctrings.py
ls $DOXYGEN_PYZDOC_PATH/*.pyzdoc | sed -e "s/$/ \\\\/"  \
>> Doxyfile_INPUT

