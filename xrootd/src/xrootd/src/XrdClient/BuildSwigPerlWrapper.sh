#!/bin/sh

# $Id$

echo "This works only (and has been tested with) swig 1.3.22"

swig -perl XrdClientAdmin_c.hh

echo "Done. Check if the perl wrapper seems OK."

