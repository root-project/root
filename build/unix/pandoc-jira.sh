#!/bin/bash

# Script to be called as filter by pandoc.
# Replaces  text (!) [ROOT-7392] with a proper link; doesn't touch any link that
# includes [ROOT-7392](...)
# Axel, 2015-06-09

# In JSON:
# Replace {"t":"Str","c":"[ROOT-7392]"}
# with {"t":"Link","c":[[{"t":"Str","c":"ROOT-7290"}],["https://sft.its.cern.ch/jira/browse/ROOT-7290",""]]}

sed -E 's@\{"t":"Str","c":"\[ROOT-([[:digit:]]+)\]"\}@{"t":"Link","c":[[{"t":"Str","c":"ROOT-\1"}],["https://sft.its.cern.ch/jira/browse/ROOT-\1",""]]}@g'
