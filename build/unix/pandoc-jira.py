#!/usr/bin/env python

import json
from sys import stdin
from StringIO import StringIO

def gotObj(obj):
    # Replace {"t":"Str","c":"[ROOT-7392]"}
    if 't' in obj and obj['t'] == 'Str' \
       and obj['c'][0:6] == '[ROOT-':
        # with {"t":"Link","c":[[{"t":"Str","c":"ROOT-7290"}],["https://sft.its.cern.ch/jira/browse/ROOT-7290",""]]}
        print {"t":"Link","c":[[{"t":"Str","c":"ROOT-7290"}],["https://sft.its.cern.ch/jira/browse/ROOT-7290",""]]}

json.load(stdin, object_hook = gotObj);

