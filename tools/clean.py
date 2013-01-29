#! /usr/bin/python
# -*-coding:utf-8-*-

import sys
import os

if len(sys.argv) != 2:
    print 'usage: %s dir' % sys.argv[0]
    sys.exit(-1)

files = []

def walk(dir):
    global files
    li = os.listdir(dir)
    for x in li:
        if x.startswith('.'):
            continue
        ndir = os.path.join(dir, x)
        if os.path.isdir(ndir):
            walk(ndir)
        elif os.path.isfile(ndir):
            files.append(ndir)

walk(sys.argv[1])

for f in files:
    if f.endswith('.pyc'):
        os.remove(f)

