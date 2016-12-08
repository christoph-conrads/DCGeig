#!/bin/bash

# Copyright 2016 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

if [ $# -lt 1 ]
then
	aout=$(basename -- "$0")
	echo "usage: ${aout} <log file 1> [log file 2] [...]"
	exit 1
fi


printf "%27s %s %s %8s %7s %s\n" filename pid sid 'min:fe' 'max:fe' 'median:fe'

fgrep  'nan      nan' $@ | \
	awk '{print $1, $2, $3, $19, $20, $21}' | \
	sort -g --key=5 | \
	xargs printf "%27s %3d %3d %s %s %s\n"
