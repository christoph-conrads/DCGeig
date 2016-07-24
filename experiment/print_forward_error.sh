#!/bin/bash

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
