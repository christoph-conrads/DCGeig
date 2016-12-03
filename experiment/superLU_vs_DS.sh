#!/bin/bash

# Copyright 2016 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

sigint_handler() {
	echo "Caught SIGINT signal"
	kill -TERM "$child" 2>/dev/null
	exit 0
}

trap sigint_handler SIGINT
trap sigint_handler SIGTERM


for file in ~/matrices/*.mtx
do
	python superLU_vs_DS.py "${file}" &

	child=$!
	wait "${child}"
done
