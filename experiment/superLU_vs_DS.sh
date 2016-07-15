#!/bin/bash

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
