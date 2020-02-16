#!/bin/bash
export LC_NUMERIC=en_US.UTF-8

input_file=$1
mapfile -t list < $input_file

tc_adapter=wlp59s0 #set to wifi/ethernet adapter

t=$(sleepenh 0)

sudo tc qdisc del dev $tc_adapter root
sudo tc qdisc add dev $tc_adapter root handle 1: htb default 11
sudo tc class add dev $tc_adapter parent 1: classid 1:11 htb rate 10mbit

#iperf3 -c netmsys.org -p 10001 -t 1000 &

for item in ${list[@]}
do
	if [ -z "${item//[$'\t\r\n ']}" ] #skip empty var
 	then 
		continue
	fi
	date +"B | %H:%M:%S.%N | $(printf "%.3f" ${item//[$'\t\r\n ']}) mbit"
	
	
	sudo tc class change dev $tc_adapter parent 1: classid 1:11 htb rate "${item//[$'\t\r\n ']}"mbit
	
	t=$(sleepenh $t 1)
done

echo "shaping completed, removing qdisc"
sudo tc qdisc del dev $tc_adapter root