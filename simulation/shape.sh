#!/bin/bash
export LC_NUMERIC=en_US.UTF-8

input_file=$1
mapfile -t list < $input_file

tc_adapter=wlp59s0 #wifi
#tc_adapter=enx106530c1958e #eth
#tc_adapter=eno1 #server eth


t=$(sleepenh 0)

sudo tc qdisc del dev $tc_adapter root
#sudo tc qdisc add dev $tc_adapter root handle 1: htb default 11
#sudo tc class add dev $tc_adapter parent 1: classid 1:11 htb rate 10mbit

sudo tc qdisc add dev $tc_adapter root tbf rate 10mbit latency 15ms burst 3000

#iperf3 -c netmsys.org -p 10001 -t 1000 &
#iperf3 -c fs0.das5.cs.vu.nl -p 10004 -t 1000 &
#iperf3 -c 130.83.163.233 -p 10001 -t 1000 &

SECONDS=0

for item in ${list[@]}
do
	if [ -z "${item//[$'\t\r\n ']}" ] #skip empty var
 	then 
		continue
	fi
	
	#sudo tc class change dev $tc_adapter parent 1: classid 1:11 htb rate "${item//[$'\t\r\n ']}"mbit
	sudo tc qdisc change dev $tc_adapter root tbf rate "${item//[$'\t\r\n ']}"mbit latency 15ms burst 3000
	
	#date +"B | %H:%M:%S.%N | $(printf "%.3f" ${item//[$'\t\r\n ']}) mbit"
	duration=$SECONDS
	echo -n "$(printf "%.3f" ${item//[$'\t\r\n ']})"
	t=$(sleepenh $t 1)
	
	duration=$SECONDS
	echo -n "$(printf "%.3f" ${item//[$'\t\r\n ']})"
	t=$(sleepenh $t 1)

	duration=$SECONDS
	echo -n "$(printf "%.3f" ${item//[$'\t\r\n ']})"
	t=$(sleepenh $t 1)
	
	duration=$SECONDS
	echo -n "$(printf "%.3f" ${item//[$'\t\r\n ']})"
	t=$(sleepenh $t 1)
	
	duration=$SECONDS
	echo -n "$(printf "%.3f" ${item//[$'\t\r\n ']})"
	t=$(sleepenh $t 1)
	
#	duration=$SECONDS
#	echo "B | $(printf "%.2d" $(($duration / 60))):$(printf "%.2d" $(($duration % 60))) | $(printf "%.3f" ${item//[$'\t\r\n ']})"
#	t=$(sleepenh $t 1)
	
#	duration=$SECONDS
#	echo "B | $(printf "%.2d" $(($duration / 60))):$(printf "%.2d" $(($duration % 60))) | $(printf "%.3f" ${item//[$'\t\r\n ']}) mbit"
#	t=$(sleepenh $t 1)
	
#	duration=$SECONDS
#	echo "B | $(printf "%.2d" $(($duration / 60))):$(printf "%.2d" $(($duration % 60))) | $(printf "%.3f" ${item//[$'\t\r\n ']}) mbit"
#	t=$(sleepenh $t 1)
	
#	duration=$SECONDS
#	echo "B | $(printf "%.2d" $(($duration / 60))):$(printf "%.2d" $(($duration % 60))) | $(printf "%.3f" ${item//[$'\t\r\n ']}) mbit"
#	t=$(sleepenh $t 1)
	
#	duration=$SECONDS
#	echo "B | $(printf "%.2d" $(($duration / 60))):$(printf "%.2d" $(($duration % 60))) | $(printf "%.3f" ${item//[$'\t\r\n ']}) mbit"
#	t=$(sleepenh $t 1)
done

#echo "shaping completed, removing qdisc"
sudo tc qdisc del dev $tc_adapter root