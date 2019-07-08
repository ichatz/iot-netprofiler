#!/bin/bash
total=200
delay=5
logfile=rnd_$1_$(date +%F_%H:%M)_
declare -a nodes=("aaaa::212:7402:2:202"
		"aaaa::212:7403:3:303"
		"aaaa::212:7404:4:404"
		"aaaa::212:7405:5:505"
		"aaaa::212:7406:6:606"
		"aaaa::212:7407:7:707"
		"aaaa::212:7408:8:808"
		"aaaa::212:7409:9:909"
		"aaaa::212:740a:a:a0a")

for i in "${nodes[@]}"
  do
    echo "Ping $i"
    log="$logfile$i.log"
    resp=`ping6 -c $total -i $delay $i > $log &`
    echo $resp
    sleep 1
  done

log="$logfile""routes.log"
resp=`lynx -dump http://[aaaa::212:7401:1:101] > $log &`
echo $resp

