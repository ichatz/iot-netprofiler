#! /bin/bash

#set -e
#set -x
#

logfile=logs/test_1BH_$(date +%F_%T).log

declare -a nodes=("aaaa::212:7404:4:404" 
		  "aaaa::212:7402:2:202"
		  "aaaa::212:7403:3:303"
		  "aaaa::212:7405:5:505"
		  "aaaa::212:740b:b:b0b"
		  "aaaa::212:7407:7:707"
		  "aaaa::212:7409:9:909"
		  "aaaa::212:7408:8:808"
		  "aaaa::212:740a:a:a0a"
                 )
count=0 
while [[ $count -lt 100 ]]
  do
    for i in "${nodes[@]}"
    do
       echo "$i"
       resp=$(ping6 -c 1 $i | grep ttl)
       echo $resp
       echo "${count} ${resp}"  >> $logfile
       sleep 1
    done
  count=$(( $count + 1 ))
  done


#resp=$(lynx -dump http://[aaaa::212:7401:1:101])
#echo $resp

	