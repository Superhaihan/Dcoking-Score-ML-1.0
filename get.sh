#!/bin/bash

if [ ! -d "$1" ]||[ "$(\ls -1 ${1}/*_pv.maegz | wc -l >/dev/null 2>&1)" = '0' ];then
	echo "Unable to find $1"
	exit
fi

echo "Extract the score:"
sign=$(date +%m%d%H%M%S)
p=$(pwd)
mkdir /tmp/pdbbind${sign}
n=0
cd $1
for k in *_pv.maegz;do
	gzip -cd ${k} > /tmp/pdbbind${sign}/SP.mae
	a=`cat /tmp/pdbbind${sign}/SP.mae | grep -n ' s_m_title' | cut -d : -f 1 | sed -n '2p'`
	b=`cat /tmp/pdbbind${sign}/SP.mae | grep -n ' r_i_docking_score' | cut -d : -f 1 | sed -n '1p'`
	for i in `cat /tmp/pdbbind${sign}/SP.mae | grep -n ' :::' | cut -d : -f 1`;do
		if ((a<i));then
			b=$(tail -n +${i} /tmp/pdbbind${sign}/SP.mae | sed "{:begin;  /[^ ]\"/! { $! { N; b begin }; }; s# \".*\"#XXX#; };" | sed -n "$((b-a+2))p" | tr -d ' ')
			echo "${k%_${1}_sp_pv.maegz},${b}" >> /tmp/pdbbind${sign}/score.csv
			break 1
		fi
	done
	n=$((n+1))
	echo -ne "\r$n"
done
echo

echo 'item,docking_score' > ${p}/${1}_pdbbind_score.csv
sort -t ',' -k 2 -n /tmp/pdbbind${sign}/score.csv >> ${p}/${1}_pdbbind_score.csv
cd $p
rm -rf /tmp/pdbbind${sign}
