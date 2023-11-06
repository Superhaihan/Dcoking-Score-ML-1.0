#!/bin/bash

path='/cadd2/users/liuhaihan/dockdb2/grid'
if [ -n "$1" ];then
	if [ -s "$1" ]&&[ ${1##*.} = mae ]||[ ${1##*.} = maegz ];then
		file=$1
		name=${1%.mae*}
	else
		echo "Unable not find $1, $1 must be a mae or margz file"
		exit
	fi
else
	lig=($(echo *.mae *.maegz | sed -s 's/*.maegz//;s/*.mae//'))
	if [ ! -n "${lig[0]}" ];then
		echo "Unable to find mae or margz files"
		exit
	fi

	if [ ${#lig[*]} = 1 ];then
		file=${lig[0]}
	else
		until [ -n "$name" ];do
			n=1
			for i in ${lig[*]};do
				echo "${n} = ${i}"
				n=$((n+1))
			done
			
			echo 'Choose the number of mae or maegz file:'
			read num
			
			if [ ! -n "`echo $num | tr -d 0-9`" ]&&[[ "$num" -gt 0 ]]&&[[ "$num" -lt "$n" ]];then
				file=${lig[$((num-1))]}
			fi
		done
	fi
	name=${file%.mae*}
fi

cpu=$[$(grep processor /proc/cpuinfo | wc -l)/2]
if [ -e /usr/bin/nvidia-smi ];then
	C=$((cpu-1))
else
	C=$cpu
fi

echo -e "There are \e[1;32m${cpu}\e[0m cores. Press \e[1;31mEnter\e[0m key to use \e[1;32m${C}\e[0m parallels or enter the parallels number: "
read thread
echo
thread=${thread:-$C}
until [ -n "$thread" ]&&[ ! -n "$(echo $thread | tr -d 0-9)" ]&&[[ "$thread" != 0 ]]&&[[ "$thread" -le "$cpu" ]];do
	echo "Invalid Input! please enter again!"
	echo -e "There are \e[1;32m${cpu}\e[0m cores. Press \e[1;31mEnter\e[0m key to use \e[1;32m${C}\e[0m parallels or enter the parallels number: "
	read thread
	echo
	thread=${thread:-$C}
done

tmp_file="/tmp/$$.fifo"
mkfifo $tmp_file
exec 6<>$tmp_file
rm -f $tmp_file
for ((i=0;i<thread;i++));do
	echo >&6
done

sign=$(date +%m%d%H%M%S)
p=$(pwd)
mkdir ${name}
mkdir /tmp/pdbbind${sign}
cd /tmp/pdbbind${sign}
n=0
for i in ${path}/glide-grid_*.zip;do
	id=$(echo $i | sed 's/^.*glide-grid_//;s/\.zip$//')
	read -u6
	{
		cat << EOF > ${id}_${name}_sp.in
FORCEFIELD   S-OPLS
GRIDFILE   $i
LIGANDFILE   ${p}/$file
PRECISION   SP
EOF
		$SCHRODINGER/glide ${id}_${name}_sp.in -HOST localhost -WAIT > ${id}_${name}_sp.log
		echo >&6
		if [ -s ${id}_${name}_sp_pv.maegz ];then
			mv ${id}_${name}_sp_pv.maegz ${p}/${name}
		else
			echo $id fail
			echo $id >> ${p}/error.log
		fi
		#rm ${id}_${name}_xp.in
	} &
	n=$((n+1))
	echo -ne "\r$n"
done
cd $p
rm -rf /tmp/${sign}/${task}
echo

wait
exec 6<&-
exec 6>&-
