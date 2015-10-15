#!/bin/bash
dbNames=('cudb' 'afdb' 'svdb' 'mitdb' 'nsrdb')
readSignalCommand='rdsamp -r '
readAnnotationCommand='rdann -a qrs -r '
annotateCommand='gqrs -r '
format1='signal'
format2='annotation'

WFDB=". ./nsrdbRaw ./svdbRaw ./afdbRaw ./cudbRaw ./mitdbRaw"
export WFDB

for dbName in ${dbNames[@]}
do
	rsync -Cavz physionet.org::${dbName} ./${dbName}Raw

	mkdir -p "./${dbName}"

	cat "./${dbName}Raw/records" |
		while read line; 
		do 
			$annotateCommand $line 
			$readSignalCommand $line > "./${dbName}/${line}.${format1}"
			$readAnnotationCommand $line > "./${dbName}/${line}.${format2}"
	done
	cp ./${dbName}Raw/records ./${dbName}
done
	
# nsrdb's last 5 records have a weird shape, deleting them
sed -i '/^19/d' ./nsrdb/records