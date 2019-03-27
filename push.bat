echo off
set arg1="all"
if [%1]==[] (
	git add .
	git commit -m "win"
	git push
	
	set arg1="NONE"
)


if %arg1%=="all" (
	set currentpath=%cd%
	echo all
	cd ..
	call pushall
	
	cd %currentpath%
)