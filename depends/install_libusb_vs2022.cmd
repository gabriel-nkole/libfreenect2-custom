rem This can only be run in a Git Shell or similar environments
rem with access to git.exe and msbuild.exe.

rmdir /s /q libusb_src libusb

git clone https://github.com/libusb/libusb.git libusb_src || exit /b
cd libusb_src

set CONFIG=Release
set YEAR=2022
set MSBUILD="C:\Program Files (x86)\MSBuild\17.0\Bin\MSBuild.exe"

%MSBUILD% msvc\libusb_dll_%YEAR%.vcxproj /p:Platform=x64 /p:Configuration=%CONFIG% /target:Rebuild || exit /b

mkdir ..\libusb\include\libusb-1.0
copy libusb\libusb.h ..\libusb\include\libusb-1.0
mkdir ..\libusb\MS64\dll
copy x64\%CONFIG%\dll\*.lib ..\libusb\MS64\dll
copy x64\%CONFIG%\dll\*.dll ..\libusb\MS64\dll
copy x64\%CONFIG%\dll\*.pdb ..\libusb\MS64\dll
