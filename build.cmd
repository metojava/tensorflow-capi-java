@echo off
rem Builds target\native\tf_capi_jni.dll using MSYS2 gcc.
rem Usage: build.cmd [path-to-libtensorflow-cpu-windows-x86_64]
setlocal
if "%JAVA_HOME%"=="" (
    echo JAVA_HOME is not set
    exit /b 2
)
set TFHOME=%~1
if "%TFHOME%"=="" set TFHOME=C:\Users\dell\Desktop\libtensorflow-cpu-windows-x86_64
C:\msys64\usr\bin\bash.exe "%~dp0src\main\native\build-native.sh" "%TFHOME%"
