# C Compiler

## Compile and Link a C Program

```shell
gcc main.c -o main.out
```

## Run a C Binary File

```shell
./main.out
```

## Only Compile a C Program

C compiler will only compile but not link C programs when given a `-c` argument, and the compiled (but not linked) object is not executable.

```shell
gcc -c main.c -o main.obj
```

Compiled objects need to be linked before executing.

```shell
gcc -c main.c -o main.obj
gcc main.obj -o main.out
./main.out
```
