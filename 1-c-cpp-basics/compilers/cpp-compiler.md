# C++ Compiler

## Compile and Link a CPP Program

```shell
g++ main.cpp -o main.out
```

## Execute a CPP Binary File

```shell
./main.out
```

## Only Compile a CPP Program

C++ compiler will only compile but not link C++ programs when given a `-c` argument, and the compiled (but not linked) object is not executable.

```shell
g++ -c main.cpp -o main.obj
```

Compiled objects need to be linked before executing.

```shell
g++ -c main.cpp -o main.obj
g++ main.obj -o main.out
./main.out
```
