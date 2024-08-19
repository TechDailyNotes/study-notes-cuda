# CUDA Basics

## CPU vs. GPU

### CPU Programming

- Optimized for latency
- Complex control logic
- Temporal programming

### GPU Programming

- Optimized for throughput
- Simple instructions (SIMT architecture)
- Spatial programming

## Fused vs. Coalesced

- Similarity: Accelerate memory access/memory IO

### Fused Operations

- Purpose
  - Access data from closer locations
- Procedure
  - Move data from HBM/DRAM to Cache/SRAM once
  - Access data from Cache/SRAM multiple times

### Coalesced Operations

- Purpose
  - Access data in fewer shots
- Procedure
  - Put data accessed together in adjacent locations in the memory
  - Transpose matrix in advance
