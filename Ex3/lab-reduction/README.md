# reduction lab

## Build

```bash
mkdir release
cd release
cmake -DCMAKE_BUILD_TYPE=Release ..
```

## reduction-lib with CUB

Get CUB:

```bash
git clone https://github.com/NVlabs/cub.git ext/
```

## Performance

Using atomics for final sum computation after block-reduce with shared memory can have a significant impact on performance. In that case use two kernel passes, where the second kernel with `kernel<<<1, blocksize>>>(y,y,blocks.x)` reduces the partial results of the blocks.
