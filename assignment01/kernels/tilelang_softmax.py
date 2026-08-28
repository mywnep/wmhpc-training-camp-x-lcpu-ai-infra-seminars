"""问题 7.7（压轴）：softmax in TileLang（FROM-SCRATCH）。

contract：
- softmax(x) 接收形状 (M, N) 的 float32 CUDA tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 用 TileLang 自己写，一个 block 处理一行（或一小批行）；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意，可以假设 N <= 4096。TileLang 的 kernel 按形状编译，
  用 make_xxx(M, N) 针对形状生成、在 wrapper 里按形状缓存编译结果
  是常见做法（结构可以参考 7.3、7.4）；
- 归约用 T.reduce_max / T.reduce_sum，逐元素部分用 T.Parallel 加 T.exp；
- fragment 的宽度建议取不小于 N 的 2 的幂（类比 Triton 的
  next_power_of_2），不足的位置补 -inf（T.if_then_else 加 T.infinity），
  否则布局推断可能报 no available layout；
- 通过 pytest tests/test_tilelang_softmax.py 即为完成。

(Optional) 将你的实现和 torch.softmax 比较一下性能（行宽取 256/1024/4096），
Tip: elementwise + 行内归约的 kernel 大概率是带宽瓶颈，可以想想理论上限是多少。
"""

import tilelang
import tilelang.language as T
import torch


def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    BLOCK_N = 1 << (N - 1).bit_length()
    threads = 128
    prog = make_softmax(M, N, BLOCK_N, threads)
    kernel = tilelang.compile(prog, out_idx=[1])
    return kernel(x)


def make_softmax(M, N, BLOCK_N, threads=128, dtype="float32"):
    @T.prim_func
    def main(A: T.Buffer((M, N), dtype), B: T.Buffer((M, N), dtype)):
        with T.Kernel(M, 1, threads=threads) as (bx, _):
            # in to x
            x_local = T.alloc_fragment((BLOCK_N,), dtype)
            for col in T.Parallel(BLOCK_N):
                x_local[col] = T.if_then_else(col < N, A[bx, col], -T.infinity(dtype))

            # reduce_max
            max_local = T.alloc_fragment((1,), dtype)
            T.reduce_max(x_local, max_local, dim=0)

            # exp(x - max)
            for col in T.Parallel(BLOCK_N):
                x_local[col] = T.exp(x_local[col] - max_local[0])

            # reduce_sum
            sum_local = T.alloc_fragment((1,), dtype)
            T.reduce_sum(x_local, sum_local, dim=0)

            for col in T.Parallel(BLOCK_N):
                if col < N:
                    B[bx, col] = x_local[col] / sum_local[0]

    return main
