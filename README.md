# Record

#### Softmax


![](images/softmax-performance.png)
- 平台 4090D 显卡

性能对比结论：

​Fused Softmax 优势显著​

- 通过单次内存加载（single-pass memory access）处理整个输入，减少全局内存访问次数，充分发挥 GPU 并行计算与共享内存（Shared Memory）优化，计算效率更高。


Online Softmax 的长序列优势​

- 采用分块计算策略​（tile-based computation），通过多次迭代逐步计算最大值和求和项，显著降低单次 SRAM 占用。
- ​性能拐点：当序列长度 N>7000 时，Online Softmax 的计算耗时低于 Native Softmax

#### Flash Attention

| Implementation |  Speed Up | 
| -------------- | --------- |
| Pytorch Spec  | 1x        |
| 2-Pass Impl    | 3.5x      |
| 1-Pass Impl    | 4.35x     |

- [ ] Flash Attention with tiling
- [ ] TODO: Add More test to measure performance

#### 2D-Convolution


## Reference 

https://github.com/srush/Triton-Puzzles

https://github.com/SiriusNEO/Triton-Puzzles-Lite