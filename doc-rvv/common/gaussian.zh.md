# `gaussian.cpp`（`PointCloud<float>` 卷积）：RVV 优化实现说明

本文记录 `common/src/gaussian.cpp` 中 `pcl::GaussianKernel` 对 `pcl::PointCloud<float>` 的 `convolveRows` / `convolveCols` 在本仓库相对上游的 `__RVV10__` 扩展、分流方式、测试与板卡性能数据。`compute` 两版本仍为标量，不在 RVV 改造范围内。

本仓库文件：[common/src/gaussian.cpp](../../common/src/gaussian.cpp)

上游对照文件：[gaussian.cpp](https://github.com/PointCloudLibrary/pcl/blob/master/common/src/gaussian.cpp)、[impl/gaussian.hpp](https://github.com/PointCloudLibrary/pcl/blob/master/common/include/pcl/common/impl/gaussian.hpp)

条带尾段 `_tu` 语义、固定 `LMUL` 与 `vfloat32m2_t` 选型见各模块 RVV 说明中的统一约定；

本仓库在 Gaussian（`PointCloud<float>` 可分离行/列卷积） 上已完成工作概览（细节见后文各节与 `test-rvv/common/gaussian`）：

- 库实现（`common/src/gaussian.cpp`）
  - 为 `convolveRows` / `convolveCols` 增加 `convolve*Standard`（与上游标量语义一致）与 `convolve*RVV`（`#if __RVV10__`：行向连续条带 + `vfslide1down` + `prefetch`，列向 `vlse` / `vsse` + `prefetch`）。
  - 成员函数内按编译本 TU 时是否定义 `__RVV10__` 在 Standard 与 RVV 间二选一；`compute` 两版本未 RVV 化；模板版卷积（`impl/gaussian.hpp`）未改。
- 板卡与 bench 对比流程（`test-rvv/common/gaussian`）
  - 说明并约定：标量/RVV 计时公平性依赖两套 `libpcl_common`（分别编进/不编进 `__RVV10__`）+ `run_bench_std` / `run_bench_rvv` 与 `analyze_bench_compare` 等；
- 单测（`test_gaussian.cpp` + `make run_test_*`）
  - 链当前安装的 `libpcl_common`，对 `compute`、`convolveRows` / `convolveCols` 及别名等做 API/数值回归。
- 同进程标量/RVV 对拍（不链 `libpcl_common`）
  - `gaussian_convolve_float_local.hpp` 与 `test_gaussian_convolve_compare.cpp`：`make run_test_convolve_compare`；与库侧算法需人工同步，与双库对拍互补（见 §5.4）。
- 文档（本文）
  - 记录分流条件、行/列 RVV 设计、数例、测试与板卡数据、与上游差异及维护约定。

---

## 1. 背景与需求

上游 `common/src/gaussian.cpp` 中，`GaussianKernel::compute` 与 `PointCloud<float>` 的 `convolveRows` / `convolveCols` 均为标量三重循环（行或列外层、列或行中层、核 tap 内层）；`operator()(列,行)` 对应行主序栅格。上游该文件不涉及 x86 SSE/AVX intrinsic，亦无可移植向量宏。

本仓库在 `__RVV10__` 下的约束如下：

- API 与模板签名保持不变：`gaussian.h` 中 `convolveRows` / `convolveCols` 声明未改。
- 语义与边界：左右或上下各 `radius` 个样本位置输出置零；中间区域为核权与邻域样本的乘加和，与上游循环变量对应关系一致。`input` 与 `output` 为同一对象时先拷贝到局部云再卷积，行为与上游一致。
- 数据布局：`PointCloud<float>` 为连续 `float` 缓冲，行方向步长为 $1$，列方向（固定列、沿行增长）步长为 `width` 个 `float`，列卷积采用 strided load/store，而非 segment gather。
- 不适合或未完成向量化的情形：`compute` 仍为短循环、`exp` 与异常路径为主，当前未 RVV 化。模板 `impl/gaussian.hpp` 中经 `std::function` 取值的路径未在本 TU 修改。
- 编译期分流与测试：`convolveRows` / `convolveCols` 在 RVV 构建下是否走向量路径，取决于本 TU 编入 `libpcl_common` 时是否定义 `__RVV10__`；应用或 bench 可执行文件自身的宏不能改写已安装共享库中的实现。标量与 RVV 的公平对比需分别编译两套 `libpcl_common`（见 `gaussian.cpp` 文件头注释与第 5.1 节板卡流程）。

---

## 2. 与上游实现的差异


| 条目                                  | 上游实现要点                                       | 本仓库在 **RVV10** 下的变化                                                                                                                                |
| ----------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `convolveRows`（`PointCloud<float>`） | 三重标量循环，`k` 从 `kernel_width` 递减到 $0$，`l` 同步递增 | `#if defined(__RVV10__)` 时调用 `convolveRowsRVV`：按行 `vsetvl` 条带、`vle32` 连续加载、`vfslide1down` 滑动窗口、`vfmacc`、`vse32` 写回；否则 `convolveRowsStandard` 保留原语义 |
| `convolveCols`（`PointCloud<float>`） | 同上，列方向内层访问 `input(i,l)`                      | `#if defined(__RVV10__)` 时调用 `convolveColsRVV`：按列条带、`vlse32` / `vsse32` 字节步长 `width*sizeof(float)`；否则 `convolveColsStandard`                       |
| `compute`（单核 / 核+导数）                | 标量 `exp`、裁剪、归一化                              | 未改，始终标量                                                                                                                                            |
| 模板卷积 `impl/gaussian.hpp`            | 头文件内实例化，`std::function` 取字段                  | 本仓库未改；不经过 `gaussian.cpp`                                                                                                                           |
| 成员入口 `GaussianKernel::convolve`*    | 别名处理 + 直接调用标量循环体                             | 别名处理不变；中间改为 `#if defined(__RVV10__)` 选择 `*RVV` 或 `*Standard`                                                                                       |
| 运行期回退                               | 无                                            | 卷积路径无规模阈值回退；仅编译期二分                                                                                                                                 |


数值一致性说明：

- RVV 行卷积按 tap 索引 `s = 0..kw` 与标量 `k = kw..0` 逐项对齐，同一 `s` 下使用 `kptr[kw - s]` 与 `vfmacc`，有限项浮点和的顺序与标量三重循环一致，预期与标量路径在 `float` 意义上对齐。
- 列卷积同样按 `s` 与 `kptr[kw-s]` 对齐；无向量归约重排。

---

## 3. 总体设计

### 3.1 分流条件

- 编译期：`#if defined(__RVV10__)` 时成员函数调用 `convolveRowsRVV` / `convolveColsRVV`；未定义时仅 `convolveRowsStandard` / `convolveColsStandard`。
- 运行期：卷积无额外开关；是否执行 RVV 指令完全由链接的 `libpcl_common` 构建选项决定。
- 边界：左右（或上下）`radius` 区域在 RVV 与标量路径中均写 $0$；中间区域进入条带循环。

### 3.2 组织方式

- `convolveRowsStandard` / `convolveColsStandard` 与 `convolveRowsRVV` / `convolveColsRVV` 同置于 `gaussian.cpp` 内匿名命名空间；成员函数仅做别名解析后二选一调用。
- 未引入共用 `rvv_point_load` 封装；卷积直接对 `float`* 与 `Eigen::VectorXf::data()` 使用 RVV intrinsic。

---

## 4. 详细实现

### 4.1 入口与函数分发

公开 `convolveRows` / `convolveCols` 在完成 `input == output` 时的拷贝与 `output` 尺寸处理后，根据编译本 TU 时是否定义 `__RVV10__` 选择 RVV 或标量助手。不存在按图像尺寸或核长度的运行期回退。


| 符号                     | 回退条件                                                |
| ---------------------- | --------------------------------------------------- |
| `convolveRowsRVV`      | 无；仅当未以 `__RVV10__` 编译本 TU 时不被调用                     |
| `convolveColsRVV`      | 同上                                                  |
| `convolveRowsStandard` | `__RVV10__` 未定义时固定路径；或 RVV 构建下逻辑等价备份（当前未被成员内二次分发调用） |
| `convolveColsStandard` | 同上                                                  |


对应实现中，成员 `convolveRows` 的分发骨架如下：

```cpp
// common/src/gaussian.cpp: GaussianKernel::convolveRows(...)
#if defined(__RVV10__)
  convolveRowsRVV (*unaliased_input, output, kernel, kernel_width, radius);
#else
  convolveRowsStandard (*unaliased_input, output, kernel, kernel_width, radius);
#endif
```

`convolveCols` 对称使用 `convolveColsRVV` / `convolveColsStandard`。

### 4.2 `convolveRowsRVV`

语义与 `convolveRowsStandard` 一致：对固定行下标 `j`，中间列 $i$ 的输出为：
$$
\sum_{s=0}^{kw} \mathrm{in}(i-\mathrm{radius}+s,j) \cdot \mathrm{kptr}[kw-s]
$$


其中 `kw = (int)kernel_width`，`kernel_width = kernel.size()-1`。

数据流：外层遍历 `j`；每行左侧与右侧 `radius` 列标量写 $0$；中间列以 `__riscv_vsetvl_e32m2(i_end - i)` 推进条带。`win_base = in_row + (i - radius)` 对齐到当前条带最左 tap 的起始地址；首 tap 用 `__riscv_vle32_v_f32m2(win_base, vl)` 装入连续 `vl` 个样本，乘加 `kptr[kw]`；后续 tap 用 `__riscv_vfslide1down_vf_f32m2` 将窗口沿列方向滑动一位，并由 `win_base[s + vl - 1]` 提供滑入元素，再与 `kptr[kw-s]` 做 `__riscv_vfmacc_vf_f32m2`。该滑动策略避免每个 tap 重复从内存覆盖整个窗口，连续读仍集中在行缓冲上。条带末用 `__riscv_vse32_v_f32m2` 写回。`__builtin_prefetch` 针对下一chunk 窗口起点，不改变数值结果。

对应实现中条带内核片段如下：

```cpp
// common/src/gaussian.cpp: convolveRowsRVV(...)
      const std::size_t vl = __riscv_vsetvl_e32m2 (i_end - i);
      __builtin_prefetch (in_row + (i + vl) - radius, 0, 3);
      const float *const win_base = in_row + (i - radius);
      vfloat32m2_t vin = __riscv_vle32_v_f32m2 (win_base, vl);
      vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2 (0.f, vl);
      acc = __riscv_vfmacc_vf_f32m2 (acc, kptr[kw], vin, vl);
      for (int s = 1; s <= kw; ++s)
      {
        const float tail = win_base[s + vl - 1];
        vin = __riscv_vfslide1down_vf_f32m2 (vin, tail, vl);
        acc = __riscv_vfmacc_vf_f32m2 (acc, kptr[kw - s], vin, vl);
      }
      __riscv_vse32_v_f32m2 (out_row + i, acc, vl);
      i += vl;
```

### 4.3 `convolveColsRVV`

语义与 `convolveColsStandard` 一致：固定列 $i$，沿行 $j$ 在中间段做同样系数顺序的乘加和。访存沿行步长为 `width` 个元素，故使用 `__riscv_vlse32_v_f32m2(src, row_stride_bytes, vl)` 与 `__riscv_vsse32_v_f32m2`，字节步长为 `(ptrdiff_t)width * sizeof(float)`。每个 tap `s` 独立一次 strided load，未使用 `vfslide1down`，因各 tap 起始行不同，滑动关系不对应单一 contiguous 窗口的错位复制。

对应实现中条带与 tap 循环片段如下：

```cpp
// common/src/gaussian.cpp: convolveColsRVV(...)
      const std::size_t vl = __riscv_vsetvl_e32m2 (j_end - j);
      __builtin_prefetch (in_p + i + (j + vl) * width, 0, 3);
      vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2 (0.f, vl);
      for (int s = 0; s <= kw; ++s)
      {
        const float *const src =
            in_p + i + (j - radius + static_cast<std::size_t> (s)) * width;
        const vfloat32m2_t vin = __riscv_vlse32_v_f32m2 (src, row_stride_bytes, vl);
        acc = __riscv_vfmacc_vf_f32m2 (acc, kptr[kw - s], vin, vl);
      }
      __riscv_vsse32_v_f32m2 (out_p + i + j * width, row_stride_bytes, acc, vl);
      j += vl;
```

### 4.4 数值算例

#### 约定

`input` / `output` 为 `pcl::PointCloud<float>`，行主序：`index = i + j * width`，其中 `0 <= i < width` 且 `0 <= j < height`，与 `operator()(i,j)` 一致。核与内层下标关系：

```text
kernel = [A, B, C]   ->  kptr[0]=A, kptr[1]=B, kptr[2]=C
kernel_width = 2, kw = 2, radius = 1, 共 3 个 tap
标量内层: k 自 kw 递减到 0, l 自 (i 或 j) - radius 起与 k 同步推进

--- input 在下方两子节中的取法（数字均为便于手算的教学取值）---

[convolveRows]  单行情形: width=8, height=1, 只考察 j=0；input(i,0)=(float)(i+1), i=0..7
  俯视一行，列下标 i 向右（与 in_row 上连续 float 同序）:

  列下标 i:  0    1    2    3    4    5    6    7
            +----+----+----+----+----+----+----+----+
  input:    | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  |   行 j=0
            +----+----+----+----+----+----+----+----+
  输出:      0   |<------  内区 i=1..6 有卷积和  ---->| 0
            ^列0、列7 为界外，实现中置 0，不参与中间乘加^

[convolveCols]  单列情形: height=8, 列下标 i=0, width=W>=1（步长为 W*sizeof(float)）；input(0,j)=(float)(j+1)
  侧视一列，行下标 j 向下（内存沿列每步 +width 个 float）:

  行下标 j:  0    1    2    3    4    5    6    7
            +----+
            | 1  |  j=0
            +----+
            | 2  |
            +----+
            | 3  |
            +----+
            | 4  |   列 i=0，input(0,j)=j+1；与下行卷积小节
            +----+
            | 5  |   「input(.,0) [1..8]」为同一批数，仅固定 i、沿 j 看
            +----+
            | 6  |
            +----+
            | 7  |
            +----+
            | 8  |  j=7
            +----+

  输出:     0   |<---- 内区 j=1..6 有卷积和 ---->|   0
           ^j=0^                               ^j=7^
           界外置0                              界外置0
```

#### `convolveRows` 与 `convolveRowsRVV`（`width=8`、行 `j=0`、`vl=2` 条带覆盖列 1 与 2）

在「约定」中单行 `input` 上，内区列 1 到 6 的标量加和即：

```text
  列下标 i:  0  1  2  3  4  5  6  7
  input(.,0) [1, 2, 3, 4, 5, 6, 7, 8]

i=1:  1*C + 2*B + 3*A
i=2:  2*C + 3*B + 4*A
i=3:  3*C + 4*B + 5*A
i=4:  4*C + 5*B + 6*A
i=5:  5*C + 6*B + 7*A
i=6:  6*C + 7*B + 8*A
```

`convolveRowsRVV` 在条带起点列 `i=1`、`vl=2` 时，本算例中 `j=0`，故 `in_row` 即 `input.data()` 上该行的首地址，沿行连续访问；`win_base = in_row + (i - radius)` 对应该条最左 tap（列 0），与 `tmp/讲解.md` 中「逐指令详解（`vl=2`）」同构。

```text
win_base 指向本行列 0，首元素为 input(0,0)=1
i - radius = 0

s=0:  vle 得 vin = [1, 2]；fmac 以 kptr[kw] 即 C
      acc = [1*C, 2*C]

s=1:  tail = win_base[1 + vl - 1] = win_base[2] = 3
      vfslide1down 得 vin = [2, 3]；fmac 以 kptr[1] 即 B
      acc = [1*C + 2*B, 2*C + 3*B]

s=2:  tail = win_base[2 + 2 - 1] = win_base[3] = 4
      vfslide1down 得 vin = [3, 4]；fmac 以 kptr[0] 即 A
      acc = [1*C + 2*B + 3*A, 2*C + 3*B + 4*A]

acc[0] = 列1 标量式, acc[1] = 列2 标量式；vse 写回后 i += vl
```

行左右界仍由条带外标量写 0。有限项和顺序与 `convolveRowsStandard` 一致，在 `float` 上应与标量式对齐。

#### `convolveCols` 与 `convolveColsRVV`（列 `i=0`、`height=8`、`vl=2` 条带覆盖行 2 与 3）

在「约定」中单列 `input(0, j)` 上，内区行 1 到 6 的标量加和为（与实现中 `in_p + i + j * width` 一致，此处 `i=0`，沿列 `j` 增大即沿行主序步进 `width` 个 `float`）：

```text
  out(0,j) = in(0,j-1)*C + in(0,j)*B + in(0,j+1)*A

j=1:  1*C + 2*B + 3*A
j=2:  2*C + 3*B + 4*A
j=3:  3*C + 4*B + 5*A
j=4:  4*C + 5*B + 6*A
j=5:  5*C + 6*B + 7*A
j=6:  6*C + 7*B + 8*A
```

`convolveColsRVV` 在条带起点行 j=2、`vl=2` 时，对 s=0、1、2 各一次 `__riscv_vlse32_v_f32m2`，列起点为 `i + (j - radius + s) * width`：

```text
s=0, 起点行 j-1 = 1:  vlse 得：列 [2, 3]  -> 乘 C 累入 acc
s=1, 起点行 j   = 2:  vlse 得：列 [3, 4]  -> 乘 B
s=2, 起点行 j+1 = 3:  vlse 得：列 [4, 5]  -> 乘 A

acc[0] = 2*C+3*B+4*A  对应 j=2
acc[1] = 3*C+4*B+5*A  对应 j=3
```

与上块中 j=2、j=3 两行标量式一致。各 tap 起点差一整行，不是行向一维连续窗的一格平移，故本路径不用 `__riscv_vfslide1down`（与 4.3 节实现一致）。

---

## 5. 测试与验证

### 5.1 测试入口与运行方式

#### 单测与本地 QEMU

- 路径：`test-rvv/common/gaussian/test_gaussian.cpp`；`test-rvv/common/gaussian/Makefile` 目标 `run_test` / `run_test_std` / `run_test_rvv`；默认以 `qemu-riscv64` 执行，CPU 与 `vlen` 以该 Makefile 中 `RUN_CMD` 为准。
- 可执行文件侧：`USE_PCL_RVV10=0` 与 `=1` 只影响是否向 bench/test 的翻译单元定义 `__RVV10__`（例如 banner），不改变已链接的 `libpcl_common` 中 `gaussian.cpp` 的机器码。单测验证的是当前安装前缀中的库与参考值是否一致，并非在同一进程内用两套不同 `.so` 对拍库内标量/RVV。与库无关的「同进程、仅验证算法两条路径是否逐元一致」见 5.4 节 `run_test_convolve_compare`。

#### 板卡性能对比

`GaussianKernel::convolveRows` / `convolveCols` 的实现位于共享库，是否调用 `*RVV` 或 `*Standard` 由把 `common/src/gaussian.cpp` 编入 `libpcl_common` 时是否定义 `__RVV10__` 决定。若只替换 `bench_gaussian_std` / `bench_gaussian_rvv` 而不重编库，两次 bench 仍可能执行同一条库内路径，表观加速比接近 $1$。因此，Std 与 RVV 的计时对比需两次在交叉工具链下构建 `pcl_common`，通过 `CMAKE_CXX_FLAGS` 中的 `ARCH_FLAGS` 区分是否带 `-D__RVV10__`（见 [RISC-V PCL Cross-Compilation Guide.zh.md](../build/RISC-V PCL Cross-Compilation Guide.zh.md) 第 9 节全量安装与第 10 节增量 `make pcl_common` + 拷贝 `libpcl_common.so*`）：一次产物中 `gaussian.cpp` 走 `convolveRowsStandard` / `convolveColsStandard`，一次走 `convolveRowsRVV` / `convolveColsRVV`。切换 `-D__RVV10__` 后应重新 `cmake` 或清理缓存再配置，否则 `CXXFLAGS` 可能仍用旧缓存。将安装前缀下的 `libpcl_common.so*` 同步到板卡 `LD_LIBRARY_PATH` 所指目录（例如 `test-rvv/common/gaussian` 的 `deploy_lib`）。

**目前操作顺序**（两次库构建之间勿混用同一前缀下的 `.so`）：

1. 配置时使 `ARCH_FLAGS` 不含 `-D__RVV10__`（仅 `-march=... -mabi=... -O3` 等），可参考 [编译指南](doc-rvv/build/RISC-V PCL Cross-Compilation Guide.zh.md) 第 10 节执行 `make pcl_common` 后将 `lib/libpcl_common.so*` 拷入安装前缀，得到标量卷积库。
2. 在 `test-rvv/common/gaussian` 执行 `make deploy_lib deploy_bench_std`，将库与 `bench_gaussian_std` 部署到板卡。
3. 在板卡工作目录执行 `make run_bench_std`（日志写入 `output/run_bench_std.log` 等路径，以板卡 Makefile 为准）。
4. 重新配置并使 `ARCH_FLAGS` 含 `-D__RVV10__`，再编译 `pcl_common` 并更新安装前缀中的 `libpcl_common.so*`。
5. 再次 `make deploy_lib deploy_bench_rvv`。
6. 在板卡执行 `make run_bench_rvv`，再执行 `make analyze_bench_compare`，得到 Std 与 RVV 并置的 Avg 与 Speedup。

汇总与原始 bench 输出可置于 `test-rvv/common/gaussian/output/board/`（以本仓库中实际文件名为准；若文件名为 `banch_compare.log` 则属拼写变体，与脚本一致即可）。

#### Benchmark 可执行文件说明

- 源文件：`test-rvv/common/gaussian/bench_gaussian.cpp`；`bench_gaussian_std` / `bench_gaussian_rvv` 的宏仅影响打印的 build 信息行，卷积热点仍解析自 `libpcl_common`。
- 可传参：`WIDTH HEIGHT ITERS SIGMA`（以 `main` 为准）。

#### 本文件性能数据来源

- 下表及 `test-rvv/common/gaussian/output/board/banch_compare.log` 所反映环境：板卡 Milkv-Jupiter ；Std 与 RVV 行分别对应上述两套库与同一套 bench 流程下的 Avg 列（见 5.2 节路径说明）。若未按两套库采集，表内 convolve 项 Speedup 可能失去对比意义，当前材料不足则未在本文另作推断。

### 5.2 日志与数据来源

- 性能表数据取自板卡侧对比输出副本：`test-rvv/common/gaussian/output/board/banch_compare.log`（与 `run_bench_std.log`、`run_bench_rvv.log` 同一轮 `960×540`、10 次迭代、`sigma=5`）。
- 仓库内 `test-rvv/common/gaussian/output/qemu/` 下保留的 `run_bench_std.log` 与 `run_bench_rvv.log` 在采集时分辨率与迭代次数不一致，未用于下表。
- 其他 vec-missed 与过滤日志位于 `test-rvv/common/gaussian/log/`，评估性文字见 `test-rvv/common/gaussian/gaussian-evaluation.zh.md`。

### 5.3 测试结果与说明


| Benchmark 项                                      | Std Avg (ms) | RVV Avg (ms) | Speedup |
| ------------------------------------------------ | ------------ | ------------ | ------- |
| compute(sigma, kernel) [gaussian.cpp]            | 0.0062       | 0.0062       | 1.00    |
| convolveRows (PointCloud) [gaussian.cpp]         | 53.6127      | 6.3381       | 8.46    |
| convolveCols (PointCloud) [gaussian.cpp]         | 105.7745     | 42.4242      | 2.49    |
| convolveRows then convolveCols [gaussian.cpp x2] | 160.0322     | 48.7031      | 3.29    |


`compute` 未走 RVV，两行时间相同符合预期。`convolveRows` 为连续读与滑动窗口，加速比高于 `convolveCols`；列方向 `vlse`/`vsse` 步长大，带宽与延迟行为不同，故 Speedup 较低。合并两步的 Speedup 介于两者之间，与两项耗时占比一致。

### 5.4 算法对拍

定位：在同一可执行文件、同一组翻译单元内对 float 可分离行/列卷积的标量与 RVV 实现做全缓冲逐元素 `float` 比较（`gtest` 的 `EXPECT_NEAR`），用于证明「算法上两条路径输出一致」。该流程不链接 `libpcl_common`（也无需 PCL 头/依赖），与 5.1 节中「需两套 `libpcl_common` 才能对比 bench 里库内实际走的是哪条机器码」是不同层面的结论：本节对拍回答的是 `RVV` 与 `Standard`  实现是否对齐；不替代在板卡上双库对装后的性能或「已安装库内」行为验证。

| 项 | 说明 |
| -- | -- |
| `test-rvv/common/gaussian/gaussian_convolve_float_local.hpp` | 与 `common/src/gaussian.cpp` 中匿名命名空间里 `convolve*Standard` / `convolve*RVV` 算法保持一致的人工同步副本；接口为行主序 `float*`、宽/高、一维核与 tap 数（奇数 `nkernel`），避免依赖 `PointCloud`/`Eigen`，便于无 PCL 链接。 |
| `test-rvv/common/gaussian/test_gaussian_convolve_compare.cpp` | 用自写一维高斯核生成系数，覆盖行卷积、列卷积、先 `convolveRows` 再 `convolveCols` 的可分离场；分别调用 `*Standard` 与 `*RVV` 后逐元比对（如容差 `1e-4f`）。 |
| 链接与宏 | 仅 `-lgtest -lpthread -lm`；`Makefile` 中 `CXXFLAGS_TEST_COMPARE` 固定 带 `-D__RVV10__`，不跟随全局 `USE_PCL_RVV10`，无需手写 `USE_PCL_RVV10=1`。 |

**如何运行（在 `test-rvv/common/gaussian` 目录）：**

```bash
make run_test_convolve_compare
```

默认用 `RUN_CMD_TEST_COMPARE`（`qemu-riscv64` 等，以该 Makefile 为准）执行，日志可写入 `output/qemu/run_test_convolve_compare.log`。

维护约定：若修改 `common/src/gaussian.cpp` 中 float 卷积循环，须同步更新 `gaussian_convolve_float_local.hpp`，否则本节对拍与库中实际行为可能脱节。

---

## 6. 总结

`PointCloud<float>` 的 `convolveRows` / `convolveCols` 在以 `__RVV10__` 编译本 TU 时走 `*RVV` 条带，否则走 `*Standard`；`gaussian.h` 中声明与边界语义与上游标量实现一致。主要优化为行向连续块上的 `__riscv_vle32`、滑动 `__riscv_vfslide1down` 与 FMA 累加，列向的 `__riscv_vlse32` / `__riscv_vsse32` 与 `__builtin_prefetch` 提示。`compute` 仍为标量。板卡上在分别安装标量与 RVV 两套 `libpcl_common` 的前提下，对 `960×540` 随机图、10 次迭代、`sigma=5` 的 bench 给出约 $8.5$ 倍、$2.5$ 倍与合并约 $3.3$ 倍量级的加速（见 5.3 表）。

若需从可执行文件/安装库侧对比「库内标量与 RVV 的逐输出或计时」，仍须双库对拍与 bench（5.1 节）或等效方式；`run_test` / 单次 `bench` 的宏不替换已链接的 `libpcl_common`。同进程、无 PCL 动态库的算法对拍由 5.4 节 `run_test_convolve_compare` 承担：验证 `gaussian_convolve_float_local.hpp` 中 `*Standard` 与 `*RVV` 一致，与双库对拍互补；改 `gaussian.cpp` 后须同步该副本。
