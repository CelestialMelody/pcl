# `selectWithinDistance` / `getDistancesToModel` ：RVV 优化实现说明

面向 RVV 1.0（`__RVV10__`），本文记录 `pcl::SampleConsensusModelNormalPlane` 中上述两个接口在本仓库相对上游 PCL 的差异、实现要点，以及测试与基准入口。与上游同名文件对照：[`sac_model_normal_plane.hpp`](https://github.com/PointCloudLibrary/pcl/blob/master/sample_consensus/include/pcl/sample_consensus/impl/sac_model_normal_plane.hpp)。

---

## 1. 背景

`SampleConsensusModelNormalPlane` 在 SAC 流水线里反复调用距离判定：按阈值筛内点（`selectWithinDistance`），或输出每个索引对应的标量距离（`getDistancesToModel`）。距离定义是几何项与法线角向项的加权组合：
\[
D = \left| w \cdot d_{ang} + (1-w)\cdot d_{geom} \right|,\quad
w = w_{user}\cdot(1-\text{curvature})
\]

其中 \(d_{geom} = |ax + by + cz + d|\)，\(d_{ang}\) 来自点法线与平面法线夹角并截断为锐角侧；\(w\) 随点曲率变化。

实现上的制约主要来自数据布局与访存形态：`indices_` 指向点云与法线数组，访问路径为「索引表 → `real_id` → AoS 中的坐标与法线」。该间接寻址阻断了沿 `i` 连续加载 `input_` 的假设；`selectWithinDistance` 的输出又是变长内点列表，不是与 `indices_` 等长的稠密数组。编译器自动向量化在这类路径上往往覆盖不全，因此在 RVV 上用手写 strip、`vluxei32` gather、掩码与压缩写回，与 `countWithinDistanceRVV`、平面距离/锐角等已有 `f32m2` 内核对齐，减少重复逻辑并便于后续维护。

---

## 2. 与上游实现的差异

### 2.1 RVV 运算类型

在 `#if defined(__RVV10__)` 下，`selectWithinDistance` 调用 `selectWithinDistanceRVV`，`getDistancesToModel` 调用 `getDistancesToModelRVV`。算术在 `float` 向量寄存器（`vfloat32m2_t` 等）中完成；接口仍向调用方提供 `std::vector<double>`（距离）与 `Indices`（内点索引），写回处通过 `vfwcvt` 将 `float` 宽化为 `double`，与 PCL 原有 API 类型保持一致。

### 2.2 数据写回

上游 `selectWithinDistance` 通常对 `inliers`、`error_sqr_dists_` 使用 `reserve` 后在命中时 `push_back`。该模式对「掩码 + 压缩」写回不匹配：`vcompress` 产出的是连续 lane 块，需要已知基址与偏移的一次性顺序写入，而不是由 `push_back` 隐式推进尾部。本仓库在入口层改为先按 `indices_->size()` `resize` 两处缓冲区，标量路径与 RVV 路径均按偏移写入，最后按实际内点数 `resize` 收缩。

上游`getDistancesToModel`按 `distances[i]` 顺序写回；本仓库主要增加 RVV 实现分支及上述 float 计算 / double 存储的拆分，不改变「第 `i` 个索引对应 `distances[i]`」的语义。

### 2.3 标量接口

标量逻辑抽成 `selectWithinDistanceStandard(..., i, current_count)`，用 `current_count` 表示下一次写入位置，与 RVV 侧 `nr_p` 的语义一致，便于与 SIMD 尾段或将来其他 ISA 扩展共用同一套「预分配 + 定址」模型。

---

## 3. 详细实现

### 3.1 入口：`resize`、分流、`resize` 收缩

预分配、`__RVV10__` 分支与最终收缩如下：

```cpp
// 66:87:sample_consensus/include/pcl/sample_consensus/impl/sac_model_normal_plane.hpp

  // --- 内存预分配的关键步骤 ---
  inliers.clear();
  error_sqr_dists_.clear();
  // 注意：使用 resize 而不是 reserve，以便我们可以直接通过指针写入
  inliers.resize(indices_->size());
  error_sqr_dists_.resize(indices_->size());

  std::size_t nr_p = 0; // 记录实际内点数量

#if defined (__RVV10__)
  // 调用 RVV 版本，返回找到的内点数量
  nr_p = selectWithinDistanceRVV(model_coefficients, threshold, inliers);
#else
  // 调用标准版本
  nr_p = selectWithinDistanceStandard(model_coefficients, threshold, inliers, 0, 0);
#endif

  // --- 收缩内存 ---
  // 将容器大小调整为实际内点数量
  inliers.resize(nr_p);
  error_sqr_dists_.resize(nr_p);
```

### 3.2 标量路径：定址写入

循环内通过 `(*indices_)[i]` 取点与法线，`getAngle3D` 参与角度项；命中阈值时写入 `inliers[current_count]`、`error_sqr_dists_[current_count]`。

```cpp
  for (; i < indices_->size (); ++i)
  {
    // 双重间接寻址：
    // indices_[i] -> 获取点在原始云中的真实 ID (Real ID)
    // (*input_)[Real ID] -> 获取具体的坐标点 PointT
    // (*normals_)[Real ID] -> 获取对应的法线点 PointNT
    const PointT  &pt = (*input_)[(*indices_)[i]];
    const PointNT &nt = (*normals_)[(*indices_)[i]];

    // 计算欧氏距离 (Euclidean Distance)
    // 公式：D_geom = |ax + by + cz + d|
    // coeff.dot(p) 计算 ax + by + cz
    // model_coefficients[3] 是 d
    Eigen::Vector4f p (pt.x, pt.y, pt.z, 0.0f);
    // 注意：这里定义的 n 只是为了给 getAngle3D 传参，实际上 NormalPlane 需要更高效的计算
    Eigen::Vector4f n (nt.normal_x, nt.normal_y, nt.normal_z, 0.0f);
    double d_euclid = std::abs (coeff.dot (p) + model_coefficients[3]);

    // 计算角度差异 (Angular Distance)
    // 计算点法线 n 与平面法线 coeff 的夹角 (弧度)
    // getAngle3D 内部通常计算 acos(dot(n1, n2))
    double d_normal = std::abs (getAngle3D (n, coeff));
    // 处理钝角情况：我们只关心锐角差异。如果夹角 > 90度 (PI/2)，说明法线反向了，取补角
    // 实际上 PCL 这里写的 min(d, PI-d) 是为了把范围限制在 [0, PI/2]
    d_normal = (std::min) (d_normal, M_PI - d_normal);

    // 计算权重
    // 基于曲率 (curvature) 的权重。
    // 如果曲率小（平坦表面），权重高，更多参考法线距离；
    // 如果曲率大（噪点或边缘），权重低，更多参考欧氏距离。
    double weight = normal_distance_weight_ * (1.0 - nt.curvature);

    // 综合距离计算
    // 线性插值混合两种距离
    double distance = std::abs (weight * d_normal + (1.0 - weight) * d_euclid);

    if (distance < threshold)
    {
      // 直接写入预分配的内存
      inliers[current_count] = (*indices_)[i];
      error_sqr_dists_[current_count] = distance;
      current_count++;
    }
  }
  return current_count;
```

### 3.3 `selectWithinDistanceRVV`：VLA、`vluxei32`、掩码、`vcompress`

- `__riscv_vsetvl_e32m2` 控制每轮 `vl`；
- `vle32` 加载 `indices_` 片段，索引乘以 `sizeof(PointT)` / `sizeof(PointNT)` 得到字节偏移，对坐标、法线、曲率做 `vluxei32`；
- 平面距离与锐角项调用 `SampleConsensusModelPlane<PointT>::distRVV_f32m2`、`getAcuteAngle3DRVV_f32m2`，与同文件中的 `countWithinDistanceRVV` 共用 `f32m2` 内核；
- `vmflt` 比较阈值，`vcpop` 计数；`active_count > 0` 时对索引与距离做 `vcompress`，距离经 `vfwcvt` 宽化后 `vse64` 写入 `error_sqr_dists_`。

```cpp
  for (; i < total_n; ) {
    // 动态设定向量长度
    const size_t vl = __riscv_vsetvl_e32m2(total_n - i);

    // --- A. 加载索引 ---
    const vuint32m2_t v_idx = __riscv_vle32_v_u32m2((const uint32_t*)(indices_ptr + i), vl);
    const vuint32m2_t v_off_pt = __riscv_vmul_vx_u32m2(v_idx, sizeof(PointT), vl);
    const vuint32m2_t v_off_norm = __riscv_vmul_vx_u32m2(v_idx, sizeof(PointNT), vl);

    // --- B. 加载数据 (Gather Load) ---
    // 加载 PointT (x, y, z)
    const vfloat32m2_t v_px = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, x)), v_off_pt, vl);
    const vfloat32m2_t v_py = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, y)), v_off_pt, vl);
    const vfloat32m2_t v_pz = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, z)), v_off_pt, vl);

    // 加载 PointNT (nx, ny, nz)
    const vfloat32m2_t v_nx = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_x)), v_off_norm, vl);
    const vfloat32m2_t v_ny = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_y)), v_off_norm, vl);
    const vfloat32m2_t v_nz = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_z)), v_off_norm, vl);

    // 加载 Curvature (单独加载)
    const vfloat32m2_t v_curv = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, curvature)), v_off_norm, vl);

    // --- C. 计算距离 (全程 Float) ---
    // 广播系数
    const vfloat32m2_t v_a = __riscv_vfmv_v_f_f32m2(a, vl);
    const vfloat32m2_t v_b = __riscv_vfmv_v_f_f32m2(b, vl);
    const vfloat32m2_t v_c = __riscv_vfmv_v_f_f32m2(c, vl);
    const vfloat32m2_t v_d = __riscv_vfmv_v_f_f32m2(d, vl);

    // Math Kernels
    const vfloat32m2_t v_d_euc = pcl::SampleConsensusModelPlane<PointT>::distRVV_f32m2(v_px, v_py, v_pz, v_a, v_b, v_c, v_d, vl);
    const vfloat32m2_t v_d_norm = pcl::getAcuteAngle3DRVV_f32m2(v_nx, v_ny, v_nz, v_a, v_b, v_c, vl);

    // Weight Calculation
    const vfloat32m2_t v_w = __riscv_vfmul_vf_f32m2(
                          __riscv_vfrsub_vf_f32m2(v_curv, 1.0f, vl), w_scalar, vl);

    // Final Distance
    vfloat32m2_t v_dist = __riscv_vfmacc_vv_f32m2(
                            __riscv_vfmul_vv_f32m2(v_w, v_d_norm, vl),
                            __riscv_vfrsub_vf_f32m2(v_w, 1.0f, vl), v_d_euc, vl);
    v_dist = __riscv_vfsgnjx_vv_f32m2(v_dist, v_dist, vl); // abs(dist)

    // --- D. 筛选与存储 (Select & Store) ---

    // 1. 生成 Mask
    const vbool16_t v_mask = __riscv_vmflt_vf_f32m2_b16(v_dist, th, vl);

    // 2. 统计有效数量
    long active_count = __riscv_vcpop_m_b16(v_mask, vl);

    if (active_count > 0) {
      // 1. 存储 Inliers (uint32_t)
      // m2 类型的索引压缩后依然是 m2
      vuint32m2_t v_idx_compressed = __riscv_vcompress_vm_u32m2(v_idx, v_mask, vl);
      __riscv_vse32_v_u32m2((uint32_t*)(inliers_out_ptr + nr_p), v_idx_compressed, active_count);

      // 2. 存储 Distances (float -> double)
      // a. 在 m2 级别进行压缩
      vfloat32m2_t v_dist_compressed = __riscv_vcompress_vm_f32m2(v_dist, v_mask, vl);

      // b. 由 m2 宽化为 m4。
      // 这是 RVV 硬件定义的 SEW*2 必须对应 LMUL*2 的要求
      vfloat64m4_t v_dist_double = __riscv_vfwcvt_f_f_v_f64m4(v_dist_compressed, active_count);

      // c. 使用 vse64 直接写回 double 数组
      __riscv_vse64_v_f64m4(dists_out_ptr + nr_p, v_dist_double, active_count);

      nr_p += active_count;
  }

    i += vl;
  }
```

实现上多出的代价是：在 `resize` 至 `indices_->size()` 后，内点数量远小于 `indices_->size()` 时，峰值会短暂占用满长度缓冲区；换得的是 `vcompress` 结果可连续 `vse` 写回。单轮循环内 `v_idx` 只加载一次，点与法线与 `v_off_pt` / `v_off_norm` 对齐。gather 与 `vluxseg3ei32` 的对比见 `doc-rvv/rvv/RVV Load Store Strategy.zh.md`。

### 3.4 `getDistancesToModel` 分流与 `getDistancesToModelRVV` 写回

对外接口在 `distances.resize` 后按 `__RVV10__` 选择实现：

```cpp
  distances.resize (indices_->size ());

#if defined (__RVV10__)
  // RVV 优化版本
  getDistancesToModelRVV(model_coefficients, distances);
#else
  // 标准版本
  getDistancesToModelStandard(model_coefficients, distances, 0);
#endif
```

`getDistancesToModelRVV` 无阈值与压缩；每轮 `vl` 个 `float` 距离宽化为 `double` 后顺序写入。

```cpp
    // 5. 写回：float → double
    // 遵循 SEW*2 对应 LMUL*2 的原则
    vfloat64m4_t v_final_dist_d = __riscv_vfwcvt_f_f_v_f64m4(v_final_dist_f, vl);
    __riscv_vse64_v_f64m4(dists_out_ptr + i, v_final_dist_d, vl);

    i += vl;
```

同文件 `getDistancesToModelRVV` 中保留了对 `vluxseg3ei32` 加载的注释块，便于在点类型或微架构变化时替换 gather 形态而不改动距离算术。

---

## 4. 测试与基准

### 4.1 数据与运行环境说明

测试与基准使用同一份点云数据文件 `sac_plane_test.pcd`。板卡（Milk-V Jupyter）侧的 bench 输出会打印该文件的点数（示例日志为 3283 points），并打印目标的 RVV 配置（示例日志为 `rv64gcv`，`VLEN 256-bit (zvl256b)`）。

单元测试中同时包含两类数据来源：

- **文件点云**：用于模型流程与回归（如 RANSAC/LMedS/MSAC 等用例）；
- **生成点云**：用于稳定复现 SIMD 路径差异与计时。板卡日志中 NormalPlane 的性能报告会显示“Points per cloud”，例如 `selectWithinDistance`/`getDistancesToModel` 采用 2000 点，`countWithinDistance` 采用 10000 点，并用固定次数迭代（示例日志为 1000 iterations）。

### 4.2 正确性验证

`test-rvv/sample_consensus/plane_models/test_sample_consensus_plane_models.cpp` 中 `SampleConsensusModelNormalPlane.SIMD_selectWithinDistance` 对比 `selectWithinDistanceStandard` 与 `selectWithinDistanceRVV`；`SIMD_getDistancesToModel` 对比 `getDistancesToModelStandard` 与 `getDistancesToModelRVV`，后者用 `EXPECT_NEAR` 约束数值误差。阈值边界附近对浮点舍入留有容差，避免把舍入差异当作实现错误。

```bash
cd test-rvv/sample_consensus/plane_models
make run_test
```

默认可通过 QEMU（如 `qemu-riscv64 -cpu rv64,v=true,vlen=256,elen=64`）在无硬件环境下执行；板卡上跑同一目标可反映真实 RVV 行为。同目录 `Makefile` 中另有 `deploy_test` 等目标供实际设备使用。

板卡上的输出日志位于 `test-rvv/sample_consensus/plane_models/output/board/`，常用文件包括：

- `test-nornmal-plane-rvv.log`：`make run_test` 的完整 GTest 输出，包含 correctness 与测试内的性能报告块；
- `bench-normal-plane-rvv.log`：`make run_bench` 的表格化基准输出，包含上下文（设备、VLEN、数据集点数、迭代次数）与每个条目的 Avg/Total/Speedup。

### 4.3 基准方法与结果

`test-rvv/sample_consensus/plane_models/bench_sac_normal_plane.cpp` 直接调用 `*_Standard` 与 `*_RVV`，绕过对外接口内的自动 dispatch，便于对比 `selectWithinDistance`、`countWithinDistance`、`getDistancesToModel` 三条路径。

```bash
cd test-rvv/sample_consensus/plane_models
make run_bench
```

未定义 `__RVV10__` 时 bench 会跳过 RVV 分支并打印警告。

板卡侧 `bench-normal-plane-rvv.log` 的一组样例（`Iterations: 50`，`Dataset: sac_plane_test.pcd (3283 points)`）如下。该表直接来自日志中的 Avg/Speedup 列，便于与后续改动对比：

| Item | Std Avg (ms) | RVV Avg (ms) | Speedup |
|---|---:|---:|---:|
| selectWithinDistance | 0.8148 | 0.0677 | 12.03x |
| countWithinDistance | 0.6924 | 0.0860 | 8.05x |
| getDistancesToModel | 0.7720 | 0.0896 | 8.62x |

`make run_test` 的 GTest 输出也会打印另一组以 1000 iterations 计时的报告块。考虑到本文聚焦两个接口，下面只摘录 `selectWithinDistance` 与 `getDistancesToModel` 两项（见 `test-rvv/sample_consensus/plane_models/output/board/test-nornmal-plane-rvv.log`）：

```text
[ Performance Report: NormalPlane (Select Within Distance) ]
Points per cloud    : 2000
Iterations          : 1000
Standard Total Time : 480.5778 ms
RVV Total Time      : 34.6177 ms
Speedup (Std/RVV)   : 13.8824x

[ Performance Report: NormalPlane (getDistancesToModel) ]
Points per cloud    : 2000
Iterations          : 1000
Standard Total Time : 452.0214 ms
RVV Total Time      : 35.5684 ms
Speedup (Std/RVV)   : 12.7085x
```

---

## 5. 总结

本仓库在 NormalPlane 上为 RVV 做的改动可归纳为三类：

1. 输出缓冲区从动态追加改为预分配定址，以支持掩码压缩写回；
2. 算术复用既有 `f32m2` 平面距离与锐角向量函数，减少分叉；
3. 对外仍保持 `double` 距离向量与原有索引语义。
