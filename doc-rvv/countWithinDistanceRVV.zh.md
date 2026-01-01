# RVV 实现：RANSAC 模型的 countWithinDistance 优化

本文档将深入探讨内存模型、算法原理，详细解析如何将 PCL RANSAC 模块 (包括 plane、 noraml_plane、circle、sphere) 中的内点统计 (countWithinDistance & dist) 从 AVX、SSE 迁移至 RISC-V Vector (RVV 1.0)。

---

## 内存模型分析与数学原理

### 1. 内存模型：AoS 布局与对齐

PCL 使用 Eigen 库进行数学运算，其核心数据结构采用了 AoS (Array of Structures) 布局以及 16字节对齐。

#### 1.1 点坐标 (`pcl::PointXYZ`)

根据 `point_types.hpp` 中的定义，`PointXYZ` 通过联合体 (`union`) 实现了数据复用与对齐：

```cpp
// point_types.hpp
struct _PointXYZ {
    PCL_ADD_POINT4D // 占用 16 字节
    PCL_MAKE_ALIGNED_OPERATOR_NEW
 };

#define PCL_ADD_POINT4D \
  PCL_ADD_UNION_POINT4D \
  PCL_ADD_EIGEN_MAPS_POINT4D

#define PCL_ADD_UNION_POINT4D \
  union EIGEN_ALIGN16 { \
    float data[4]; \
    struct { \
      float x; \
      float y; \
      float z; \
    }; \
  };

```

**内存视图**：在 `std::vector<PointXYZ>` 中，数据是紧密排列的。

| **偏移 (Offset)** | **内容** | **说明**                |
| ----------------------- | -------------- | ----------------------------- |
| `0x00`                | x              | 有效数据                      |
| `0x04`                | y              | 有效数据                      |
| `0x08`                | z              | 有效数据                      |
| `0x0C`                | *Padding*    | 填充 (通常为 1.0f 或无用数据) |

#### 1.2 法线数据 (`pcl::Normal`)

相比坐标点，法线结构体 `pcl::Normal` 更为庞大。它不仅包含法向量，还包含曲率，采用了双重 16 字节对齐。

```cpp
// point_types.hpp
struct EIGEN_ALIGN16 _Normal {
    PCL_ADD_NORMAL4D // Part 1: 法线 (16 bytes)
    union { // Part 2: 曲率 (16 bytes)
        struct { float curvature; };
        float data_c[4];
    };
    PCL_MAKE_ALIGNED_OPERATOR_NEW
};

#define PCL_ADD_POINT4D \
  PCL_ADD_UNION_POINT4D \
  PCL_ADD_EIGEN_MAPS_POINT4D

#define PCL_ADD_UNION_NORMAL4D \
  union EIGEN_ALIGN16 { \
    float data_n[4]; \
    float normal[3]; \
    struct { \
      float normal_x; \
      float normal_y; \
      float normal_z; \
    }; \
  };
```

**内存视图**：

| **偏移 (Hex)** | **偏移 (Dec)** | **变量**          | **类型** | **说明** |
| -------------------- | -------------------- | ----------------------- | -------------- | -------------- |
| `0x00`             | 0                    | **`normal_x`**  | float          | 法线 X         |
| `0x04`             | 4                    | **`normal_y`**  | float          | 法线 Y         |
| `0x08`             | 8                    | **`normal_z`**  | float          | 法线 Z         |
| `0x0C`             | 12                   | *Padding*             | float          | 填充           |
| `0x10`             | 16                   | **`curvature`** | float          | 曲率           |
| `0x14`             | 20                   | *Padding*             | float          | 填充           |
| `0x18`             | 24                   | *Padding*             | float          | 填充           |
| `0x1C`             | 28                   | *Padding*             | float          | 填充           |

### 2. 数据访问机制：双重间接寻址 (Double Indirection)

PCL RANSAC 算法从不直接遍历整个点云，而是基于一个索引数组 (`indices_`) 来访问点的子集。这导致了 **物理内存访问的非连续性**。

#### 2.1 坐标访问分析 (`PCLAT` 与 `AT`  宏) 解析

```cpp
// sac_model_plane.h
#define PCLAT(POS) ((*input_)[(*indices_)[(POS)]])

// sac_model_sphere.h, sac_model_circle.h
#define AT(POS) ((*input_)[(*indices_)[(POS)]])

// sac_model.h
  template <typename PointT>
  class SampleConsensusModel
  {
      using PointCloud = pcl::PointCloud<PointT>;
      using PointCloudConstPtr = typename PointCloud::ConstPtr;
      using PointCloudPtr = typename PointCloud::Ptr;
      SampleConsensusModel (const PointCloudConstPtr &cloud, // 传入实例数据指针
                  const std::vector<int> &indices,
                  bool random = false)
      // shared_ptr 拷贝构造，让成员变量 input_ 指向了堆内存中
      // 已存在的 PointCloud 对象，并将引用计数 +1
      : input_ (cloud) , indices_ (new std::vector<int> (indices))

      // 指向点云数据数组的 boost 共享指针
      PointCloudConstPtr input_;
      // 指向要使用的点索引向量的指针
      IndicesPtr indices_;
  }

// point_cloud.h
template <typename PointT>
class PointCloud
{
    // point data
    std::vector<PointT, Eigen::aligned_allocator<PointT> > points;

    using ConstPtr = shared_ptr<const PointCloud<PointT> >;
    // 对 PointCloud 类型的数据访问（先解引用，在运用运算符 [] 重载访问 points
    // 读写 points
    inline const PointT& operator[] (std::size_t n) const { return (points[n]); }
    inline PointT& operator[] (std::size_t n) { return (points[n]); }

    // 将点云复制到堆并返回一个智能指针
    // 会执行深拷贝，避免在非空点云上使用此函数
    // 返回的点云的更改不会反映回此点云
    inline Ptr makeShared () const { return Ptr (new PointCloud<PointT> (*this)); }
}
```

1. **`POS` (循环变量 `i`)**：
   - 逻辑索引，代表“当前 RANSAC 迭代中的第 i 个点”。
   
2. **`(*indices_)[POS]` (获取真实 ID)**：
   - `indices_` 类型是 `std::vector<int>`。
   - 这里获取的是点在原始点云中的真实物理 ID (Real ID)。
   - **内存特征**：虽然 `indices_` 数组本身是连续的，但里面的值可以是乱序的（例如：0, 105, 3, 99...）。
   
3. **`(*input_)[Real ID]` (获取数据)**：
   - `input_` 类型是 `shared_ptr<const PointCloud<PointT> >`。
   
   - `PointCloud` 类重载了 `operator[]`，利用乱序的 ID 去访问结构体数组。
   
   - **潜在问题**：每次内存访问都在整个点云内存空间中大范围跳跃，这对缓存（Cache）不友好，且无法使用普通的 SIMD 连续加载指令（因此建议使用 gather 指令）。
   
     > gather 指令是一种用于 **非连续内存加载** 的特殊向量指令。它允许从内存中的不连续地址一次性加载多个数据元素到一个向量寄存器中。

#### 2.2 法线访问分析

```cpp
// sac_model_normal_plane.hpp
((*normals_)[(*indices_)[i]]).normal_x
```

这与 PCLAT 的逻辑完全一致，基于 PCL 数据对齐：input_ (点) 和 normals_ (法线) 是两个独立的数组，但它们的大小相同，且索引一一对应。例如，Point[100] 的法线存储在 Normal[100] 中。

**推论（索引复用）**：同一个 Real ID = `(indices)[i]`，既用于在 `input_` 中找坐标，也用于在 `normals_` 中找法线。这意味着在 RVV 中，我们只需要加载一次索引向量。

### 3. 算法数学原理

#### 3.1 点到平面距离计算 (SampleConsensusModelPlane)

计算点 $P(x,y,z)$ 到平面 $ax+by+cz+d=0$ 的欧氏距离：

$$
d_{geom} = |ax + by + cz + d|
$$

#### 3.2 带法线点到平面距离综合计算 (SampleConsensusModelNormalPlane)

除了几何距离，还需计算点法线 $\vec{n_p}$ 与平面法线 $\vec{n_m}$ 的夹角差异，并根据点的曲率 $c$ 进行加权：

1. 角度距离：

   $$
   d_{ang} = getAcuteAngle(\vec{n_p}, \vec{n_m})
   $$
2. 权重：

   $$
   w = w_{user} \times (1.0 - c)
   $$
3. 最终距离：

   $$
   D = w \cdot d_{ang} + (1 - w) \cdot d_{geom}
   $$

#### 3.3 点到圆（2D）距离计算 (SampleConsensusModelCircle2D)

Circle2D 模型在三维点云中通常只处理 $X$ 和 $Y$ 维度（或投影到特定平面）。圆心 $(x_0, y_0)$，半径 $r$，欧式距离：

$$
d = \left| \sqrt{(x-x_0)^2 + (y-y_0)^2} - r \right|
$$

#### 3.4 点到球心距离计算 (SampleConsensusModelSphere)

Sphere 模型在三维空间中进行判定。球心 $(x_0, y_0, z_0)$，半径 $r$。欧式距离：

$$
d = \left| \sqrt{(x-x_0)^2 + (y-y_0)^2 + (z-z_0)^2} - r \right|
$$

#### 3.5 各模型内点判断方式 (Inlier Criteria)

为了优化计算性能（避免在循环中进行 `sqrt` 开方运算），PCL 的 SIMD 实现通常采用平方比较或区间判定。

##### Plane / NormalPlane

- **不带法线(Plane)** 内点条件：
  $$
  |ax + by + cz + d| < threshold
  $$
- **带法线 (NormalPlane)** 内点条件：

  $$
  |w \cdot d_{ang} + (1 - w) \cdot d_{geom}| < threshold
  $$

##### Circle2D / Sphere (平方区间法)

为了消除 `sqrt`，将距离公式

$$
d = |{dist\_to\_centor} - r| < threshold
$$

转换为以下逻辑：

1. 定义有效半径区间：$[r_{min}, r_{max}]$，其中 $r_{min}$ = r - threshold，$r_{max}$ = r + threshold，若 r < threshold，则 $r_{min}$ 取 0。
2. 计算点到中心（圆心/球心）的**平方距离** $d_{sq}$。
3. **判断逻辑**：若 $r_{min}^2$ < $d_{sq}$ < $r_{max}^2$，则该点为内点。

---

## 代码迁移分析：从 AVX 到 RVV

### 1.循环控制与尾部处理

由于 AVX 是固定长度向量，必须手动处理“尾部”；而 RVV 是变长向量，硬件自动处理“尾部”。

#### AVX 的实现方式 (定长 + 标量尾部)

AVX 寄存器固定处理 8 个 float。如果剩余点数不足 8 个，AVX 无法处理，必须退出循环，调用标量函数 `countWithinDistanceStandard` 处理剩余数据。

```cpp
// 步长固定为 8
for (; (i + 8) <= indices_->size (); i += 8) {
    // ... 向量化处理 8 个点 ...
}

// 问题：必须手动处理尾部 (0~7 个点)
// 增加了代码复杂度和二进制体积，且尾部无法享受加速
nr_p += countWithinDistanceStandard(model_coefficients, threshold, i);
```

#### RVV 的实现方式 (变长 VLA)

RVV 通过 `vsetvl` 指令，根据剩余元素数量 `n` 动态调整当前向量长度 `vl`。如果剩 100 个，`vl` 设为硬件最大值（如 8）。如果剩 3 个，`vl` 自动设为 3。因此不需要尾部处理函数。

```cpp
for (; i < total_n; ) {
    // 1. 自动计算本次处理的元素个数 vl
    // 当 total_n - i < VLMAX 时，vl 会自动变小
    size_t vl = __riscv_vsetvl_e32m2(total_n - i);

    // ... 向量化处理 vl 个点 ...

    // 2. 推进循环
    i += vl;
}
// 循环结束即处理完毕，无剩余数据
```

### 2.数据加载

基于之前的内存模型分析，在 RVV 中我们采用 Gather 策略。

#### AVX 的实现方式 (标量拼凑)

由于数据在内存中非连续（AoS + 间接寻址），AVX 这里使用手动使用 `_mm256_set_ps`。(可考虑使用 gather 指令改进)

```cpp
// AVX 代码
_mm256_set_ps (PCLAT(i  ).x, PCLAT(i+1).x, PCLAT(i+2).x, PCLAT(i+3).x, PCLAT(i+4).x, PCLAT(i+5).x, PCLAT(i+6).x, PCLAT(i+7).x)
```

#### RVV 实现 (Indexed Load / Gather)

RVV 提供了向量索引无序加载（indexed unordered load）指令 `vluxei32.v`。实现步骤如下：

1. **加载索引**：使用 `vle32.v` 从 `indices_` 数组中连续加载一批 Real IDs 到向量寄存器 `v_idx`。
2. **计算偏移**：
   - 对于坐标：`v_off_pt = v_idx * 16` (16 = sizeof PointXYZ)。
   - 对于法线：`v_off_norm = v_idx * 32` (32 = sizeof Normal)。
3. **Gather 数据**：
   - 使用 `vluxei32.v`，以 `points_base` 为基地址，`v_off_pt` 为偏移量，抓取 X, Y, Z。
   - 使用 `vluxei32.v`，以 `normals_base` 为基地址，`v_off_norm` 为偏移量，抓取 Nx, Ny, Nz。

```cpp
// RVV 实现
// 1. 加载索引向量 (0, 5, 12...)
vuint32m2_t v_idx = __riscv_vle32_v_u32m2((const uint32_t*)(indices_ptr + i), vl);
// 2. 计算字节偏移: offset = index * sizeof(PointT)
vuint32m2_t v_off_pt = __riscv_vmul_vx_u32m2(v_idx, sizeof(PointT), vl);
// 3. Gather X 分量: base_addr + offset + offsetof(x)
vfloat32m2_t v_px = __riscv_vluxei32_v_f32m2(
    (const float*)(points_base + offsetof(PointT, x)), // 基地址
    v_off_pt,                                          // 偏移向量
    vl                                                 // 向量长度
);
```

此外，为了精确地计算出结构体内部成员（如 `y` 或 `z`）的物理内存地址，我们将基地址强转为 `uint8_t*`，并手动计算字节偏移。

```cpp
// 1. 获取字节级基地址
const uint8_t* points_base = reinterpret_cast<const uint8_t*>(input_->points.data());

// 2. 计算字节偏移 (例如 offsetof(PointXYZ, y) = 4)
const vfloat32m2_t v_py = __riscv_vluxei32_v_f32m2((const float*)(points_base + offsetof(PointT, y)), v_off_pt, vl);
```

### 3.距离计算

以计算公式：$d = |ax + by + cz + d|$ 为例

#### AVX 的实现方式

```cpp
// 绝对值: 需要构造一个掩码 (0x80000000) 做 ANDNOT 操作
  return _mm256_andnot_ps (abs_help,
        _mm256_add_ps (_mm256_add_ps (
            _mm256_mul_ps (a_vec, _mm256_set_ps (PCLAT(i  ).x, PCLAT(i+1).x, PCLAT(i+2).x, PCLAT(i+3).x, PCLAT(i+4).x, PCLAT(i+5).x, PCLAT(i+6).x, PCLAT(i+7).x)), // a*x
            _mm256_mul_ps (b_vec, _mm256_set_ps (PCLAT(i  ).y, PCLAT(i+1).y, PCLAT(i+2).y, PCLAT(i+3).y, PCLAT(i+4).y, PCLAT(i+5).y, PCLAT(i+6).y, PCLAT(i+7).y))), // b*y
            _mm256_add_ps (_mm256_mul_ps (c_vec, _mm256_set_ps (PCLAT(i  ).z, PCLAT(i+1).z, PCLAT(i+2).z, PCLAT(i+3).z, PCLAT(i+4).z, PCLAT(i+5).z, PCLAT(i+6).z, PCLAT(i+7).z)), // c*z
             d_vec)) // 加上 d
```

#### RVV 的实现方式

```cpp
// res = a*x + b*y + c*z + d
const vfloat32m2_t res = __riscv_vfmacc_vv_f32m2(
                    __riscv_vfmacc_vv_f32m2(
                      __riscv_vfmacc_vv_f32m2(d_vec, a_vec, x_vec, vl),
                      b_vec, y_vec, vl),
                    c_vec, z_vec, vl);_c, v_pz, vl);

// 绝对值: 使用 "异或符号位注入" 指令，无需额外加载掩码常量
dist = __riscv_vfsgnjx_vv_f32m2(dist, dist, vl);
```

### 4.双边界判断

在 Circle2D 和 Sphere 模型中，内点被定义为位于一个“球壳”（或“圆环”）内的点。这需要同时满足两个不等式条件。

#### AVX 的实现方式 (组合掩码)

AVX 通过两次独立的比较操作生成两个浮点掩码向量，然后使用按位与（`_mm256_and_ps`）将它们组合成最终的内点掩码。

```cpp
// 1. 计算平方距离向量
const __m256 sqr_dist = ...;

// 2. 生成两个条件掩码
//    mask1: sqr_dist > sqr_inner_radius
//    mask2: sqr_dist < sqr_outer_radius
const __m256 mask1 = _mm256_cmp_ps(sqr_inner_radius, sqr_dist, _CMP_LT_OQ); // inner < dist
const __m256 mask2 = _mm256_cmp_ps(sqr_dist, sqr_outer_radius, _CMP_LT_OQ); // dist < outer

// 3. 组合掩码: inlier = mask1 AND mask2
const __m256 inlier_mask = _mm256_and_ps(mask1, mask2);
```

此方法逻辑清晰，但需要两条比较指令和一条逻辑与指令，并且生成的是浮点类型的全零/全一掩码，后续还需转换为整数才能计数。

#### RVV 的实现方式 (谓词掩码与逻辑操作)

RVV 的设计天然支持高效的条件分支和掩码操作。其比较指令直接返回**压缩布尔掩码**（`vbool16_t`），并提供了专用的布尔逻辑运算指令。

```cpp
// 1. 计算平方距离向量
const vfloat32m2_t v_sqr_dist = ...;

// 2. 直接生成布尔掩码
//    mask_inner: v_sqr_dist > sqr_inner_radius
//    mask_outer: v_sqr_dist < sqr_outer_radius
const vbool16_t mask_inner = __riscv_vmfgt_vf_f32m2_b16(v_sqr_dist, sqr_inner_radius, vl);
const vbool16_t mask_outer = __riscv_vmflt_vf_f32m2_b16(v_sqr_dist, sqr_outer_radius, vl);

// 3. 使用专用布尔“与”指令组合
const vbool16_t inliers_mask = __riscv_vmand_mm_b16(mask_inner, mask_outer, vl);
```

### 5.掩码统计

> 在 Circle2D 和 Sphere 模型中，掩码统计与 *4.双边界判断* 内容大部分重合，还有些收尾工作与下文相同。
>

对于平面模型，以 NoramlPlane 模型为例，给定浮点距离值的向量，判断每个距离是否小于阈值 `threshold`，若是，则视为“内点（inlier）”，最终统计内点总数 `nr_p`。

#### AVX 的实现方式 (向量累加 + 水平归约)

AVX2 没有直接“统计掩码中 1 的个数”的指令。这里的做法是将掩码当作整数 0 或 1，加到向量计数器中，最后再把向量里的 8 个数加起来。

```cpp
// 1. 向量比较：生成掩码向量（float 类型，每个元素为全0或全1）
const __m256 mask = _mm256_cmp_ps(dist_vec, threshold_vec, _CMP_LT_OQ);
// 2. castps 将 float 掩码转为整型，并与全1向量按位与，得到 0/1 整数向量
res = _mm256_add_epi32(res,
        _mm256_and_si256(
            _mm256_set1_epi32(1),
            _mm256_castps_si256(mask)
        )
    );
// ... 循环结束后 ...
// 3. 将 res 中的 8 个整数提取出来相加
nr_p += _mm256_extract_epi32(res, 0);
nr_p += _mm256_extract_epi32(res, 1);
// ... 重复 8 次 ...
```

#### RVV 实现 (Mask PopCount)

RVV 拥有独立的 Mask 寄存器 和 Population Count 指令。

```cpp
// 1. 比较生成掩码 (Result stored in mask register v0)
// vmflt: Vector Mask Float Less Than
vbool16_t v_mask = __riscv_vmflt_vf_f32m2_b16(v_dist, t, vl); // sac_model_normal_plane.hpp

// 2. 直接统计: vcpop 指令返回 mask 中 1 的个数
// 这是一个标量结果，直接加到总数 nr_p
nr_p += __riscv_vcpop_m_b16(v_mask, vl);
```

### 小结

| **操作**  | **SSE**                                | **AVX2**                                     | **RVV**                           |
| --------------- | -------------------------------------------- | -------------------------------------------------- | --------------------------------------- |
| 广播中心坐标    | `_mm_set1_ps`                              | `_mm256_set1_ps`                                 | `__riscv_vfmv_v_f_f32m2`              |
| 计算平方距离    | 自定义 `sqr_dist4`                         | 自定义 `sqr_dist8`                               | 自定义 `sqr_distRVV`                  |
| 双边界判断      | `_mm_cmplt_ps` ×2 + `_mm_and_ps`        | `_mm256_cmp_ps` ×2 + `_mm256_and_ps`          | `vmflt` + `vmfgt` + `vmand`       |
| 转换为 0/1 计数 | `_mm_castps_si128 + _mm_and_si128(1)`      | `_mm256_castps_si256 + _mm256_and_si256(1)`      | **无需转换**                      |
| 累计 inlier 数  | `_mm_add_epi32` + 4×`_mm_extract_epi32` | `_mm256_add_epi32` + 8×`_mm256_extract_epi32` | `__riscv_vcpop_m_b16`（单指令）       |
| 处理尾部元素    | 调用 `countWithinDistanceStandard`         | 同左                                               | **自动处理（VLA）**，无需额外代码 |

---

## 指令集函数速查手册

为了方便理解 SIMD 代码，以下列出了本文涉及的关键 Intrinsic 函数说明。

### AVX Intrinsic (Intel x86)

| **函数名**                                   | **功能说明**                                                       | **参数含义**                                                | **返回值**       | **备注**                                     |
| -------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------- | ---------------------- | -------------------------------------------------- |
| `_mm256_set1_ps(float a)`                        | **广播**：将标量 `a` 复制 8 次，填满 256 位 float 向量           | `a`: 要广播的单精度浮点数                                       | `__m256`             | 高效初始化常量向量                                 |
| `_mm256_set_ps(e7,e6,...,e0)`                    | **构造**：用 8 个 float **逆序** 构造向量（e7 → lane 0）    | `e7` 到 `e0`：8 个 float 常量或变量                           | `__m256`             | 参数顺序反直觉；优先用 `_mm256_load_ps`          |
| `_mm256_load_ps(const float* p)`                 | **加载**：从对齐地址加载 8 个连续 float                            | `p`: 32 字节对齐的 float 数组指针                               | `__m256`             | 若未对齐，用 `_mm256_loadu_ps`                   |
| `_mm256_add_ps(__m256 a, __m256 b)`              | **加法**：逐元素相加                                               | `a`, `b`: 两个 float 向量                                     | `__m256`             | —                                                 |
| `_mm256_mul_ps(__m256 a, __m256 b)`              | **乘法**：逐元素相乘                                               | `a`, `b`: 两个 float 向量                                     | `__m256`             | —                                                 |
| `_mm256_cmp_ps(a, b, imm8)`                      | **比较**：逐元素比较，返回掩码向量（全 0 或全 1）                  | `a`, `b`: 输入向量；`imm8`: 比较操作码（如 `_CMP_LT_OQ`） | `__m256`             | `_CMP_LT_OQ` = “小于”，有序且不抛 NaN 异常     |
| `_mm256_and_ps(a, b)`                            | **按位与**：常用于掩码应用                                         | `a`, `b`: 两个向量（通常一个是数据，一个是掩码）              | `__m256`             | —                                                 |
| `_mm256_andnot_ps(a, b)`                         | **按位与非**：计算 `(~a) & b`，常用于清除某些位                  | `a`: 掩码（被取反）；`b`: 数据                                | `__m256`             | 取绝对值常用：`_mm256_andnot_ps(sign_mask, x)`   |
| `_mm256_movemask_ps(__m256 a)`                   | **提取符号位**：将每个 float 元素的符号位（bit 31）组合成 8 位整数 | `a`: float 向量                                                 | `int`（低 8 位有效） | bit i 对应 `a[i]` 的符号；负数/−0 → 1          |
| `_mm256_castps_si256(__m256 a)`                  | **类型转换**：将 float 向量 reinterpret 为整型向量（不改变位模式） | `a`: float 向量                                                 | `__m256i`            | 用于后续整数运算（如 `_mm256_and_si256`）        |
| `_mm256_set1_epi32(int a)`                       | **广播整数**：将 32 位整数复制 8 次                                | `a`: 要广播的整数                                               | `__m256i`            | —                                                 |
| `_mm256_and_si256(__m256i a, __m256i b)`         | **整数按位与**                                                     | `a`, `b`: 整型向量                                            | `__m256i`            | 用于生成 0/1 计数向量                              |
| `_mm256_add_epi32(__m256i a, __m256i b)`         | **整数加法**：32 位整数逐元素相加                                  | `a`, `b`: 整型向量                                            | `__m256i`            | 累加 inlier 计数                                   |
| `_mm256_extract_epi32(__m256i a, const int imm)` | **提取整数元素**：从向量中提取第 `imm` 个 32 位整数              | `a`: 整型向量；`imm`: **编译时常量**，范围 0–7         | `int`                | 当向量来自 `load` 时，`imm=i` 对应 `data[i]` |

### RVV Intrinsic (RISC-V Vector Extension)

RVV 函数命名规则：`__riscv_<op>_<type><lmul>[_suffix]`。

- `f32m2`：float32，LMUL=2（使用 2 个向量寄存器）
- `b16`：布尔掩码，每个 active 元素占 1 bit，共 16 位
- 所有函数最后参数为 `vl`（Vector Length），表示当前处理的有效元素数。

| **函数名**                                                              | **功能说明**                                                                                                                                       | **参数含义**                                                                 | **返回值**     | **备注**                       |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | -------------------- | ------------------------------------ |
| `__riscv_vsetvl_e32m2(size_t avl)`                                          | **自动定长 (VLA)**：申请处理 `avl` 个元素。硬件返回实际能处理的个数 `vl`。若 `avl > 硬件最大值`，返回最大值；若 `avl` 较小，返回 `avl`。 | `avl`: Application Vector Length（如剩余数组元素个数）                           | `size_t vl`        | 必须在向量操作前调用                 |
| `__riscv_vle32_v_u32m2(const uint32_t* base, size_t vl)`                    | **连续加载**：加载 `vl` 个 32 位无符号整数（用于索引）                                                                                           | `base`: 索引数组起始地址；`vl`: 向量长度                                       | `vuint32m2_t`      | PCL 中用于加载 `indices_`          |
| `__riscv_vmul_vx_u32m2(vuint32m2_t v, uint32_t x, size_t vl)`               | **向量×标量乘法**：计算字节偏移 `offset[i] = index[i] * sizeof(PointT)`                                                                         | `v`: 索引向量；`x`: `sizeof(PointT)`；`vl`: 向量长度                       | `vuint32m2_t`      | 为 gather 提供字节偏移               |
| `__riscv_vluxei32_v_f32m2(const float* base, vuint32m2_t index, size_t vl)` | **索引加载（Gather）**：`result[i] = base[index[i]]`                                                                                             | `base`: 基地址（如 `&points[0].x`）；`index`: 字节偏移向量；`vl`: 向量长度 | `vfloat32m2_t`     | 支持 AoS 布局，无需 struct-of-arrays |
| `__riscv_vfmv_v_f_f32m2(float scalar, size_t vl)`                           | **广播**：将标量 `scalar` 复制到 `vl` 个元素                                                                                                   | `scalar`: 标量值；`vl`: 向量长度                                               | `vfloat32m2_t`     | 类似 `_mm256_set1_ps`              |
| `__riscv_vmflt_vf_f32m2_b16(v, float scalar, vl)`                           | **向量-标量比较（小于）**：`mask[i] = (v[i] < scalar)`                                                                                           | `v`: float 向量；`scalar`: 标量阈值；`vl`: 向量长度                          | `vbool16_t`        | 返回压缩布尔掩码                     |
| `__riscv_vmfgt_vf_f32m2_b16(v, float scalar, vl)`                           | **向量-标量比较（大于）**：`mask[i] = (v[i] > scalar)`                                                                                           | `v`: float 向量；`scalar`: 标量阈值；`vl`: 向量长度                          | `vbool16_t`        | 用于 `sqr_dist > sqr_inner_radius` |
| `__riscv_vmand_mm_b16(vbool16_t a, vbool16_t b, size_t vl)`                 | **布尔掩码“与”**：`result[i] = a[i] && b[i]`                                                                                                   | `a`, `b`: 两个布尔掩码；`vl`: 向量长度                                       | `vbool16_t`        | 组合内外球壳条件                     |
| `__riscv_vcpop_m_b16(vbool16_t mask, size_t vl)`                            | **掩码 popcount**：统计 `mask` 中为 true 的元素个数                                                                                              | `mask`: 布尔掩码；`vl`: 向量长度                                               | `size_t`（计数值） | 一条指令完成计数                     |
