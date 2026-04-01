# PCL 编译告警/测试差异汇总与修复记录

本文用于记录在 RISC-V / RVV10（Eigen RVV 分支）环境中，编译/运行 PCL 过程中遇到的典型告警与问题，以及对应的修复方式。

---

## 1. Eigen `cross3()`：used but never defined

### 现象

在编译 `pcl_common`（例如 `common/src/distances.cpp`）时出现类似告警：

- `Eigen::MatrixBase::cross3(...) used but never defined`

### 原因

PCL 的 `pcl/common/distances.h` 中使用了 `cross3()`：

- `sqrPointToLineDistance()` 里调用 `line_dir.cross3(line_pt - pt)`

但是该头文件只包含了 `<Eigen/Core>`。在当前 Eigen 版本中：

- `<Eigen/Core>` 里只有 `cross3()` 的声明
- `cross3()` 的定义在 `<Eigen/Geometry>`（`OrthoMethods.h`）中

因此，在只包含 `<Eigen/Core>` 的编译单元里，会出现“使用了内联函数但找不到定义”的告警。

### 修复

在 `common/include/pcl/common/distances.h` 中增加：

- `#include <Eigen/Geometry>`

---

## 2. Eigen `jacobiSvd(...)`：deprecated（ComputeThinU/ComputeThinV）

### 现象

在编译/链接涉及 `sac_model_torus` 的目标时出现告警：

- `MatrixBase::jacobiSvd(unsigned int) is deprecated`
- 提示：Options should be specified using method's template parameter

### 原因

旧写法使用了：

- `A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);`

在较新 Eigen 版本中，此 API 被弃用，推荐改为模板参数指定 Options。

### 修复

在 `sample_consensus/include/pcl/sample_consensus/impl/sac_model_torus.hpp` 中修改为：

- `A.template jacobiSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(B);`

---

## 3. Eigen `eulerAngles(...)`：deprecated（改 `canonicalEulerAngles(...)`）

### 现象

编译以下模块时出现告警：

- `MatrixBase::eulerAngles(...) is deprecated: Use .canonicalEulerAngles() instead`

典型位置包括：

- `recognition/src/face_detection/face_detector_data_provider.cpp`
- `recognition/src/face_detection/rf_face_detector_trainer.cpp`
- `registration/include/pcl/registration/impl/ndt.hpp`

### 原因

Eigen 新版本将 `eulerAngles(...)` 标记为 deprecated，要求使用 `canonicalEulerAngles(...)`。

### 修复

将以下调用替换：

- `...eulerAngles(0, 1, 2)` → `...canonicalEulerAngles(0, 1, 2)`

已修改的具体点位：

- `recognition/src/face_detection/face_detector_data_provider.cpp`（两处）
- `recognition/src/face_detection/rf_face_detector_trainer.cpp`（一处）
- `registration/include/pcl/registration/impl/ndt.hpp`（一处）

