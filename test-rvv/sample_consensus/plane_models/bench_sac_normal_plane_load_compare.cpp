#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h> // getAcuteAngle3DRVV_f32m2
#include <pcl/common/utils.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#if defined(__RVV10__)
#if !defined(__riscv_vector)
#error "Expected RVV intrinsics available when __RVV10__ is defined."
#endif
#include <riscv_vector.h>
#endif

using PointT = pcl::PointXYZ;
using PointNT = pcl::Normal;

static inline double
now_ms()
{
  return std::chrono::duration<double, std::milli>(
           std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

#if defined(__RVV10__)
static std::size_t
countWithinDistanceRVV_gather(const pcl::PointCloud<PointT>& cloud,
                              const pcl::PointCloud<PointNT>& normals,
                              const std::vector<pcl::index_t>& indices,
                              const Eigen::Vector4f& model_coefficients,
                              double threshold,
                              double normal_distance_weight)
{
  const float a = model_coefficients[0];
  const float b = model_coefficients[1];
  const float c = model_coefficients[2];
  const float d = model_coefficients[3];
  const float th = static_cast<float>(threshold);
  const float w_scalar = static_cast<float>(normal_distance_weight);

  const pcl::index_t* indices_ptr = indices.data();
  const uint8_t* points_base = reinterpret_cast<const uint8_t*>(cloud.points.data());
  const uint8_t* normals_base = reinterpret_cast<const uint8_t*>(normals.points.data());

  std::size_t nr_p = 0;
  const std::size_t total_n = indices.size();
  for (std::size_t i = 0; i < total_n;) {
    const size_t vl = __riscv_vsetvl_e32m2(total_n - i);
    const vuint32m2_t v_idx = __riscv_vle32_v_u32m2((const uint32_t*)(indices_ptr + i), vl);

    const vfloat32m2_t v_a = __riscv_vfmv_v_f_f32m2(a, vl);
    const vfloat32m2_t v_b = __riscv_vfmv_v_f_f32m2(b, vl);
    const vfloat32m2_t v_c = __riscv_vfmv_v_f_f32m2(c, vl);
    const vfloat32m2_t v_d = __riscv_vfmv_v_f_f32m2(d, vl);

    const vuint32m2_t v_off_pt = __riscv_vmul_vx_u32m2(v_idx, sizeof(PointT), vl);
    const vuint32m2_t v_off_norm = __riscv_vmul_vx_u32m2(v_idx, sizeof(PointNT), vl);

    const vfloat32m2_t v_px = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, x)), v_off_pt, vl);
    const vfloat32m2_t v_py = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, y)), v_off_pt, vl);
    const vfloat32m2_t v_pz = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, z)), v_off_pt, vl);

    const vfloat32m2_t v_nx = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_x)), v_off_norm, vl);
    const vfloat32m2_t v_ny = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_y)), v_off_norm, vl);
    const vfloat32m2_t v_nz = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_z)), v_off_norm, vl);
    const vfloat32m2_t v_curv = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, curvature)), v_off_norm, vl);

    // Euclidean term: |a*x + b*y + c*z + d|
    const vfloat32m2_t v_euc = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmacc_vv_f32m2(__riscv_vfmacc_vv_f32m2(v_d, v_a, v_px, vl), v_b, v_py, vl),
        v_c,
        v_pz,
        vl);
    const vfloat32m2_t v_d_euc = __riscv_vfsgnjx_vv_f32m2(v_euc, v_euc, vl);

    // Angular term: acute angle between (nx,ny,nz) and plane normal (a,b,c)
    const vfloat32m2_t v_d_norm = pcl::getAcuteAngle3DRVV_f32m2(v_nx, v_ny, v_nz, v_a, v_b, v_c, vl);

    const vfloat32m2_t v_w = __riscv_vfmul_vf_f32m2(
        __riscv_vfrsub_vf_f32m2(v_curv, 1.0f, vl), w_scalar, vl);

    vfloat32m2_t v_dist = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmul_vv_f32m2(v_w, v_d_norm, vl),
        __riscv_vfrsub_vf_f32m2(v_w, 1.0f, vl), v_d_euc, vl);
    v_dist = __riscv_vfsgnjx_vv_f32m2(v_dist, v_dist, vl);

    const vbool16_t v_mask = __riscv_vmflt_vf_f32m2_b16(v_dist, th, vl);
    nr_p += __riscv_vcpop_m_b16(v_mask, vl);

    i += vl;
  }
  return nr_p;
}

static std::size_t
countWithinDistanceRVV_vluxseg3(const pcl::PointCloud<PointT>& cloud,
                                const pcl::PointCloud<PointNT>& normals,
                                const std::vector<pcl::index_t>& indices,
                                const Eigen::Vector4f& model_coefficients,
                                double threshold,
                                double normal_distance_weight)
{
  const float a = model_coefficients[0];
  const float b = model_coefficients[1];
  const float c = model_coefficients[2];
  const float d = model_coefficients[3];
  const float th = static_cast<float>(threshold);
  const float w_scalar = static_cast<float>(normal_distance_weight);

  const pcl::index_t* indices_ptr = indices.data();
  const uint8_t* points_base = reinterpret_cast<const uint8_t*>(cloud.points.data());
  const uint8_t* normals_base = reinterpret_cast<const uint8_t*>(normals.points.data());

  std::size_t nr_p = 0;
  const std::size_t total_n = indices.size();
  for (std::size_t i = 0; i < total_n;) {
    const size_t vl = __riscv_vsetvl_e32m2(total_n - i);
    const vuint32m2_t v_idx = __riscv_vle32_v_u32m2((const uint32_t*)(indices_ptr + i), vl);

    const vfloat32m2_t v_a = __riscv_vfmv_v_f_f32m2(a, vl);
    const vfloat32m2_t v_b = __riscv_vfmv_v_f_f32m2(b, vl);
    const vfloat32m2_t v_c = __riscv_vfmv_v_f_f32m2(c, vl);
    const vfloat32m2_t v_d = __riscv_vfmv_v_f_f32m2(d, vl);

    const vuint32m2_t v_off_pt = __riscv_vmul_vx_u32m2(v_idx, sizeof(PointT), vl);
    const vuint32m2_t v_off_norm = __riscv_vmul_vx_u32m2(v_idx, sizeof(PointNT), vl);

    const vfloat32m2x3_t v_xyz = __riscv_vluxseg3ei32_v_f32m2x3(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, x)), v_off_pt, vl);
    const vfloat32m2_t v_px = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 0);
    const vfloat32m2_t v_py = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 1);
    const vfloat32m2_t v_pz = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 2);

    const vfloat32m2x3_t v_nxyz = __riscv_vluxseg3ei32_v_f32m2x3(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_x)), v_off_norm, vl);
    const vfloat32m2_t v_nx = __riscv_vget_v_f32m2x3_f32m2(v_nxyz, 0);
    const vfloat32m2_t v_ny = __riscv_vget_v_f32m2x3_f32m2(v_nxyz, 1);
    const vfloat32m2_t v_nz = __riscv_vget_v_f32m2x3_f32m2(v_nxyz, 2);

    const vfloat32m2_t v_curv = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, curvature)), v_off_norm, vl);

    const vfloat32m2_t v_euc = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmacc_vv_f32m2(__riscv_vfmacc_vv_f32m2(v_d, v_a, v_px, vl), v_b, v_py, vl),
        v_c,
        v_pz,
        vl);
    const vfloat32m2_t v_d_euc = __riscv_vfsgnjx_vv_f32m2(v_euc, v_euc, vl);
    const vfloat32m2_t v_d_norm = pcl::getAcuteAngle3DRVV_f32m2(v_nx, v_ny, v_nz, v_a, v_b, v_c, vl);

    const vfloat32m2_t v_w = __riscv_vfmul_vf_f32m2(
        __riscv_vfrsub_vf_f32m2(v_curv, 1.0f, vl), w_scalar, vl);

    vfloat32m2_t v_dist = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmul_vv_f32m2(v_w, v_d_norm, vl),
        __riscv_vfrsub_vf_f32m2(v_w, 1.0f, vl), v_d_euc, vl);
    v_dist = __riscv_vfsgnjx_vv_f32m2(v_dist, v_dist, vl);

    const vbool16_t v_mask = __riscv_vmflt_vf_f32m2_b16(v_dist, th, vl);
    nr_p += __riscv_vcpop_m_b16(v_mask, vl);

    i += vl;
  }
  return nr_p;
}
#endif

int
main(int argc, char** argv)
{
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " sac_plane_test.pcd [iters]\n";
    return 2;
  }

  const char* pcd_path = argv[1];
  int iters = 200;
  if (argc >= 3)
    iters = std::max(1, std::atoi(argv[2]));

  pcl::PCLPointCloud2 cloud_blob;
  if (pcl::io::loadPCDFile(pcd_path, cloud_blob) < 0) {
    std::cerr << "Failed to read test file: " << pcd_path << "\n";
    return 2;
  }

  pcl::PointCloud<PointT> cloud;
  pcl::PointCloud<PointNT> normals;
  pcl::fromPCLPointCloud2(cloud_blob, cloud);
  pcl::fromPCLPointCloud2(cloud_blob, normals);

  std::vector<pcl::index_t> indices(cloud.size());
  for (std::size_t i = 0; i < indices.size(); ++i)
    indices[i] = static_cast<pcl::index_t>(i);

  Eigen::Vector4f coeffs;
  coeffs[0] = -0.8964f;
  coeffs[1] = -0.5868f;
  coeffs[2] = -1.208f;
  coeffs[3] = 1.0f;

  const double threshold = 0.05;
  const double weight = 0.1;

#if !defined(__RVV10__)
  std::cout << "[SKIP] __RVV10__ not enabled; cannot run RVV load-compare benchmark.\n";
  pcl::utils::ignore(indices, coeffs, threshold, weight);
  return 0;
#else
  // Warmup
  constexpr int warmup = 5;
  for (int i = 0; i < warmup; ++i) {
    pcl::utils::ignore(countWithinDistanceRVV_gather(cloud, normals, indices, coeffs, threshold, weight));
    pcl::utils::ignore(countWithinDistanceRVV_vluxseg3(cloud, normals, indices, coeffs, threshold, weight));
  }

  // Measure
  double t0 = now_ms();
  std::size_t res_gather = 0;
  for (int i = 0; i < iters; ++i)
    res_gather += countWithinDistanceRVV_gather(cloud, normals, indices, coeffs, threshold, weight);
  double t1 = now_ms();

  double t2 = now_ms();
  std::size_t res_seg = 0;
  for (int i = 0; i < iters; ++i)
    res_seg += countWithinDistanceRVV_vluxseg3(cloud, normals, indices, coeffs, threshold, weight);
  double t3 = now_ms();

  // Correctness: per-iter count must match.
  const std::size_t one_gather =
      countWithinDistanceRVV_gather(cloud, normals, indices, coeffs, threshold, weight);
  const std::size_t one_seg =
      countWithinDistanceRVV_vluxseg3(cloud, normals, indices, coeffs, threshold, weight);
  if (one_gather != one_seg) {
    std::cerr << "[FAIL] gather vs vluxseg3 count mismatch: gather=" << one_gather
              << ", vluxseg3=" << one_seg << "\n";
    return 1;
  }

  const double gather_ms = (t1 - t0) / iters;
  const double seg_ms = (t3 - t2) / iters;
  const double speedup = (seg_ms > 0.0) ? (gather_ms / seg_ms) : 0.0;

  std::cout << "NormalPlane Load Strategy Compare (RVV):\n";
  std::cout << "  dataset: " << pcd_path << " (points=" << cloud.size() << ")\n";
  std::cout << "  iters=" << iters << ", warmup=" << warmup << "\n";
  std::cout << "  threshold=" << threshold << ", normal_weight=" << weight << "\n";
  std::cout << "  gather(vluxei32):    " << std::fixed << std::setprecision(6) << gather_ms << " ms/iter\n";
  std::cout << "  vluxseg3(vluxseg3ei32): " << std::fixed << std::setprecision(6) << seg_ms << " ms/iter\n";
  std::cout << "  speedup (gather/seg): " << std::fixed << std::setprecision(3) << speedup << "x\n";
  std::cout << "  check_count=" << one_gather << "\n";

  // Prevent DCE of whole benchmark in LTO-ish scenarios.
  pcl::utils::ignore(res_gather, res_seg);
  return 0;
#endif
}

