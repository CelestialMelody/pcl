#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/sac_model_normal_plane.h>
#include <pcl/common/utils.h>

#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <functional>

using PointT = pcl::PointXYZ;
using PointNT = pcl::Normal;
using ModelT = pcl::SampleConsensusModelNormalPlane<PointT, PointNT>;

// Proxy to expose protected RVV APIs, same style as test file
template <typename PointT_, typename PointNT_>
class SampleConsensusModelNormalPlaneBench
  : public pcl::SampleConsensusModelNormalPlane<PointT_, PointNT_>
{
  using Base = pcl::SampleConsensusModelNormalPlane<PointT_, PointNT_>;

public:
  using Base::Base;

  using Base::selectWithinDistanceRVV;
  using Base::selectWithinDistanceStandard;
  using Base::countWithinDistanceRVV;
  using Base::countWithinDistanceStandard;
  using Base::getDistancesToModelRVV;
  using Base::getDistancesToModelStandard;
  using Base::error_sqr_dists_;
};

class Benchmarker {
public:
  explicit Benchmarker(const std::string& name) : name_(name) {}

  struct Result {
    double avg_ms_per_iter{};
    double total_ms{};
  };

  Result run(const std::function<void()>& func, int iterations = 20, int warmup = 3) {
    for (int i = 0; i < warmup; ++i) func();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) func();
    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / iterations;
    return {.avg_ms_per_iter = avg_ms, .total_ms = total_ms};
  }

private:
  std::string name_;
};

static void
print_bar(char ch, int width = 85)
{
  for (int i = 0; i < width; ++i)
    std::cout << ch;
  std::cout << "\n";
}

int
main (int argc, char** argv)
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " sac_plane_test.pcd [iters]\n";
    return -1;
  }

  const char* pcd_path = argv[1];
  int iters = 50;
  if (argc >= 3)
    iters = std::max(1, std::atoi(argv[2]));

  pcl::PCLPointCloud2 cloud_blob;
  if (pcl::io::loadPCDFile (pcd_path, cloud_blob) < 0)
  {
    std::cerr << "Failed to read test file: " << pcd_path << "\n";
    return -1;
  }

  pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointNT>::Ptr normals (new pcl::PointCloud<PointNT>);

  pcl::fromPCLPointCloud2 (cloud_blob, *cloud);
  pcl::fromPCLPointCloud2 (cloud_blob, *normals);

  pcl::Indices indices (cloud->size ());
  for (std::size_t i = 0; i < indices.size (); ++i)
    indices[i] = static_cast<int>(i);

  Eigen::VectorXf coeffs (4);
  coeffs[0] = -0.8964f;
  coeffs[1] = -0.5868f;
  coeffs[2] = -1.208f;
  coeffs[3] = 1.0f;

  SampleConsensusModelNormalPlaneBench<PointT, PointNT> model (cloud, true);
  model.setInputNormals (normals);
  model.setIndices (indices);
  model.setNormalDistanceWeight (0.1);

  const double threshold = 0.05;

  // 预分配误差与 inliers 缓冲区，避免 RVV 写越界（与测试用例保持一致）
  model.error_sqr_dists_.assign(indices.size(), 0.0);
  pcl::Indices inliers_buffer(indices.size());
  std::vector<double> distances_buffer(indices.size());

  // 为了做“标量 vs RVV”对比，这里直接调用 *_Standard 和 *_RVV，避免 selectWithinDistance/countWithinDistance 的自动 dispatch。
  // 注意：selectWithinDistanceStandard 需要 current_count（写入偏移），这里从 0 开始。

#ifdef __RVV10__
  const bool has_rvv = true;
#else
  const bool has_rvv = false;
#endif

  // --- 1) selectWithinDistance ---
  Benchmarker bench_sel_std("Std selectWithinDistanceStandard");
  const auto sel_std = bench_sel_std.run([&](){
    model.error_sqr_dists_.assign(indices.size(), 0.0);
    inliers_buffer.assign(indices.size(), 0);
    const std::size_t nr = model.selectWithinDistanceStandard(coeffs, threshold, inliers_buffer, 0, 0);
    pcl::utils::ignore(nr);
  }, iters);

  Benchmarker::Result sel_rvv{};
  if (has_rvv) {
#ifdef __RVV10__
    Benchmarker bench_sel_rvv("RVV selectWithinDistanceRVV");
    sel_rvv = bench_sel_rvv.run([&](){
      model.error_sqr_dists_.assign(indices.size(), 0.0);
      inliers_buffer.assign(indices.size(), 0);
      const std::size_t nr = model.selectWithinDistanceRVV (coeffs, threshold, inliers_buffer);
      pcl::utils::ignore(nr);
    }, iters);
#endif
  } else {
    std::cout << "[WARN] __RVV10__ not defined, RVV selectWithinDistance not benchmarked.\n";
  }

  // --- 2) countWithinDistance ---
  Benchmarker bench_cnt_std("Std countWithinDistanceStandard");
  const auto cnt_std = bench_cnt_std.run([&](){
    const std::size_t nr = model.countWithinDistanceStandard(coeffs, threshold, 0);
    pcl::utils::ignore(nr);
  }, iters);

  Benchmarker::Result cnt_rvv{};
  if (has_rvv) {
#ifdef __RVV10__
    Benchmarker bench_cnt_rvv("RVV countWithinDistanceRVV");
    cnt_rvv = bench_cnt_rvv.run([&](){
      const std::size_t nr = model.countWithinDistanceRVV (coeffs, threshold, 0);
      pcl::utils::ignore(nr);
    }, iters);
#endif
  } else {
    std::cout << "[WARN] __RVV10__ not defined, RVV countWithinDistance not benchmarked.\n";
  }

  // --- 3) getDistancesToModel ---
  Benchmarker bench_dist_std("Std getDistancesToModelStandard");
  const auto dist_std = bench_dist_std.run([&](){
    model.getDistancesToModelStandard(coeffs, distances_buffer, 0);
  }, iters);

  Benchmarker::Result dist_rvv{};
  if (has_rvv) {
#ifdef __RVV10__
    Benchmarker bench_dist_rvv("RVV getDistancesToModelRVV");
    dist_rvv = bench_dist_rvv.run([&](){
      model.getDistancesToModelRVV(coeffs, distances_buffer);
    }, iters);
#endif
  } else {
    std::cout << "[WARN] __RVV10__ not defined, RVV getDistancesToModel not benchmarked.\n";
  }

  // ==========================================================
  // 参数化表格排版设置
  // ==========================================================
  // 1. 定义各列的字符宽度
  constexpr int w_item = 24;
  constexpr int w_impl = 6;
  constexpr int w_avg  = 12;
  constexpr int w_tot  = 12;
  constexpr int w_spd  = 12;

  // 2. 自动计算总宽度 (4个 " | " 分隔符，每个占3个字符)
  constexpr int total_width = w_item + w_impl + w_avg + w_tot + w_spd + (4 * 3) + 1; // +1 for the last vertical bar

  // pretty summary table
  print_bar('=', total_width);
  std::cout << " PCL SampleConsensus: NormalPlane Benchmark (RVV 1.0)\n";
  print_bar('=', total_width);
  std::cout << "[Benchmark Context]\n";
  std::cout << "  Device     : RISC-V RVV target (rv64gcv)\n";
  std::cout << "  VLEN       : 256-bit (zvl256b)\n";
  std::cout << "  Dataset    : " << pcd_path << " (" << cloud->size () << " points)\n";
  std::cout << "  Iterations : " << iters << "\n\n";

  print_bar('-', total_width);

  // 3. 打印表头
  std::cout << std::left  << std::setw(w_item) << "Benchmark Item" << " | "
            << std::left  << std::setw(w_impl) << "Impl" << " | "
            << std::right << std::setw(w_avg)  << "Avg Time"       << " | "
            << std::right << std::setw(w_tot)  << "Total Time"     << " | "
            << std::right << std::setw(w_spd)  << "Speedup"        << "\n";

  // 4. 自动生成对应的十字分隔线
  const std::string separator =
      std::string(w_item + 1, '-') + "|" +
      std::string(w_impl + 2, '-') + "|" +
      std::string(w_avg  + 2, '-') + "|" +
      std::string(w_tot  + 2, '-') + "|" +
      std::string(w_spd  + 2, '-');
  std::cout << separator << "\n";

  // 5. 打印行
  auto print_row = [&](const std::string& item,
                       const char* impl,
                       const Benchmarker::Result& r,
                       double speedup) {
    std::ostringstream avg_os, tot_os, spd_os;
    avg_os << std::fixed << std::setprecision(4) << r.avg_ms_per_iter << " ms";
    tot_os << std::fixed << std::setprecision(4) << r.total_ms << " ms";
    spd_os << "[ " << std::setw(6) << std::right << std::fixed << std::setprecision(2) << speedup << "x ]";

    std::cout << std::left  << std::setw(w_item) << item << " | "
              << std::left  << std::setw(w_impl) << impl << " | "
              << std::right << std::setw(w_avg)  << avg_os.str() << " | "
              << std::right << std::setw(w_tot)  << tot_os.str() << " | "
              << std::right << std::setw(w_spd)  << spd_os.str() << "\n";
  };

  auto print_group = [&](const std::string& item,
                         const Benchmarker::Result& std_r,
                         const Benchmarker::Result& rvv_r) {
    const double speedup_rvv = (rvv_r.total_ms > 0.0) ? (std_r.total_ms / rvv_r.total_ms) : 0.0;
    print_row(item, "Std", std_r, 1.00);
    print_row("",   "RVV", rvv_r, speedup_rvv);
    std::cout << separator << "\n";
  };

  if (has_rvv) {
    print_group("selectWithinDistance", sel_std, sel_rvv);
    print_group("countWithinDistance",  cnt_std, cnt_rvv);
    print_group("getDistancesToModel",  dist_std, dist_rvv);
  }

  print_bar('=', total_width);

  return 0;
}

