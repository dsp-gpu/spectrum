#pragma once

/**
 * @file test_fft_matrix_rocm.hpp
 * @brief FFT Matrix Benchmark — таблица времени (лучи × nFFT) на ROCm/hipFFT
 *
 * Запускает матрицу замеров: 20 значений beam_count × 13 значений nFFT.
 * Контрольная точка 320×1024 в начале и конце для оценки drift.
 *
 * Выход: Results/Profiler/FFT_Matrix/fft_matrix_YYYY-MM-DD_HH-MM-SS.md
 *   - Таблица 1: Только FFT (hipfftExecC2C)
 *   - Таблица 2: GPU processing (Pad + FFT)
 *   - Таблица 3: Полный цикл (Upload + Pad + FFT + Download)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-02
 * @see MemoryBank/tasks/TASK_FFT_Matrix_Benchmark.md
 */

#if ENABLE_ROCM

#include <spectrum/fft_processor_rocm.hpp>
#include <core/backends/rocm/rocm_backend.hpp>

#include <complex>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <string>
#include <functional>

namespace test_fft_matrix_rocm {

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

struct MatrixBenchConfig {
  int fft_exp_min  = 4;       // 2^4 = 16
  int fft_exp_max  = 17;      // 2^16 = 65536 (exclusive upper bound)
  int beam_min     = 20;
  int beam_max     = 400;
  int beam_step    = 20;
  int n_warmup     = 5;
  int n_runs       = 10;
  int ctrl_beams   = 320;
  int ctrl_npoint  = 1024;
  float sample_rate = 1e6f;
  std::string output_dir = "../Results/Profiler/FFT_Matrix";
};

// ═══════════════════════════════════════════════════════════════════════════
// Timing data per matrix cell
// ═══════════════════════════════════════════════════════════════════════════

struct CellTiming {
  double upload_ms   = 0.0;
  double pad_ms      = 0.0;
  double fft_ms      = 0.0;
  double download_ms = 0.0;

  double FftOnly()       const { return fft_ms; }
  double GpuProcessing() const { return pad_ms + fft_ms; }
  double FullCycle()     const { return upload_ms + pad_ms + fft_ms + download_ms; }
};

using CellKey    = std::pair<int, int>;  // (beams, nfft)
using MatrixData = std::map<CellKey, CellTiming>;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> GenerateData(size_t total_points) {
  std::vector<std::complex<float>> data(total_points);
  for (size_t i = 0; i < total_points; ++i) {
    float v = static_cast<float>(i % 1000) * 0.001f;
    data[i] = {v, 1.0f - v};
  }
  return data;
}

inline CellTiming ExtractTiming(const fft_processor::ROCmProfEvents& events) {
  CellTiming t;
  for (const auto& [name, data] : events) {
    double ms = static_cast<double>(data.end_ns - data.start_ns) / 1e6;
    std::string s(name);
    if      (s == "Upload")   t.upload_ms   += ms;
    else if (s == "Pad")      t.pad_ms      += ms;
    else if (s == "FFT")      t.fft_ms      += ms;
    else if (s == "Download") t.download_ms += ms;
  }
  return t;
}

inline CellTiming AverageTiming(const std::vector<CellTiming>& timings) {
  CellTiming avg;
  if (timings.empty()) return avg;
  for (const auto& t : timings) {
    avg.upload_ms   += t.upload_ms;
    avg.pad_ms      += t.pad_ms;
    avg.fft_ms      += t.fft_ms;
    avg.download_ms += t.download_ms;
  }
  double n = static_cast<double>(timings.size());
  avg.upload_ms   /= n;
  avg.pad_ms      /= n;
  avg.fft_ms      /= n;
  avg.download_ms /= n;
  return avg;
}

inline std::string GetTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto tt  = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&tt, &tm);
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d_%02d-%02d-%02d",
      tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
      tm.tm_hour, tm.tm_min, tm.tm_sec);
  return buf;
}

// ═══════════════════════════════════════════════════════════════════════════
// Measure a single (beams, nfft) combination
// ═══════════════════════════════════════════════════════════════════════════

inline CellTiming MeasureCell(
    drv_gpu_lib::IBackend* backend,
    int beams, int nfft, float sample_rate,
    int n_warmup, int n_runs)
{
  fft_processor::FFTProcessorParams params;
  params.beam_count  = static_cast<uint32_t>(beams);
  params.n_point     = static_cast<uint32_t>(nfft);
  params.sample_rate = sample_rate;
  params.output_mode = fft_processor::FFTOutputMode::COMPLEX;

  size_t total_points = static_cast<size_t>(beams) * nfft;
  auto data = GenerateData(total_points);

  // Fresh processor per cell — AllocateBuffers doesn't track nFFT changes
  fft_processor::FFTProcessorROCm proc(backend);

  for (int i = 0; i < n_warmup; ++i) {
    proc.ProcessComplex(data, params);
  }

  std::vector<CellTiming> timings;
  timings.reserve(n_runs);
  for (int i = 0; i < n_runs; ++i) {
    fft_processor::ROCmProfEvents events;
    proc.ProcessComplex(data, params, &events);
    timings.push_back(ExtractTiming(events));
  }

  return AverageTiming(timings);
}

// ═══════════════════════════════════════════════════════════════════════════
// Markdown export
// ═══════════════════════════════════════════════════════════════════════════

inline void WriteTable(
    std::ostream& out,
    const std::string& title,
    const MatrixData& matrix,
    const std::vector<int>& beams_list,
    const std::vector<int>& nfft_list,
    std::function<double(const CellTiming&)> extractor)
{
  out << "## " << title << "\n\n";
  out << "*Время в мс, среднее по N замерам*\n\n";

  out << "| Лучи \\ nFFT |";
  for (int nfft : nfft_list)
    out << " " << nfft << " |";
  out << "\n";

  out << "|---:|";
  for (size_t i = 0; i < nfft_list.size(); ++i)
    out << "---:|";
  out << "\n";

  for (int beams : beams_list) {
    out << "| " << beams << " |";
    for (int nfft : nfft_list) {
      auto it = matrix.find({beams, nfft});
      if (it != matrix.end()) {
        out << " " << std::fixed << std::setprecision(3)
            << extractor(it->second) << " |";
      } else {
        out << " - |";
      }
    }
    out << "\n";
  }
  out << "\n";
}

inline void ExportMarkdown(
    const std::string& filepath,
    const std::string& gpu_name,
    size_t gpu_mem_mb,
    const MatrixBenchConfig& cfg,
    const MatrixData& matrix,
    const std::vector<int>& beams_list,
    const std::vector<int>& nfft_list,
    const CellTiming& ctrl_start,
    const CellTiming& ctrl_end)
{
  std::ofstream f(filepath);
  if (!f) {
    std::cerr << "  [ERROR] Cannot open: " << filepath << "\n";
    return;
  }

  f << "# FFT Matrix Benchmark (ROCm/hipFFT)\n\n";
  f << "| Параметр | Значение |\n";
  f << "|----------|----------|\n";
  f << "| **GPU** | " << gpu_name << " |\n";
  f << "| **Память** | " << gpu_mem_mb << " MB |\n";
  f << "| **Дата** | " << GetTimestamp() << " |\n";
  f << "| **Mode** | COMPLEX (чистый FFT) |\n";
  f << "| **Warmup** | " << cfg.n_warmup << " |\n";
  f << "| **Runs** | " << cfg.n_runs << " |\n";
  f << "| **Beams** | " << cfg.beam_min << "..." << cfg.beam_max
    << " (шаг " << cfg.beam_step << ") |\n";
  f << "| **nFFT** | 2^" << cfg.fft_exp_min << "...2^"
    << (cfg.fft_exp_max - 1) << " |\n\n";

  f << "---\n\n";

  // Control point comparison
  f << "## Контрольная точка: " << cfg.ctrl_beams << "×" << cfg.ctrl_npoint << "\n\n";
  f << "| Позиция | Upload | Pad | FFT | Download | Полный цикл |\n";
  f << "|---------|-------:|----:|----:|---------:|------------:|\n";

  auto writeCtrl = [&](const char* label, const CellTiming& c) {
    f << "| " << label << " | "
      << std::fixed << std::setprecision(3)
      << c.upload_ms << " | " << c.pad_ms << " | "
      << c.fft_ms << " | " << c.download_ms << " | "
      << c.FullCycle() << " |\n";
  };
  writeCtrl("Начало", ctrl_start);
  writeCtrl("Конец", ctrl_end);

  double drift_pct = 0.0;
  if (ctrl_start.FullCycle() > 1e-9) {
    drift_pct = (ctrl_end.FullCycle() - ctrl_start.FullCycle())
              / ctrl_start.FullCycle() * 100.0;
  }
  f << "| **Drift** | | | | | "
    << (drift_pct >= 0 ? "+" : "") << std::fixed << std::setprecision(1)
    << drift_pct << "% |\n\n";
  f << "---\n\n";

  // 3 tables
  WriteTable(f, "Таблица 1: Только FFT (hipfftExecC2C)",
      matrix, beams_list, nfft_list,
      [](const CellTiming& c) { return c.FftOnly(); });

  WriteTable(f, "Таблица 2: GPU Processing (Pad + FFT)",
      matrix, beams_list, nfft_list,
      [](const CellTiming& c) { return c.GpuProcessing(); });

  WriteTable(f, "Таблица 3: Полный цикл (Upload + Pad + FFT + Download)",
      matrix, beams_list, nfft_list,
      [](const CellTiming& c) { return c.FullCycle(); });

  f << "---\n\n";
  f << "## Легенда\n\n";
  f << "| Стадия | Описание |\n";
  f << "|--------|----------|\n";
  f << "| **Upload** | hipMemcpyHtoD (CPU → GPU) |\n";
  f << "| **Pad** | Zero-padding kernel (input → FFT buffer) |\n";
  f << "| **FFT** | hipfftExecC2C (batch FFT) |\n";
  f << "| **Download** | hipMemcpyDtoH (GPU → CPU) |\n";
  f << "| **GPU Processing** | Pad + FFT (данные уже на GPU) |\n";
  f << "| **Полный цикл** | Upload + Pad + FFT + Download |\n\n";
  f << "*Сгенерировано автоматически: test_fft_matrix_rocm.hpp*\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// Plain-text export (.txt)
// ═══════════════════════════════════════════════════════════════════════════

inline void WriteTxtTable(
    std::ostream& out,
    const std::string& title,
    const MatrixData& matrix,
    const std::vector<int>& beams_list,
    const std::vector<int>& nfft_list,
    std::function<double(const CellTiming&)> extractor)
{
  std::string sep(80, '=');
  out << sep << "\n  " << title << "\n  Time in ms, average over N runs\n"
      << sep << "\n\n";

  out << " Beams |";
  for (int nfft : nfft_list)
    out << std::setw(7) << nfft << " |";
  out << "\n";

  out << "-------+";
  for (size_t i = 0; i < nfft_list.size(); ++i)
    out << "--------+";
  out << "\n";

  for (int beams : beams_list) {
    out << std::setw(5) << beams << " |";
    for (int nfft : nfft_list) {
      auto it = matrix.find({beams, nfft});
      if (it != matrix.end()) {
        out << std::setw(7) << std::fixed << std::setprecision(3)
            << extractor(it->second) << " |";
      } else {
        out << "      - |";
      }
    }
    out << "\n";
  }
  out << "\n";
}

inline void ExportTxt(
    const std::string& filepath,
    const std::string& gpu_name,
    size_t gpu_mem_mb,
    const MatrixBenchConfig& cfg,
    const MatrixData& matrix,
    const std::vector<int>& beams_list,
    const std::vector<int>& nfft_list,
    const CellTiming& ctrl_start,
    const CellTiming& ctrl_end)
{
  std::ofstream f(filepath);
  if (!f) {
    std::cerr << "  [ERROR] Cannot open: " << filepath << "\n";
    return;
  }

  std::string sep(80, '=');
  f << sep << "\n  FFT Matrix Benchmark (ROCm/hipFFT)\n" << sep << "\n\n";
  f << "  GPU:      " << gpu_name << "\n";
  f << "  Memory:   " << gpu_mem_mb << " MB\n";
  f << "  Date:     " << GetTimestamp() << "\n";
  f << "  Mode:     COMPLEX (pure FFT)\n";
  f << "  Warmup:   " << cfg.n_warmup << "\n";
  f << "  Runs:     " << cfg.n_runs << "\n";
  f << "  Beams:    " << cfg.beam_min << "..." << cfg.beam_max
    << " (step " << cfg.beam_step << ")\n";
  f << "  nFFT:     2^" << cfg.fft_exp_min << "...2^"
    << (cfg.fft_exp_max - 1) << "\n\n";

  f << sep << "\n  Control Point: "
    << cfg.ctrl_beams << "x" << cfg.ctrl_npoint << "\n" << sep << "\n\n";

  f << "  Position  |  Upload  |   Pad   |   FFT   | Download | Full Cycle\n";
  f << "  ----------+----------+---------+---------+----------+-----------\n";

  auto writeCtrl = [&](const char* label, const CellTiming& c) {
    f << "  " << std::setw(9) << std::left << label << std::right
      << " |" << std::setw(8) << std::fixed << std::setprecision(3) << c.upload_ms
      << "  |" << std::setw(7) << c.pad_ms
      << "  |" << std::setw(7) << c.fft_ms
      << "  |" << std::setw(8) << c.download_ms
      << "  |" << std::setw(9) << c.FullCycle() << "\n";
  };
  writeCtrl("Start", ctrl_start);
  writeCtrl("End", ctrl_end);

  double drift_pct = 0.0;
  if (ctrl_start.FullCycle() > 1e-9) {
    drift_pct = (ctrl_end.FullCycle() - ctrl_start.FullCycle())
              / ctrl_start.FullCycle() * 100.0;
  }
  f << "  Drift     |          |         |         |          |  "
    << (drift_pct >= 0 ? "+" : "") << std::fixed << std::setprecision(1)
    << drift_pct << "%\n\n";

  WriteTxtTable(f, "Table 1: FFT Only (hipfftExecC2C)",
      matrix, beams_list, nfft_list,
      [](const CellTiming& c) { return c.FftOnly(); });

  WriteTxtTable(f, "Table 2: GPU Processing (Pad + FFT)",
      matrix, beams_list, nfft_list,
      [](const CellTiming& c) { return c.GpuProcessing(); });

  WriteTxtTable(f, "Table 3: Full Cycle (Upload + Pad + FFT + Download)",
      matrix, beams_list, nfft_list,
      [](const CellTiming& c) { return c.FullCycle(); });

  f << sep << "\n  Legend\n" << sep << "\n\n";
  f << "  Upload         - hipMemcpyHtoD (CPU -> GPU)\n";
  f << "  Pad            - Zero-padding kernel (input -> FFT buffer)\n";
  f << "  FFT            - hipfftExecC2C (batch FFT)\n";
  f << "  Download       - hipMemcpyDtoH (GPU -> CPU)\n";
  f << "  GPU Processing - Pad + FFT (data already on GPU)\n";
  f << "  Full Cycle     - Upload + Pad + FFT + Download\n\n";
  f << "  Generated by: test_fft_matrix_rocm.hpp\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry point
// ═══════════════════════════════════════════════════════════════════════════

inline int run() {
  MatrixBenchConfig cfg;

  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  FFT Matrix Benchmark (ROCm/hipFFT)\n";
  std::cout << "  COMPLEX mode | warmup=" << cfg.n_warmup
            << " | runs=" << cfg.n_runs << "\n";
  std::cout << "============================================================\n";

  int device_count = drv_gpu_lib::ROCmCore::GetAvailableDeviceCount();
  if (device_count == 0) {
    std::cout << "  [SKIP] No ROCm devices\n";
    return 0;
  }

  try {
    drv_gpu_lib::ROCmBackend backend;
    backend.Initialize(0);

    auto device_info = backend.GetDeviceInfo();
    size_t gpu_mem_mb = backend.GetGlobalMemorySize() / (1024 * 1024);
    std::cout << "  GPU: " << device_info.name << "\n";
    std::cout << "  Memory: " << gpu_mem_mb << " MB\n\n";

    // Build parameter lists
    std::vector<int> beams_list;
    for (int b = cfg.beam_min; b <= cfg.beam_max; b += cfg.beam_step)
      beams_list.push_back(b);

    std::vector<int> nfft_list;
    for (int e = cfg.fft_exp_min; e < cfg.fft_exp_max; ++e)
      nfft_list.push_back(1 << e);

    int total_cells = static_cast<int>(beams_list.size() * nfft_list.size());
    std::cout << "  Matrix: " << beams_list.size() << " beams x "
              << nfft_list.size() << " nFFT = " << total_cells << " cells\n";
    std::cout << "  Control: " << cfg.ctrl_beams << "x" << cfg.ctrl_npoint
              << " (start + end)\n\n";

    // ── Control point START ──────────────────────────────────────────
    std::cout << "  [CTRL-START] " << cfg.ctrl_beams << "x"
              << cfg.ctrl_npoint << " ...";
    std::cout.flush();

    CellTiming ctrl_start = MeasureCell(
        &backend, cfg.ctrl_beams, cfg.ctrl_npoint,
        cfg.sample_rate, cfg.n_warmup, cfg.n_runs);

    std::cout << " FFT=" << std::fixed << std::setprecision(3) << ctrl_start.fft_ms
              << "  Full=" << ctrl_start.FullCycle() << " ms\n";

    // ── Matrix ───────────────────────────────────────────────────────
    MatrixData matrix;
    int cell_idx = 0;

    for (int beams : beams_list) {
      for (int nfft : nfft_list) {
        ++cell_idx;

        CellTiming avg = MeasureCell(
            &backend, beams, nfft,
            cfg.sample_rate, cfg.n_warmup, cfg.n_runs);

        matrix[{beams, nfft}] = avg;

        bool show = (cell_idx <= 3)
                 || (cell_idx % 20 == 0)
                 || (cell_idx >= total_cells - 2);
        if (show) {
          std::cout << "  [" << std::setw(3) << cell_idx << "/"
                    << total_cells << "] "
                    << std::setw(3) << beams << "x" << std::setw(5) << nfft
                    << "  FFT=" << std::fixed << std::setprecision(3)
                    << avg.fft_ms
                    << "  Full=" << avg.FullCycle() << " ms\n";
        }
      }
    }

    // ── Control point END ────────────────────────────────────────────
    std::cout << "  [CTRL-END]   " << cfg.ctrl_beams << "x"
              << cfg.ctrl_npoint << " ...";
    std::cout.flush();

    CellTiming ctrl_end = MeasureCell(
        &backend, cfg.ctrl_beams, cfg.ctrl_npoint,
        cfg.sample_rate, cfg.n_warmup, cfg.n_runs);

    std::cout << " FFT=" << std::fixed << std::setprecision(3) << ctrl_end.fft_ms
              << "  Full=" << ctrl_end.FullCycle() << " ms\n";

    // Drift
    double drift_pct = 0.0;
    if (ctrl_start.FullCycle() > 1e-9) {
      drift_pct = (ctrl_end.FullCycle() - ctrl_start.FullCycle())
                / ctrl_start.FullCycle() * 100.0;
    }
    std::cout << "  [DRIFT] "
              << (drift_pct >= 0 ? "+" : "") << std::fixed
              << std::setprecision(1) << drift_pct << "%\n\n";

    // ── Export ────────────────────────────────────────────────────────
    std::filesystem::create_directories(cfg.output_dir);
    std::string ts = GetTimestamp();
    std::string md_path = cfg.output_dir + "/fft_matrix_" + ts + ".md";

    ExportMarkdown(md_path, device_info.name, gpu_mem_mb, cfg,
        matrix, beams_list, nfft_list, ctrl_start, ctrl_end);

    std::string txt_path = cfg.output_dir + "/fft_matrix_" + ts + ".txt";
    ExportTxt(txt_path, device_info.name, gpu_mem_mb, cfg,
        matrix, beams_list, nfft_list, ctrl_start, ctrl_end);

    std::cout << "  Exported: " << md_path << "\n";
    std::cout << "  Exported: " << txt_path << "\n";
    std::cout << "  [OK] FFT Matrix Benchmark complete\n";

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "  FATAL: " << e.what() << "\n";
    return 1;
  }
}

}  // namespace test_fft_matrix_rocm

#endif  // ENABLE_ROCM
