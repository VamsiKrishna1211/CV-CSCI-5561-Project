/*
Copyright 2022 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---- Original license for this file: ----
 * Copyright (C) 2012-2013 Simon Lynen, ASL, ETH Zurich, Switzerland
 * You can contact the author at <slynen at ethz dot ch>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Adapted from Paul Furgale Schweizer Messer sm_timing */

#pragma once

#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "nvblox/utils/nvtx_ranges.h"

namespace nvblox {
namespace timing {

// Forward declaration
class TimerChrono;
class TimerNvtx;
class DummyTimer;

// The project wide typedef defining the type of timer to be used
typedef TimerNvtx Timer;

template <typename T, typename Total, int N>
class Accumulator {
 public:
  Accumulator()
      : window_samples_(0),
        totalsamples_(0),
        window_sum_(0),
        sum_(0),
        min_(std::numeric_limits<T>::max()),
        max_(std::numeric_limits<T>::min()) {}

  void Add(T sample) {
    if (window_samples_ < N) {
      samples_[window_samples_++] = sample;
      window_sum_ += sample;
    } else {
      T& oldest = samples_[window_samples_++ % N];
      window_sum_ += sample - oldest;
      oldest = sample;
    }
    sum_ += sample;
    ++totalsamples_;
    if (sample > max_) {
      max_ = sample;
    }
    if (sample < min_) {
      min_ = sample;
    }
  }

  int TotalSamples() const { return totalsamples_; }

  int WindowSamples() const { return std::min(window_samples_, N); }

  double Sum() const { return sum_; }

  double Mean() const { return static_cast<double>(sum_) / totalsamples_; }

  double RollingMean() const {
    return static_cast<double>(window_sum_) / std::min(window_samples_, N);
  }

  double Max() const { return max_; }

  double Min() const { return min_; }

  double LazyVariance() const {
    if (window_samples_ == 0) {
      return 0.0;
    }
    double var = 0;
    double mean = RollingMean();
    for (int i = 0; i < std::min(window_samples_, N); ++i) {
      var += (samples_[i] - mean) * (samples_[i] - mean);
    }
    var /= std::min(window_samples_, N);
    return var;
  }

 private:
  int window_samples_;
  int totalsamples_;
  Total window_sum_;
  Total sum_;
  T min_;
  T max_;
  T samples_[N];
};

struct TimerMapValue {
  TimerMapValue() {}

  /// Create an accumulator with specified window size.
  Accumulator<double, double, 50> acc_;
};

/**
 * A class that has the timer interface but does nothing. Swapping this in in
 * place of the Timer class (say with a typedef) should allow one to disable
 * timing. Because all of the functions are inline, they should just disappear.
 */
class DummyTimer {
 public:
  explicit DummyTimer(size_t /*handle*/, bool /*constructStopped*/ = false) {}
  explicit DummyTimer(std::string const& /*tag*/,
                      bool /*constructStopped*/ = false) {}
  ~DummyTimer() {}

  void Start() {}
  void Stop() {}
  bool IsTiming() { return false; }
};

class TimerChrono {
 public:
  explicit TimerChrono(size_t handle, bool constructStopped = false);
  explicit TimerChrono(std::string const& tag, bool constructStopped = false);
  ~TimerChrono();

  void Start();
  void Stop();
  bool IsTiming() const;

 private:
  std::chrono::time_point<std::chrono::system_clock> time_;

  bool timing_;
  size_t handle_;
};

class TimerNvtx {
 public:
  explicit TimerNvtx(std::string const& tag, const Color& color,
                     bool constructStopped = false);
  explicit TimerNvtx(std::string const& tag, bool constructStopped = false);

  void Start();
  void Stop();
  bool IsTiming() const;

 private:
  TimerChrono timer_;
  NvtxRange nvtx_range_;
};

class Timing {
 public:
  typedef std::map<std::string, size_t> map_t;
  friend class TimerChrono;
  // Definition of static functions to query the timers.
  static size_t GetHandle(std::string const& tag);
  static std::string GetTag(size_t handle);
  static double GetTotalSeconds(size_t handle);
  static double GetTotalSeconds(std::string const& tag);
  static double GetMeanSeconds(size_t handle);
  static double GetMeanSeconds(std::string const& tag);
  static size_t GetNumSamples(size_t handle);
  static size_t GetNumSamples(std::string const& tag);
  static double GetVarianceSeconds(size_t handle);
  static double GetVarianceSeconds(std::string const& tag);
  static double GetMinSeconds(size_t handle);
  static double GetMinSeconds(std::string const& tag);
  static double GetMaxSeconds(size_t handle);
  static double GetMaxSeconds(std::string const& tag);
  static double GetHz(size_t handle);
  static double GetHz(std::string const& tag);
  static void Print(std::ostream& out);
  static std::string Print();
  static std::string SecondsToTimeString(double seconds);
  static void Reset();
  static const map_t& GetTimers() { return Instance().tagMap_; }

 private:
  void AddTime(size_t handle, double seconds);

  static Timing& Instance();

  Timing();
  ~Timing();

  typedef std::vector<TimerMapValue> list_t;

  list_t timers_;
  map_t tagMap_;
  size_t maxTagLength_;
  std::mutex mutex_;
};

}  // namespace timing
}  // namespace nvblox
