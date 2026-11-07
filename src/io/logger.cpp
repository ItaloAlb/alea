// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "io/logger.h"

#include <chrono>
#include <ctime>
#include <filesystem>

namespace fs = std::filesystem;

namespace alea::io {

// ── static members ──────────────────────────────────────────────────────────

std::ofstream  Logger::file_;
std::mutex     Logger::mutex_;
fs::path       Logger::log_path_;
bool           Logger::initialized_ = false;
bool           Logger::paused_      = false;

// ── helpers ─────────────────────────────────────────────────────────────────

std::string Logger::timestamp_() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[16];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", &tm);
    return buf;
}

namespace {

// "YYYY-MM-DD_HH-MM-SS-mmm".  The millisecond suffix keeps two runs started
// within the same second from colliding on the directory name.
std::string dir_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                   now.time_since_epoch()) % 1000;
    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", &tm);
    return std::string(buf) + '-' + std::to_string(ms.count());
}

}  // namespace

fs::path make_run_dir(const std::string& results_dir,
                      const std::string& project_name) {
    const fs::path base = fs::path(results_dir) / project_name / dir_timestamp();

    // Two runs started within the same millisecond would otherwise collide and
    // clobber each other's log/results; disambiguate with a numeric suffix.
    fs::path run_dir = base;
    for (int n = 1; fs::exists(run_dir); ++n)
        run_dir = base.string() + '-' + std::to_string(n);

    fs::create_directories(run_dir);
    return run_dir;
}

const char* Logger::level_str_(LogLevel level) {
    switch (level) {
    case LogLevel::info:  return "info";
    case LogLevel::warn:  return "warn";
    case LogLevel::error: return "error";
    }
    return "???";
}

// ── public interface ────────────────────────────────────────────────────────

void Logger::init(const std::string& results_dir,
                  const std::string& project_name) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Close any previously open file.
    if (file_.is_open()) file_.close();

    // results/<project>/<timestamp>/alea.log
    log_path_ = make_run_dir(results_dir, project_name) / "alea.log";

    file_.open(log_path_);
    if (!file_.is_open()) {
        initialized_ = false;
        return;
    }

    file_ << "Alea: A quantum Monte Carlo program\n";
    file_.flush();

    initialized_ = true;
}

void Logger::info(const std::string& msg) {
    log(LogLevel::info, msg);
}

void Logger::warn(const std::string& msg) {
    log(LogLevel::warn, msg);
}

void Logger::error(const std::string& msg) {
    log(LogLevel::error, msg);
}

void Logger::milestone(const std::string& msg) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || !file_.is_open()) return;
    if (paused_) return;

    file_ << '[' << timestamp_() << "] info: " << msg << '\n';
    file_.flush();
}

void Logger::log(LogLevel level, const std::string& msg) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || !file_.is_open()) return;

    // When paused, suppress info-level chatter (e.g. from a VMC that is
    // being driven by LinearOptimize).  warn and error still pass through
    // so crash diagnostics are never lost.
    if (paused_ && level == LogLevel::info) return;

    file_ << '[' << timestamp_() << "] "
          << level_str_(level) << ": "
          << msg << '\n';

    // info messages are buffered; warn/error flush immediately
    // so they survive a crash.
    if (level == LogLevel::warn || level == LogLevel::error)
        file_.flush();
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) file_.flush();
}

void Logger::pause() {
    std::lock_guard<std::mutex> lock(mutex_);
    paused_ = true;
}

void Logger::resume() {
    std::lock_guard<std::mutex> lock(mutex_);
    paused_ = false;
}

void Logger::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
    initialized_ = false;
}

const fs::path& Logger::log_path() noexcept {
    return log_path_;
}

fs::path Logger::run_dir() {
    return log_path_.parent_path();
}

}  // namespace alea::io
