// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

namespace alea::io {

enum class LogLevel { info, warn, error };

//
// Thread-safe file logger.
//
// Writes timestamped messages to a log file under:
//     <results_dir>/<project_name>/<YYYY-MM-DD_HH-MM-SS>/alea.log
//
// Usage:
//     Logger::init("results", "exciton");
//     Logger::info("accumulation: 50% complete");
//     Logger::warn("wall time limit approaching");
//     Logger::error("failed to open file");
//     Logger::shutdown();
//
class Logger {
public:
    //
    // Open the log file.
    // Creates the directory tree and writes the program header.
    //
    static void init(const std::string& results_dir,
                     const std::string& project_name);

    //
    // Severity convenience functions.
    //
    static void info(const std::string& msg);
    static void warn(const std::string& msg);
    static void error(const std::string& msg);

    //
    // Generic write with explicit level.
    //
    static void log(LogLevel level, const std::string& msg);

    //
    // Flush buffered data to disk.
    //
    static void flush();

    //
    // Like info(), but also flushes immediately so the log file is always
    // up to date for `tail -f` during a long simulation.
    //
    static void milestone(const std::string& msg);

    //
    // Suppress info-level messages.  warn, error and milestone still pass
    // through.  Used to silence sub-drivers (e.g. the VMC inside
    // LinearOptimize).
    //
    static void pause();
    static void resume();

    //
    // Flush and close the log file. Safe to call multiple times.
    // After shutdown, logging calls are no-ops until the next init().
    //
    static void shutdown();

    //
    // Full path to the current log file. Empty path if not initialised.
    //
    [[nodiscard]] static const std::filesystem::path& log_path() noexcept;

    //
    // Directory holding the current log file (results/<project>/<timestamp>/).
    // Empty if not initialised.  Lets the results writer colocate output files
    // with the log.
    //
    [[nodiscard]] static std::filesystem::path run_dir();

    Logger(const Logger&)            = delete;
    Logger& operator=(const Logger&) = delete;

private:
    Logger() = default;

    // Current timestamp as "HH:MM:SS".
    static std::string timestamp_();

    // Level label string.
    static const char* level_str_(LogLevel level);

    static std::ofstream         file_;
    static std::mutex            mutex_;
    static std::filesystem::path log_path_;
    static bool                  initialized_;
    static bool                  paused_;
};

//
// Create (mkdir -p) a unique timestamped run directory
//     <results_dir>/<project_name>/<YYYY-MM-DD_HH-MM-SS-mmm>
// and return its path.  Used both by Logger::init and by the results writer so
// that, when logging is disabled, output still has a run directory to live in.
//
[[nodiscard]] std::filesystem::path make_run_dir(const std::string& results_dir,
                                                 const std::string& project_name);

}  // namespace alea::io
