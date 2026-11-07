// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "io/logger.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Read the entire contents of a file.
std::string read_file(const fs::path& path) {
    std::ifstream f(path);
    REQUIRE(f.is_open());
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    return content;
}

// Count non-empty lines in a file.
int count_nonempty_lines(const fs::path& path) {
    std::ifstream f(path);
    int n = 0;
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) ++n;
    return n;
}

// Fixture: creates a temporary directory and cleans up on destruction.
struct TempLogDir {
    fs::path path;

    explicit TempLogDir(const std::string& suffix) {
        path = fs::temp_directory_path() / ("alea_test_" + suffix);
        fs::remove_all(path);  // clean up any previous run
    }

    ~TempLogDir() {
        alea::io::Logger::shutdown();
        std::error_code ec;
        fs::remove_all(path, ec);  // best-effort cleanup
    }
};

// Regex: [HH:MM:SS] level: message
const std::regex kLogPattern(R"(\[\d{2}:\d{2}:\d{2}\] (info|warn|error): .+)");

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Logger::init creates directory structure and log file", "[logger]") {
    TempLogDir tmp("init_structure");
    alea::io::Logger::init(tmp.path.string(), "test_project");

    // results/<project>/ directory should exist.
    auto proj_dir = tmp.path / "test_project";
    REQUIRE(fs::exists(proj_dir));
    REQUIRE(fs::is_directory(proj_dir));

    // Exactly one timestamped subdirectory.
    std::vector<fs::path> subdirs;
    for (const auto& entry : fs::directory_iterator(proj_dir)) {
        if (entry.is_directory()) subdirs.push_back(entry.path());
    }
    REQUIRE(subdirs.size() == 1);

    // alea.log should exist inside it.
    auto log_path = subdirs[0] / "alea.log";
    REQUIRE(fs::exists(log_path));
}

TEST_CASE("Logger::init writes program header", "[logger]") {
    TempLogDir tmp("init_header");
    alea::io::Logger::init(tmp.path.string(), "exciton");

    auto log_path = alea::io::Logger::log_path();
    auto content  = read_file(log_path);

    // Header line must contain the program name.
    REQUIRE_THAT(content,
                 Catch::Matchers::ContainsSubstring(
                     "Alea: A quantum Monte Carlo program"));
}

TEST_CASE("Logger::init generates unique timestamp directories", "[logger]") {
    TempLogDir tmp("init_unique");

    alea::io::Logger::init(tmp.path.string(), "proj");
    auto first = alea::io::Logger::log_path();

    alea::io::Logger::shutdown();

    // Second init to the same project should create a different directory.
    alea::io::Logger::init(tmp.path.string(), "proj");
    auto second = alea::io::Logger::log_path();

    REQUIRE(first != second);
}

// ─────────────────────────────────────────────────────────────────────────────
// Info
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Logger::info writes timestamped message", "[logger]") {
    TempLogDir tmp("info_msg");
    alea::io::Logger::init(tmp.path.string(), "proj");

    alea::io::Logger::info("accumulation: running 100000 blocks...");
    alea::io::Logger::flush();

    auto content = read_file(alea::io::Logger::log_path());

    // Should have the header line + the info line.
    int lines = count_nonempty_lines(alea::io::Logger::log_path());
    REQUIRE(lines == 2);

    REQUIRE_THAT(content, Catch::Matchers::ContainsSubstring(
                              "info: accumulation: running 100000 blocks..."));
}

TEST_CASE("Logger::info line matches log format", "[logger]") {
    TempLogDir tmp("info_format");
    alea::io::Logger::init(tmp.path.string(), "proj");

    alea::io::Logger::info("test message");
    alea::io::Logger::flush();

    auto content = read_file(alea::io::Logger::log_path());

    // Extract the second line (first non-header line).
    std::istringstream iss(content);
    std::string line;
    std::getline(iss, line);  // skip header
    std::getline(iss, line);  // the info line

    REQUIRE(std::regex_match(line, kLogPattern));
}

// ─────────────────────────────────────────────────────────────────────────────
// Warn
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Logger::warn writes timestamped message", "[logger]") {
    TempLogDir tmp("warn_msg");
    alea::io::Logger::init(tmp.path.string(), "proj");

    alea::io::Logger::warn("wall time limit approaching");
    // warn flushes immediately, no explicit flush needed.

    auto content = read_file(alea::io::Logger::log_path());
    REQUIRE_THAT(
        content,
        Catch::Matchers::ContainsSubstring(
            "warn: wall time limit approaching"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Logger::error writes timestamped message", "[logger]") {
    TempLogDir tmp("error_msg");
    alea::io::Logger::init(tmp.path.string(), "proj");

    alea::io::Logger::error("failed to open configuration file");
    // error flushes immediately, no explicit flush needed.

    auto content = read_file(alea::io::Logger::log_path());
    REQUIRE_THAT(
        content,
        Catch::Matchers::ContainsSubstring(
            "error: failed to open configuration file"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-line / append
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Logger appends messages in order", "[logger]") {
    TempLogDir tmp("append");
    alea::io::Logger::init(tmp.path.string(), "proj");

    alea::io::Logger::info("line 1");
    alea::io::Logger::info("line 2");
    alea::io::Logger::info("line 3");
    alea::io::Logger::flush();

    auto content = read_file(alea::io::Logger::log_path());

    // Check relative order.
    auto pos1 = content.find("line 1");
    auto pos2 = content.find("line 2");
    auto pos3 = content.find("line 3");

    REQUIRE(pos1 != std::string::npos);
    REQUIRE(pos2 != std::string::npos);
    REQUIRE(pos3 != std::string::npos);
    REQUIRE(pos1 < pos2);
    REQUIRE(pos2 < pos3);
}

TEST_CASE("Logger handles empty messages gracefully", "[logger]") {
    TempLogDir tmp("empty");
    alea::io::Logger::init(tmp.path.string(), "proj");

    REQUIRE_NOTHROW(alea::io::Logger::info(""));
    REQUIRE_NOTHROW(alea::io::Logger::warn(""));
    REQUIRE_NOTHROW(alea::io::Logger::error(""));
    alea::io::Logger::flush();

    // File should exist and be readable (no crash).
    auto content = read_file(alea::io::Logger::log_path());
    REQUIRE(!content.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Milestone
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Logger::milestone writes info-level line with immediate flush",
          "[logger]") {
    TempLogDir tmp("milestone");
    alea::io::Logger::init(tmp.path.string(), "proj");

    // Write a milestone — no explicit flush needed.
    alea::io::Logger::milestone("accumulation 50%");

    // Must be readable immediately (no flush call).
    auto content = read_file(alea::io::Logger::log_path());
    REQUIRE_THAT(content, Catch::Matchers::ContainsSubstring("accumulation 50%"));
}

TEST_CASE("Logger::milestone is suppressed by pause", "[logger]") {
    TempLogDir tmp("milestone_p");
    alea::io::Logger::init(tmp.path.string(), "proj");

    alea::io::Logger::pause();
    alea::io::Logger::milestone("should not appear");
    alea::io::Logger::resume();

    auto content = read_file(alea::io::Logger::log_path());
    REQUIRE_THAT(content,
                 !Catch::Matchers::ContainsSubstring("should not appear"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Pause / resume
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Logger::pause suppresses info messages", "[logger]") {
    TempLogDir tmp("pause");
    alea::io::Logger::init(tmp.path.string(), "proj");

    alea::io::Logger::info("before pause");
    alea::io::Logger::pause();
    alea::io::Logger::info("should not appear");
    alea::io::Logger::info("also suppressed");
    alea::io::Logger::resume();
    alea::io::Logger::info("after resume");
    alea::io::Logger::flush();

    auto content = read_file(alea::io::Logger::log_path());
    REQUIRE_THAT(content, Catch::Matchers::ContainsSubstring("before pause"));
    REQUIRE_THAT(content, Catch::Matchers::ContainsSubstring("after resume"));
    REQUIRE_THAT(content,
                 !Catch::Matchers::ContainsSubstring("should not appear"));
    REQUIRE_THAT(content,
                 !Catch::Matchers::ContainsSubstring("also suppressed"));
}

TEST_CASE("Logger::pause does not block warn or error", "[logger]") {
    TempLogDir tmp("pause_warn");
    alea::io::Logger::init(tmp.path.string(), "proj");

    alea::io::Logger::pause();
    alea::io::Logger::warn("warning during pause");
    alea::io::Logger::error("error during pause");
    alea::io::Logger::resume();

    auto content = read_file(alea::io::Logger::log_path());
    REQUIRE_THAT(content,
                 Catch::Matchers::ContainsSubstring("warning during pause"));
    REQUIRE_THAT(content,
                 Catch::Matchers::ContainsSubstring("error during pause"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Thread safety
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Logger is thread-safe for concurrent writes", "[logger]") {
    TempLogDir tmp("threads");
    alea::io::Logger::init(tmp.path.string(), "proj");

    constexpr int n_threads   = 8;
    constexpr int n_msgs      = 100;
    constexpr int total_lines = n_threads * n_msgs;

    auto worker = [](int tid) {
        for (int i = 0; i < n_msgs; ++i) {
            alea::io::Logger::info("thread " + std::to_string(tid)
                                   + " msg " + std::to_string(i));
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t)
        threads.emplace_back(worker, t);

    for (auto& t : threads)
        t.join();

    alea::io::Logger::flush();

    // Every line after the header should match the pattern
    // (no interleaved/corrupted lines).
    std::ifstream f(alea::io::Logger::log_path());
    std::string   line;
    int           info_lines = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        // Skip header: contains "Alea:"
        if (line.find("Alea:") != std::string::npos) continue;

        ++info_lines;
        REQUIRE(std::regex_match(line, kLogPattern));
    }

    REQUIRE(info_lines == total_lines);
}

// ─────────────────────────────────────────────────────────────────────────────
// Shutdown / re-init
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Logger::shutdown closes file and allows re-init", "[logger]") {
    TempLogDir tmp("reinit");

    // First session.
    alea::io::Logger::init(tmp.path.string(), "proj");
    alea::io::Logger::info("session 1");
    alea::io::Logger::shutdown();

    // Second session — must not throw.
    REQUIRE_NOTHROW(alea::io::Logger::init(tmp.path.string(), "proj"));
    alea::io::Logger::info("session 2");
    alea::io::Logger::flush();

    auto content = read_file(alea::io::Logger::log_path());
    REQUIRE_THAT(content, Catch::Matchers::ContainsSubstring("session 2"));
}

TEST_CASE("Logger::log_path returns correct path after init", "[logger]") {
    TempLogDir tmp("path");
    alea::io::Logger::init(tmp.path.string(), "proj");

    auto p = alea::io::Logger::log_path();
    REQUIRE(p.extension() == ".log");
    REQUIRE(p.filename() == "alea.log");
    REQUIRE(p.parent_path().parent_path().filename() == "proj");
    REQUIRE(p.parent_path().parent_path().parent_path() == tmp.path);
}

// ─────────────────────────────────────────────────────────────────────────────
// Calling log before init is a no-op (safe default)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Logger gracefully handles calls before init", "[logger]") {
    // After any prior shutdown, logging should not crash.
    alea::io::Logger::shutdown();
    REQUIRE_NOTHROW(alea::io::Logger::info("should not crash"));
    REQUIRE_NOTHROW(alea::io::Logger::warn("should not crash"));
    REQUIRE_NOTHROW(alea::io::Logger::error("should not crash"));
    REQUIRE_NOTHROW(alea::io::Logger::flush());
    REQUIRE_NOTHROW(alea::io::Logger::shutdown());
}
