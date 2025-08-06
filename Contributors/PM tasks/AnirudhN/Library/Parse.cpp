#include "csv.hpp"

/**
 * @brief Helper to parse a string to float. Returns nan_value on failure.
 */
float safeParseFloat(const std::string& s, float nan_value) {
    const char* str = s.c_str();
    char* endptr = nullptr;
    errno = 0;
    float val = std::strtof(str, &endptr);
    if (endptr == str) {
        // no conversion performed
        return nan_value;
    }
    // skip trailing whitespace
    while (*endptr && std::isspace(static_cast<unsigned char>(*endptr))) {
        ++endptr;
    }
    if (*endptr != '\0') {
        // leftover junk (e.g., "12.3abc")
        return nan_value;
    }
    if (errno == ERANGE) {
        // out of range
        return nan_value;
    }
    return val;
}

/**
 * @brief Parses a CSV file in a single pass, pads missing values with NaN,
 * dynamically adjusts to the widest row seen, and returns the data as an Eigen::MatrixXf.
 * Individual field parse failures become NaN; entire rows are kept.
 *
 * @param filename Path to CSV.
 * @param skip_header If true, the first row is skipped.
 * @return Eigen::MatrixXf with data, missing/invalid entries as NaN.
 * @throws std::runtime_error on critical I/O/parsing failure.
 */
Eigen::MatrixXf parseCsvToEigenSinglePass(const std::string& filename, bool skip_header = false) {
    std::vector<std::vector<float>> buffer;
    int max_cols = 0;
    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    std::size_t total_fields = 0;
    std::size_t invalid_fields = 0;

    try {
        csv::CSVReader reader(filename);

        auto it = reader.begin();
        if (skip_header && it != reader.end()) {
            ++it; // skip header row properly
        }

        for (; it != reader.end(); ++it) {
            auto& row = *it;
            int current_row_size = static_cast<int>(row.size());
            if (current_row_size > max_cols) {
                // Grow previous rows to match new width
                for (auto& prev_row : buffer) {
                    prev_row.resize(current_row_size, nan_value);
                }
                max_cols = current_row_size;
            }

            std::vector<float> row_data;
            row_data.reserve(max_cols); // we might push less if new max_cols increases later; we'll pad below

            for (int i = 0; i < current_row_size; ++i) {
                std::string raw = row[i].get<>(); // get as string
                float parsed = safeParseFloat(raw, nan_value);
                if (std::isnan(parsed)) {
                    ++invalid_fields;
                }
                row_data.push_back(parsed);
                ++total_fields;
            }

            // If this row is narrower than current max_cols, pad with NaNs
            if (current_row_size < max_cols) {
                int diff = max_cols - current_row_size;
                for (int k = 0; k < diff; ++k) {
                    row_data.push_back(nan_value);
                    ++total_fields;
                    ++invalid_fields; // padded fields count as missing
                }
            }

            buffer.push_back(std::move(row_data));
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Error during CSV single-pass parsing: " + std::string(e.what()));
    }

    if (buffer.empty()) {
        std::cerr << "No data rows found (after optional header skip)." << std::endl;
        return Eigen::MatrixXf::Zero(0, 0);
    }

    int num_rows = static_cast<int>(buffer.size());
    Eigen::MatrixXf result(num_rows, max_cols);

    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < max_cols; ++j) {
            result(i, j) = buffer[i][j];
        }
    }

    std::cerr << "Parsed " << num_rows << " rows and inferred " << max_cols << " columns in one pass." << std::endl;
    std::cerr << "Total fields seen: " << total_fields << ", invalid/missing fields (as NaN): " << invalid_fields << "." << std::endl;

    return result;
}
