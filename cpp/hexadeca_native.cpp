// C++20 implementation for profile-validated Hexadeca hot paths.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr std::uint64_t kUint64Mask = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kSplitmixIncrement = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t kSplitmixMultiplier1 = 0xBF58476D1CE4E5B9ULL;
constexpr std::uint64_t kSplitmixMultiplier2 = 0x94D049BB133111EBULL;
constexpr int kBlack = 1;
constexpr int kWhite = 2;
constexpr int kFeaturePlanes = 16;
constexpr int kHistoryPositions = 4;

int opponent(const int player) {
    return player == kBlack ? kWhite : kBlack;
}

std::pair<std::uint64_t, std::uint64_t> next_splitmix64(const std::uint64_t state) {
    const auto next_state = state + kSplitmixIncrement;
    auto value = next_state;
    value = ((value ^ (value >> 30U)) * kSplitmixMultiplier1) & kUint64Mask;
    value = ((value ^ (value >> 27U)) * kSplitmixMultiplier2) & kUint64Mask;
    value ^= value >> 31U;
    return {next_state, value & kUint64Mask};
}

struct NativeScore {
    int black_score{};
    int white_score{};
    std::vector<int> owners;
    std::vector<int> distances;
};

class BoardCore {
public:
    BoardCore(const int board_size, const int neighborhood_radius,
              const std::uint64_t zobrist_seed)
        : size_(board_size), action_size_(board_size * board_size),
          radius_(neighborhood_radius), cells_(static_cast<std::size_t>(action_size_), 0),
          blocked_counts_(static_cast<std::size_t>(action_size_), 0),
          legal_count_(action_size_), to_play_(kBlack) {
        if (board_size <= 0 || neighborhood_radius <= 0) {
            throw std::invalid_argument("Board size and neighborhood radius must be positive");
        }
        initialize_zobrist(zobrist_seed);
        zobrist_hash_ = side_keys_[0];
    }

    int size() const { return size_; }
    int action_size() const { return action_size_; }
    int ply() const { return static_cast<int>(history_.size()); }
    int to_play() const { return to_play_; }
    int legal_count() const { return legal_count_; }
    bool is_terminal() const { return legal_count_ == 0; }
    std::uint64_t zobrist_hash() const { return zobrist_hash_; }
    const std::vector<std::uint8_t>& cells() const { return cells_; }
    const std::vector<int>& history() const { return history_; }

    bool is_legal(const int action) const {
        return action >= 0 && action < action_size_ && cells_[action] == 0 &&
               blocked_counts_[action] == 0;
    }

    int piece_at(const int action) const {
        validate_action(action);
        return cells_[action];
    }

    std::vector<int> legal_actions() const {
        std::vector<int> actions;
        actions.reserve(static_cast<std::size_t>(legal_count_));
        for (int action = 0; action < action_size_; ++action) {
            if (is_legal(action)) {
                actions.push_back(action);
            }
        }
        return actions;
    }

    std::vector<bool> legal_mask() const {
        std::vector<bool> result(static_cast<std::size_t>(action_size_));
        for (int action = 0; action < action_size_; ++action) {
            result[static_cast<std::size_t>(action)] = is_legal(action);
        }
        return result;
    }

    int apply(const int action) {
        validate_action(action);
        if (!is_legal(action)) {
            throw std::domain_error("Illegal placement");
        }

        const int player = to_play_;
        for_neighborhood(action, [this](const int blocked_action) {
            if (cells_[blocked_action] == 0 && blocked_counts_[blocked_action] == 0) {
                --legal_count_;
            }
            ++blocked_counts_[blocked_action];
        });
        cells_[action] = static_cast<std::uint8_t>(player);
        toggle_hash_for_move(action, player);
        to_play_ = opponent(player);
        history_.push_back(action);
        return player;
    }

    int undo() {
        if (history_.empty()) {
            throw std::domain_error("Cannot undo an empty board history");
        }
        const int action = history_.back();
        const int player = opponent(to_play_);
        cells_[action] = 0;
        for_neighborhood(action, [this](const int blocked_action) {
            if (blocked_counts_[blocked_action] == 0) {
                throw std::runtime_error("Blocked-count state is internally inconsistent");
            }
            --blocked_counts_[blocked_action];
            if (cells_[blocked_action] == 0 && blocked_counts_[blocked_action] == 0) {
                ++legal_count_;
            }
        });
        to_play_ = player;
        toggle_hash_for_move(action, player);
        history_.pop_back();
        return action;
    }

    NativeScore score() const {
        return score_cells(cells_, size_);
    }

    static NativeScore score_cells(const std::vector<std::uint8_t>& cells,
                                   const int board_size) {
        if (board_size <= 0 || static_cast<int>(cells.size()) != board_size * board_size) {
            throw std::invalid_argument("Cell count does not match board size");
        }
        std::vector<std::pair<int, int>> stones;
        stones.reserve(cells.size());
        for (int action = 0; action < static_cast<int>(cells.size()); ++action) {
            const int player = cells[static_cast<std::size_t>(action)];
            if (player != 0 && player != kBlack && player != kWhite) {
                throw std::invalid_argument("Cells must use encodings 0, 1, and 2");
            }
            if (player != 0) {
                stones.emplace_back(action, player);
            }
        }

        NativeScore result;
        result.owners.assign(cells.size(), 0);
        result.distances.assign(cells.size(), -1);
        const int maximum_distance = 2 * (board_size - 1) * (board_size - 1);
        std::vector<int> black_counts(static_cast<std::size_t>(maximum_distance + 1));
        std::vector<int> white_counts(static_cast<std::size_t>(maximum_distance + 1));
        std::vector<int> touched;
        touched.reserve(stones.size());

        for (int action = 0; action < static_cast<int>(cells.size()); ++action) {
            const int occupant = cells[static_cast<std::size_t>(action)];
            if (occupant != 0) {
                result.owners[static_cast<std::size_t>(action)] = occupant;
                result.distances[static_cast<std::size_t>(action)] = 0;
                if (occupant == kBlack) {
                    ++result.black_score;
                } else {
                    ++result.white_score;
                }
                continue;
            }

            const int row = action / board_size;
            const int column = action % board_size;
            for (const auto [stone_action, player] : stones) {
                const int stone_row = stone_action / board_size;
                const int stone_column = stone_action % board_size;
                const int row_delta = row - stone_row;
                const int column_delta = column - stone_column;
                const int distance = row_delta * row_delta + column_delta * column_delta;
                if (black_counts[static_cast<std::size_t>(distance)] == 0 &&
                    white_counts[static_cast<std::size_t>(distance)] == 0) {
                    touched.push_back(distance);
                }
                if (player == kBlack) {
                    ++black_counts[static_cast<std::size_t>(distance)];
                } else {
                    ++white_counts[static_cast<std::size_t>(distance)];
                }
            }
            std::sort(touched.begin(), touched.end());
            for (const int distance : touched) {
                const int black = black_counts[static_cast<std::size_t>(distance)];
                const int white = white_counts[static_cast<std::size_t>(distance)];
                if (black > white) {
                    result.owners[static_cast<std::size_t>(action)] = kBlack;
                    result.distances[static_cast<std::size_t>(action)] = distance;
                    ++result.black_score;
                    break;
                }
                if (white > black) {
                    result.owners[static_cast<std::size_t>(action)] = kWhite;
                    result.distances[static_cast<std::size_t>(action)] = distance;
                    ++result.white_score;
                    break;
                }
            }
            for (const int distance : touched) {
                black_counts[static_cast<std::size_t>(distance)] = 0;
                white_counts[static_cast<std::size_t>(distance)] = 0;
            }
            touched.clear();
        }
        return result;
    }

private:
    int size_;
    int action_size_;
    int radius_;
    std::vector<std::uint8_t> cells_;
    std::vector<std::uint16_t> blocked_counts_;
    std::vector<int> history_;
    int legal_count_;
    int to_play_;
    std::vector<std::array<std::uint64_t, 2>> piece_keys_;
    std::array<std::uint64_t, 2> side_keys_{};
    std::uint64_t zobrist_hash_{};

    void validate_action(const int action) const {
        if (action < 0 || action >= action_size_) {
            throw std::out_of_range("Action is outside the board");
        }
    }

    template <typename Function>
    void for_neighborhood(const int action, Function&& function) const {
        const int row = action / size_;
        const int column = action % size_;
        const int row_start = std::max(0, row - radius_);
        const int row_stop = std::min(size_, row + radius_ + 1);
        const int column_start = std::max(0, column - radius_);
        const int column_stop = std::min(size_, column + radius_ + 1);
        for (int neighbor_row = row_start; neighbor_row < row_stop; ++neighbor_row) {
            const int base = neighbor_row * size_;
            for (int neighbor_column = column_start; neighbor_column < column_stop;
                 ++neighbor_column) {
                function(base + neighbor_column);
            }
        }
    }

    void initialize_zobrist(std::uint64_t state) {
        piece_keys_.reserve(static_cast<std::size_t>(action_size_));
        for (int action = 0; action < action_size_; ++action) {
            const auto [black_state, black_key] = next_splitmix64(state);
            state = black_state;
            const auto [white_state, white_key] = next_splitmix64(state);
            state = white_state;
            piece_keys_.push_back({black_key, white_key});
        }
        const auto [black_state, black_side_key] = next_splitmix64(state);
        const auto [white_state, white_side_key] = next_splitmix64(black_state);
        side_keys_ = {black_side_key, white_side_key};
    }

    void toggle_hash_for_move(const int action, const int player) {
        const int old_player = to_play_;
        const int new_player = opponent(old_player);
        zobrist_hash_ ^= side_keys_[static_cast<std::size_t>(old_player - 1)];
        zobrist_hash_ ^= piece_keys_[static_cast<std::size_t>(action)]
                                    [static_cast<std::size_t>(player - 1)];
        zobrist_hash_ ^= side_keys_[static_cast<std::size_t>(new_player - 1)];
    }
};

std::vector<std::uint8_t> extract_cells(const py::sequence& values,
                                        const int action_size) {
    if (py::len(values) != action_size) {
        throw std::invalid_argument("State cell count does not match policy_size");
    }
    std::vector<std::uint8_t> cells(static_cast<std::size_t>(action_size));
    for (int action = 0; action < action_size; ++action) {
        const py::handle value_object = values[static_cast<std::size_t>(action)];
        if (!py::isinstance<py::int_>(value_object) ||
            py::isinstance<py::bool_>(value_object)) {
            throw std::invalid_argument("State cells must use encodings 0, 1, and 2");
        }
        const int value = value_object.cast<int>();
        if (value < 0 || value > 2) {
            throw std::invalid_argument("State cells must use encodings 0, 1, and 2");
        }
        cells[static_cast<std::size_t>(action)] = static_cast<std::uint8_t>(value);
    }
    return cells;
}

struct FeatureInput {
    std::vector<std::uint8_t> cells;
    std::vector<std::uint8_t> legal_mask;
    std::vector<std::pair<int, int>> history;
    int to_play{};
};

FeatureInput extract_feature_input(const py::handle& state, const int board_size,
                                   const std::string& ruleset_id) {
    const int action_size = board_size * board_size;
    if (state.attr("ruleset_id").cast<std::string>() != ruleset_id) {
        throw std::invalid_argument("State ruleset does not match the network specification");
    }
    if (state.attr("board_size").cast<int>() != board_size) {
        throw std::invalid_argument("State board size does not match the network specification");
    }
    FeatureInput input;
    input.cells = extract_cells(state.attr("cells").cast<py::sequence>(), action_size);
    const py::sequence legal_values = state.attr("legal_mask").cast<py::sequence>();
    if (py::len(legal_values) != action_size) {
        throw std::invalid_argument("State legal mask does not match policy_size");
    }
    input.legal_mask.resize(static_cast<std::size_t>(action_size));
    for (int action = 0; action < action_size; ++action) {
        const py::handle value = legal_values[static_cast<std::size_t>(action)];
        if (!py::isinstance<py::bool_>(value)) {
            throw std::invalid_argument("State legal mask values must be booleans");
        }
        input.legal_mask[static_cast<std::size_t>(action)] = value.cast<bool>() ? 1 : 0;
    }

    const py::sequence moves = state.attr("history").cast<py::sequence>();
    if (state.attr("ply").cast<int>() != py::len(moves)) {
        throw std::invalid_argument("State ply does not match its history length");
    }
    std::vector<std::uint8_t> reconstructed(static_cast<std::size_t>(action_size));
    int expected_player = kBlack;
    input.history.reserve(static_cast<std::size_t>(py::len(moves)));
    for (const py::handle move : moves) {
        const int action = move.attr("action").cast<int>();
        const int row = move.attr("row").cast<int>();
        const int column = move.attr("column").cast<int>();
        const int player = move.attr("player").attr("value").cast<int>();
        if (player != expected_player) {
            throw std::invalid_argument("State history does not alternate players");
        }
        if (action < 0 || action >= action_size) {
            throw std::invalid_argument("State history contains an invalid action");
        }
        if (action / board_size != row || action % board_size != column) {
            throw std::invalid_argument("State history coordinates do not match action");
        }
        if (reconstructed[static_cast<std::size_t>(action)] != 0) {
            throw std::invalid_argument("State history places on an occupied action");
        }
        reconstructed[static_cast<std::size_t>(action)] = static_cast<std::uint8_t>(player);
        input.history.emplace_back(action, player);
        expected_player = opponent(expected_player);
    }
    if (reconstructed != input.cells) {
        throw std::invalid_argument("State cells do not match its move history");
    }
    input.to_play = state.attr("to_play").attr("value").cast<int>();
    if (input.to_play != expected_player) {
        throw std::invalid_argument("State side to play does not match its history");
    }
    return input;
}

py::array_t<float> encode_feature_inputs(const std::vector<FeatureInput>& inputs,
                                         const int board_size, const int input_planes) {
    const std::array<py::ssize_t, 4> shape = {
        static_cast<py::ssize_t>(inputs.size()), input_planes, board_size, board_size};
    py::array_t<float> output(shape);
    float* const data = output.mutable_data();
    const std::size_t plane_area = static_cast<std::size_t>(board_size * board_size);
    const std::size_t state_area = static_cast<std::size_t>(input_planes) * plane_area;
    std::fill(data, data + inputs.size() * state_area, 0.0F);
    {
        py::gil_scoped_release release;
        for (std::size_t batch_index = 0; batch_index < inputs.size(); ++batch_index) {
            const FeatureInput& input = inputs[batch_index];
            float* const state_output = data + batch_index * state_area;
            auto plane = [state_output, plane_area](const int index) {
                return state_output + static_cast<std::size_t>(index) * plane_area;
            };
            for (std::size_t action = 0; action < plane_area; ++action) {
                const int cell = input.cells[action];
                plane(0)[action] = cell == kBlack ? 1.0F : 0.0F;
                plane(1)[action] = cell == kWhite ? 1.0F : 0.0F;
                plane(2)[action] = input.legal_mask[action] == 1 ? 1.0F : 0.0F;
                plane(3)[action] = input.to_play == kBlack ? 1.0F : 0.0F;
            }
            std::vector<std::uint8_t> historical = input.cells;
            for (int offset = 0; offset < kHistoryPositions; ++offset) {
                if (offset < static_cast<int>(input.history.size())) {
                    const auto [action, _] = input.history[input.history.size() - 1 - offset];
                    historical[static_cast<std::size_t>(action)] = 0;
                }
                const int plane_offset = 4 + offset * 2;
                for (std::size_t action = 0; action < plane_area; ++action) {
                    plane(plane_offset)[action] = historical[action] == kBlack ? 1.0F : 0.0F;
                    plane(plane_offset + 1)[action] = historical[action] == kWhite ? 1.0F : 0.0F;
                }
            }
            bool black_seen = false;
            bool white_seen = false;
            for (auto iterator = input.history.rbegin(); iterator != input.history.rend();
                 ++iterator) {
                const auto [action, player] = *iterator;
                if (player == kBlack && !black_seen) {
                    plane(12)[static_cast<std::size_t>(action)] = 1.0F;
                    black_seen = true;
                }
                if (player == kWhite && !white_seen) {
                    plane(13)[static_cast<std::size_t>(action)] = 1.0F;
                    white_seen = true;
                }
                if (black_seen && white_seen) {
                    break;
                }
            }
            for (int row = 0; row < board_size; ++row) {
                const float coordinate_step = board_size == 1
                    ? 0.0F
                    : 2.0F / static_cast<float>(board_size - 1);
                const auto coordinate = [board_size, coordinate_step](const int index) {
                    if (board_size == 1) {
                        return -1.0F;
                    }
                    if (index < board_size / 2) {
                        return -1.0F + coordinate_step * static_cast<float>(index);
                    }
                    return 1.0F - coordinate_step *
                        static_cast<float>(board_size - index - 1);
                };
                const float row_coordinate = coordinate(row);
                for (int column = 0; column < board_size; ++column) {
                    const float column_coordinate = coordinate(column);
                    const std::size_t action = static_cast<std::size_t>(row * board_size + column);
                    plane(14)[action] = row_coordinate;
                    plane(15)[action] = column_coordinate;
                }
            }
        }
    }
    return output;
}

std::uint16_t read_u16(const std::string& payload, const std::size_t offset) {
    if (offset + 2 > payload.size()) {
        throw std::invalid_argument("Compact inference packet is truncated");
    }
    return static_cast<std::uint16_t>(static_cast<unsigned char>(payload[offset])) |
           (static_cast<std::uint16_t>(static_cast<unsigned char>(payload[offset + 1])) << 8U);
}

std::uint64_t read_u64(const std::string& payload, const std::size_t offset) {
    if (offset + 8 > payload.size()) {
        throw std::invalid_argument("Compact inference packet is truncated");
    }
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(
            static_cast<unsigned char>(payload[offset + index])) << (index * 8U);
    }
    return value;
}

std::uint32_t read_u32(const std::string& payload, const std::size_t offset) {
    if (offset + 4 > payload.size()) {
        throw std::invalid_argument("Compact inference packet is truncated");
    }
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < 4; ++index) {
        value |= static_cast<std::uint32_t>(
            static_cast<unsigned char>(payload[offset + index])) << (index * 8U);
    }
    return value;
}

float read_float32(const std::string& payload, const std::size_t offset) {
    const std::uint32_t bits = read_u32(payload, offset);
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void append_u16(std::string& payload, const std::uint16_t value) {
    payload.push_back(static_cast<char>(value & 0xFFU));
    payload.push_back(static_cast<char>((value >> 8U) & 0xFFU));
}

void append_u32(std::string& payload, const std::uint32_t value) {
    for (std::size_t index = 0; index < 4; ++index) {
        payload.push_back(static_cast<char>((value >> (index * 8U)) & 0xFFU));
    }
}

void append_u64(std::string& payload, const std::uint64_t value) {
    for (std::size_t index = 0; index < 8; ++index) {
        payload.push_back(static_cast<char>((value >> (index * 8U)) & 0xFFU));
    }
}

void append_float32(std::string& payload, const float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    append_u32(payload, bits);
}

FeatureInput extract_packed_feature_input(const std::string& record, const int board_size) {
    constexpr std::size_t header_size = 12;
    constexpr unsigned char legacy_packet_version = 1;
    constexpr unsigned char wide_packet_version = 2;
    const int action_size = board_size * board_size;
    const std::size_t legal_byte_count = static_cast<std::size_t>((action_size + 7) / 8);
    const std::size_t minimum_size = header_size + static_cast<std::size_t>(action_size) +
        legal_byte_count;
    if (record.size() < minimum_size) {
        throw std::invalid_argument("Compact state record is truncated");
    }
    const auto packet_version = static_cast<unsigned char>(record[0]);
    if (packet_version != legacy_packet_version && packet_version != wide_packet_version) {
        throw std::invalid_argument("Compact state protocol version is unsupported");
    }
    const int to_play = static_cast<unsigned char>(record[1]);
    if (to_play != kBlack && to_play != kWhite) {
        throw std::invalid_argument("Compact state player is invalid");
    }
    const std::size_t history_size = read_u16(record, 2);
    const std::size_t history_width = packet_version == legacy_packet_version ? 1U : 2U;
    static_cast<void>(read_u64(record, 4));
    if (history_size > static_cast<std::size_t>(action_size) ||
        record.size() != minimum_size + history_size * history_width) {
        throw std::invalid_argument("Compact state history length is invalid");
    }

    FeatureInput input;
    input.cells.resize(static_cast<std::size_t>(action_size));
    for (int action = 0; action < action_size; ++action) {
        const auto value = static_cast<std::uint8_t>(
            static_cast<unsigned char>(record[header_size + static_cast<std::size_t>(action)]));
        if (value > 2) {
            throw std::invalid_argument("Compact state cells are invalid");
        }
        input.cells[static_cast<std::size_t>(action)] = value;
    }
    const std::size_t legal_offset = header_size + static_cast<std::size_t>(action_size);
    if (action_size % 8 != 0) {
        const unsigned char padding = static_cast<unsigned char>(
            record[legal_offset + legal_byte_count - 1]) >> (action_size % 8);
        if (padding != 0) {
            throw std::invalid_argument("Compact state legal mask has invalid padding");
        }
    }
    input.legal_mask.resize(static_cast<std::size_t>(action_size));
    for (int action = 0; action < action_size; ++action) {
        const auto byte = static_cast<unsigned char>(
            record[legal_offset + static_cast<std::size_t>(action / 8)]);
        input.legal_mask[static_cast<std::size_t>(action)] =
            (byte & (1U << (action % 8))) != 0U ? 1 : 0;
    }

    std::vector<std::uint8_t> reconstructed(static_cast<std::size_t>(action_size));
    input.history.reserve(history_size);
    int expected_player = kBlack;
    const std::size_t history_offset = legal_offset + legal_byte_count;
    for (std::size_t index = 0; index < history_size; ++index) {
        const std::size_t action_offset = history_offset + index * history_width;
        const int action = packet_version == legacy_packet_version
            ? static_cast<unsigned char>(record[action_offset])
            : static_cast<int>(read_u16(record, action_offset));
        if (action >= action_size) {
            throw std::invalid_argument("Compact state history contains an invalid action");
        }
        if (reconstructed[static_cast<std::size_t>(action)] != 0) {
            throw std::invalid_argument("Compact state history contains duplicates");
        }
        reconstructed[static_cast<std::size_t>(action)] =
            static_cast<std::uint8_t>(expected_player);
        input.history.emplace_back(action, expected_player);
        expected_player = opponent(expected_player);
    }
    if (reconstructed != input.cells) {
        throw std::invalid_argument("Compact state cells do not match history");
    }
    if (to_play != expected_player) {
        throw std::invalid_argument("Compact state player does not match history");
    }
    input.to_play = to_play;
    return input;
}

py::array_t<float> encode_features(const py::sequence& states, const int board_size,
                                   const int input_planes,
                                   const std::string& ruleset_id) {
    if (py::len(states) == 0) {
        throw std::invalid_argument("Cannot encode an empty state batch");
    }
    if (board_size <= 0 || input_planes != kFeaturePlanes) {
        throw std::invalid_argument("Native feature dimensions do not match the supported schema");
    }
    std::vector<FeatureInput> inputs;
    inputs.reserve(static_cast<std::size_t>(py::len(states)));
    for (const py::handle state : states) {
        inputs.push_back(extract_feature_input(state, board_size, ruleset_id));
    }
    return encode_feature_inputs(inputs, board_size, input_planes);
}

py::array_t<float> encode_packed_features(const py::bytes& packed_states,
                                          const int board_size,
                                          const int input_planes) {
    if (board_size <= 0 || board_size > 255 ||
        board_size * board_size > 65535 || input_planes != kFeaturePlanes) {
        throw std::invalid_argument("Compact feature dimensions are unsupported");
    }
    const std::string payload = packed_states;
    if (payload.size() < 2) {
        throw std::invalid_argument("Compact inference packet is truncated");
    }
    const std::size_t count = read_u16(payload, 0);
    if (count == 0) {
        throw std::invalid_argument("Compact inference packet is empty");
    }
    std::size_t offset = 2;
    std::vector<FeatureInput> inputs;
    inputs.reserve(count);
    {
        py::gil_scoped_release release;
        for (std::size_t index = 0; index < count; ++index) {
            const std::size_t record_size = read_u16(payload, offset);
            offset += 2;
            if (offset + record_size > payload.size()) {
                throw std::invalid_argument("Compact state record is truncated");
            }
            inputs.push_back(extract_packed_feature_input(
                payload.substr(offset, record_size), board_size));
            offset += record_size;
        }
    }
    if (offset != payload.size()) {
        throw std::invalid_argument("Compact inference packet has trailing data");
    }
    return encode_feature_inputs(inputs, board_size, input_planes);
}

py::dict decode_packed_hybrid_inputs(const py::bytes& packed_states,
                                     const int board_size) {
    if (board_size <= 0 || board_size > 255 ||
        board_size * board_size > 65535) {
        throw std::invalid_argument("Compact hybrid dimensions are unsupported");
    }
    const std::string payload = packed_states;
    if (payload.size() < 2) {
        throw std::invalid_argument("Compact inference packet is truncated");
    }
    const std::size_t count = read_u16(payload, 0);
    if (count == 0) {
        throw std::invalid_argument("Compact inference packet is empty");
    }
    std::size_t offset = 2;
    std::vector<FeatureInput> inputs;
    inputs.reserve(count);
    {
        py::gil_scoped_release release;
        for (std::size_t index = 0; index < count; ++index) {
            const std::size_t record_size = read_u16(payload, offset);
            offset += 2;
            if (offset + record_size > payload.size()) {
                throw std::invalid_argument("Compact state record is truncated");
            }
            inputs.push_back(extract_packed_feature_input(
                payload.substr(offset, record_size), board_size));
            offset += record_size;
        }
    }
    if (offset != payload.size()) {
        throw std::invalid_argument("Compact inference packet has trailing data");
    }

    const int action_size = board_size * board_size;
    const std::size_t byte_width = static_cast<std::size_t>(action_size * 2 + 1);
    const std::size_t action_width = static_cast<std::size_t>(kHistoryPositions + 2);
    py::array_t<std::uint8_t> byte_inputs({
        static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(byte_width)});
    py::array_t<std::int16_t> action_inputs({
        static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(action_width)});
    auto* byte_data = byte_inputs.mutable_data();
    auto* action_data = action_inputs.mutable_data();
    std::fill(action_data, action_data + count * action_width,
              static_cast<std::int16_t>(-1));

    {
        py::gil_scoped_release release;
        for (std::size_t index = 0; index < count; ++index) {
            const FeatureInput& input = inputs[index];
            auto* row_bytes = byte_data + index * byte_width;
            auto* row_actions = action_data + index * action_width;
            std::copy(input.cells.begin(), input.cells.end(),
                      row_bytes);
            std::copy(input.legal_mask.begin(), input.legal_mask.end(),
                      row_bytes + action_size);
            row_bytes[action_size * 2] = input.to_play == kBlack ? 1U : 0U;
            const std::size_t recent_count = std::min(
                input.history.size(), static_cast<std::size_t>(kHistoryPositions));
            for (std::size_t history_index = 0; history_index < recent_count;
                 ++history_index) {
                row_actions[history_index] =
                    static_cast<std::int16_t>(
                        input.history[input.history.size() - history_index - 1].first);
            }
            for (auto iterator = input.history.rbegin();
                 iterator != input.history.rend(); ++iterator) {
                const int colour_index = iterator->second == kBlack ? 0 : 1;
                auto& target = row_actions[kHistoryPositions +
                    static_cast<std::size_t>(colour_index)];
                if (target < 0) {
                    target = static_cast<std::int16_t>(iterator->first);
                }
                if (row_actions[kHistoryPositions] >= 0 &&
                    row_actions[kHistoryPositions + 1] >= 0) {
                    break;
                }
            }
        }
    }
    py::dict output;
    output["byte_inputs"] = std::move(byte_inputs);
    output["action_inputs"] = std::move(action_inputs);
    return output;
}

py::bytes pack_compact_states(const py::sequence& states, const int board_size,
                              const std::string& ruleset_id) {
    constexpr unsigned char legacy_packet_version = 1;
    constexpr unsigned char wide_packet_version = 2;
    constexpr std::size_t header_size = 12;
    if (board_size <= 0 || board_size > 255 ||
        board_size * board_size > 65535 ||
        py::len(states) == 0 || py::len(states) > 65535) {
        throw std::invalid_argument("Compact state batch dimensions are invalid");
    }
    const int action_size = board_size * board_size;
    const auto packet_version = action_size == 256
        ? legacy_packet_version
        : wide_packet_version;
    const std::size_t history_width = packet_version == legacy_packet_version ? 1U : 2U;
    const std::size_t legal_byte_count = static_cast<std::size_t>((action_size + 7) / 8);
    std::string packet;
    packet.reserve(2 + static_cast<std::size_t>(py::len(states)) *
        (2 + header_size + static_cast<std::size_t>(action_size) + legal_byte_count));
    append_u16(packet, static_cast<std::uint16_t>(py::len(states)));
    for (const py::handle state : states) {
        const FeatureInput input = extract_feature_input(state, board_size, ruleset_id);
        const auto zobrist_hash = state.attr("zobrist_hash").cast<std::uint64_t>();
        std::string record;
        record.reserve(header_size + static_cast<std::size_t>(action_size) +
            legal_byte_count + input.history.size() * history_width);
        record.push_back(static_cast<char>(packet_version));
        record.push_back(static_cast<char>(input.to_play));
        append_u16(record, static_cast<std::uint16_t>(input.history.size()));
        append_u64(record, zobrist_hash);
        for (const auto cell : input.cells) {
            record.push_back(static_cast<char>(cell));
        }
        std::string legal(legal_byte_count, '\0');
        for (int action = 0; action < action_size; ++action) {
            if (input.legal_mask[static_cast<std::size_t>(action)] == 1) {
                legal[static_cast<std::size_t>(action / 8)] = static_cast<char>(
                    static_cast<unsigned char>(legal[static_cast<std::size_t>(action / 8)]) |
                    (1U << (action % 8)));
            }
        }
        record.append(legal);
        for (const auto& [action, _] : input.history) {
            if (packet_version == legacy_packet_version) {
                record.push_back(static_cast<char>(action));
            } else {
                append_u16(record, static_cast<std::uint16_t>(action));
            }
        }
        if (record.size() > 65535) {
            throw std::invalid_argument("Compact state record exceeds protocol limit");
        }
        append_u16(packet, static_cast<std::uint16_t>(record.size()));
        packet.append(record);
    }
    return py::bytes(packet);
}

py::bytes pack_compact_evaluations(const py::sequence& evaluations,
                                   const int action_size) {
    if (action_size <= 0 || py::len(evaluations) == 0 || py::len(evaluations) > 65535) {
        throw std::invalid_argument("Compact evaluation batch dimensions are invalid");
    }
    std::string packet;
    packet.reserve(2 + static_cast<std::size_t>(py::len(evaluations)) *
        static_cast<std::size_t>(action_size + 1) * sizeof(float));
    append_u16(packet, static_cast<std::uint16_t>(py::len(evaluations)));
    for (const py::handle evaluation : evaluations) {
        const py::sequence policy = evaluation.attr("policy").cast<py::sequence>();
        if (py::len(policy) != action_size) {
            throw std::invalid_argument("Evaluation policy has an invalid length");
        }
        for (int action = 0; action < action_size; ++action) {
            const double value = policy[static_cast<std::size_t>(action)].cast<double>();
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Evaluation values must be finite");
            }
            append_float32(packet, static_cast<float>(value));
        }
        const double value = evaluation.attr("value").cast<double>();
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Evaluation values must be finite");
        }
        append_float32(packet, static_cast<float>(value));
    }
    return py::bytes(packet);
}

py::list unpack_compact_evaluations(const py::bytes& packed_evaluations,
                                    const int action_size) {
    if (action_size <= 0) {
        throw std::invalid_argument("Compact evaluation action size is invalid");
    }
    const std::string payload = packed_evaluations;
    if (payload.size() < 2) {
        throw std::invalid_argument("Compact evaluation packet is truncated");
    }
    const std::size_t count = read_u16(payload, 0);
    if (count == 0) {
        throw std::invalid_argument("Compact evaluation packet is empty");
    }
    const std::size_t record_size = static_cast<std::size_t>(action_size + 1) * sizeof(float);
    if (payload.size() != 2 + count * record_size) {
        throw std::invalid_argument("Compact evaluation packet length is invalid");
    }
    py::list evaluations;
    std::size_t offset = 2;
    for (std::size_t index = 0; index < count; ++index) {
        py::tuple policy(action_size);
        for (int action = 0; action < action_size; ++action) {
            const float value = read_float32(payload, offset);
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Compact evaluation packet is non-finite");
            }
            policy[static_cast<std::size_t>(action)] = py::float_(value);
            offset += sizeof(float);
        }
        const float value = read_float32(payload, offset);
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Compact evaluation packet is non-finite");
        }
        offset += sizeof(float);
        evaluations.append(py::make_tuple(policy, py::float_(value)));
    }
    return evaluations;
}

struct NativeEdge {
    double prior{};
    int child{-1};
    int visit_count{};
    double value_sum{};
    int virtual_visit_count{};
    double virtual_value_sum{};
    double solved_value{std::numeric_limits<double>::quiet_NaN()};
};

struct NativeNode {
    int to_play{};
    std::uint64_t zobrist_hash{};
    bool terminal{};
    int visit_count{};
    double value_sum{};
    std::map<int, NativeEdge> children;
    bool expanded{};
    bool evaluation_in_flight{};
    double terminal_value{std::numeric_limits<double>::quiet_NaN()};
    double solved_value{std::numeric_limits<double>::quiet_NaN()};
};

struct PendingLeaf {
    int node{};
    std::vector<int> nodes;
    std::vector<std::pair<int, int>> edges;
    BoardCore board{1, 1, 0};
    bool active{true};
};

class NativeSearchTree {
public:
    NativeSearchTree(const int board_size, const int neighborhood_radius,
                     const std::uint64_t zobrist_seed,
                     const std::vector<int>& root_actions, const double c_puct,
                     const double fpu_reduction, const int virtual_loss)
        : root_board_(board_size, neighborhood_radius, zobrist_seed), c_puct_(c_puct),
          fpu_reduction_(fpu_reduction), virtual_loss_(virtual_loss) {
        if (!std::isfinite(c_puct) || c_puct <= 0.0 || !std::isfinite(fpu_reduction) ||
            fpu_reduction < 0.0 || virtual_loss <= 0) {
            throw std::invalid_argument("Invalid native MCTS configuration");
        }
        for (const int action : root_actions) {
            root_board_.apply(action);
        }
        if (root_board_.is_terminal()) {
            throw std::invalid_argument("Cannot search a terminal root state");
        }
        nodes_.push_back(NativeNode{root_board_.to_play(), root_board_.zobrist_hash(), false});
    }

    void initialize_root(const py::sequence& policy) {
        // Convert Python objects while the GIL is held, then keep the actual
        // tree expansion outside it.  The latter is a pure C++ operation and
        // can otherwise block all Python workers during root setup.
        const std::vector<double> values = extract_policy(policy);
        py::gil_scoped_release release;
        expand_node(0, root_board_, values);
    }

    py::dict select_batch(const int maximum_leaves, const int remaining_simulations) {
        if (maximum_leaves <= 0 || remaining_simulations <= 0) {
            throw std::invalid_argument("Native MCTS batch limits must be positive");
        }
        std::vector<int> leaf_ids;
        leaf_ids.reserve(static_cast<std::size_t>(maximum_leaves));
        int terminal_simulations = 0;
        {
            // Selection, including board copies and PUCT traversal, never
            // touches Python objects.  Releasing the GIL lets other workers
            // progress while this tree is busy.
            py::gil_scoped_release release;
            while (static_cast<int>(leaf_ids.size()) < maximum_leaves &&
                   static_cast<int>(leaf_ids.size()) + terminal_simulations <
                       remaining_simulations) {
                const SelectResult selection = select_one();
                if (selection.kind == SelectKind::NoProgress) {
                    break;
                }
                if (selection.kind == SelectKind::Terminal) {
                    ++terminal_simulations;
                } else {
                    leaf_ids.push_back(selection.leaf_id);
                }
            }
        }

        // Materialise the small result only after reacquiring the GIL.  Do
        // not use call_guard<gil_scoped_release> here: py::list/py::dict
        // construction must happen with the interpreter lock held.
        py::list leaves(leaf_ids.size());
        py::list legal_counts(leaf_ids.size());
        for (std::size_t index = 0; index < leaf_ids.size(); ++index) {
            leaves[index] = leaf_ids[index];
            legal_counts[index] = get_active_leaf(leaf_ids[index]).board.legal_count();
        }
        py::dict result;
        result["leaf_ids"] = leaves;
        result["leaf_legal_counts"] = legal_counts;
        result["terminal_simulations"] = terminal_simulations;
        return result;
    }

    py::dict leaf_state(const int leaf_id) const {
        const std::vector<LeafSnapshot> snapshots = snapshot_leaves(
            std::vector<int>{leaf_id});
        return materialize_leaf_state(snapshots.front());
    }

    py::list leaf_states(const py::sequence& leaf_ids) const {
        // Convert the small id sequence while holding the GIL, then copy all
        // board state in one C++ critical section.  The previous Python loop
        // called leaf_state once per leaf, repeatedly crossing the extension
        // boundary and serialising each vector independently.  This batch
        // form preserves the exact state payload while reducing lock and call
        // overhead; it does not alter tree selection or backup semantics.
        const std::vector<int> ids = extract_leaf_ids(leaf_ids);
        if (ids.empty()) {
            throw std::invalid_argument("Native MCTS leaf batch cannot be empty");
        }
        const std::vector<LeafSnapshot> snapshots = snapshot_leaves(ids);
        py::list result(snapshots.size());
        for (std::size_t index = 0; index < snapshots.size(); ++index) {
            result[index] = materialize_leaf_state(snapshots[index]);
        }
        return result;
    }

    py::bytes leaf_packed_states(const py::sequence& leaf_ids) const {
        // Serialize active leaf boards directly into the compact inference
        // protocol.  This removes the old C++ dict -> Python GameState/Move[]
        // -> compact packet round trip from every neural MCTS batch.
        const std::vector<int> ids = extract_leaf_ids(leaf_ids);
        if (ids.empty() || ids.size() > 65535) {
            throw std::invalid_argument("Native MCTS leaf batch cannot be empty");
        }
        std::string packet;
        {
            py::gil_scoped_release release;
            const int action_size = root_board_.action_size();
            const unsigned char packet_version = action_size == 256 ? 1U : 2U;
            const std::size_t history_width = packet_version == 1U ? 1U : 2U;
            const std::size_t legal_byte_count =
                static_cast<std::size_t>((action_size + 7) / 8);
            append_u16(packet, static_cast<std::uint16_t>(ids.size()));
            for (const int leaf_id : ids) {
                const BoardCore& board = get_active_leaf(leaf_id).board;
                std::string record;
                record.reserve(
                    12U + static_cast<std::size_t>(action_size) + legal_byte_count +
                    static_cast<std::size_t>(board.ply()) * history_width);
                record.push_back(static_cast<char>(packet_version));
                record.push_back(static_cast<char>(board.to_play()));
                append_u16(record, static_cast<std::uint16_t>(board.ply()));
                append_u64(record, board.zobrist_hash());
                for (const auto cell : board.cells()) {
                    record.push_back(static_cast<char>(cell));
                }
                std::string legal(legal_byte_count, '\0');
                for (int action = 0; action < action_size; ++action) {
                    if (board.is_legal(action)) {
                        const std::size_t byte = static_cast<std::size_t>(action / 8);
                        legal[byte] = static_cast<char>(
                            static_cast<unsigned char>(legal[byte]) |
                            (1U << (action % 8)));
                    }
                }
                record.append(legal);
                for (const int action : board.history()) {
                    if (packet_version == 1U) {
                        record.push_back(static_cast<char>(action));
                    } else {
                        append_u16(record, static_cast<std::uint16_t>(action));
                    }
                }
                if (record.size() > 65535) {
                    throw std::runtime_error("Native MCTS leaf record is too large");
                }
                append_u16(packet, static_cast<std::uint16_t>(record.size()));
                packet.append(record);
            }
        }
        return py::bytes(packet);
    }

    void commit(const int leaf_id, const py::sequence& policy, const double value) {
        if (!std::isfinite(value) || value < -1.0 || value > 1.0) {
            throw std::invalid_argument("Leaf value must be finite and within [-1, 1]");
        }
        const std::vector<double> values = extract_policy(policy);
        py::gil_scoped_release release;
        commit_values(leaf_id, values, value);
    }

    void commit_batch(const py::sequence& leaf_ids, const py::sequence& policies,
                      const py::sequence& values) {
        const std::vector<int> ids = extract_leaf_ids(leaf_ids);
        const py::ssize_t policy_count = py::len(policies);
        const py::ssize_t value_count = py::len(values);
        if (ids.empty() || policy_count != static_cast<py::ssize_t>(ids.size()) ||
            value_count != static_cast<py::ssize_t>(ids.size())) {
            throw std::invalid_argument(
                "Native MCTS commit batch lengths must match and be nonzero");
        }
        for (std::size_t index = 0; index < ids.size(); ++index) {
            for (std::size_t other = index + 1; other < ids.size(); ++other) {
                if (ids[index] == ids[other]) {
                    throw std::invalid_argument(
                        "Native MCTS commit batch contains duplicate leaf ids");
                }
            }
        }

        // Convert and validate all Python inputs before mutating the tree.  A
        // complete validation pass makes this operation transactional from
        // the caller's perspective: if a malformed policy is supplied, no
        // leaf is committed and the Python search can safely cancel all ids.
        std::vector<std::vector<double>> policy_values;
        policy_values.reserve(ids.size());
        std::vector<double> scalar_values;
        scalar_values.reserve(ids.size());
        for (std::size_t index = 0; index < ids.size(); ++index) {
            const py::sequence policy = policies[index].cast<py::sequence>();
            const double value = values[index].cast<double>();
            if (!std::isfinite(value) || value < -1.0 || value > 1.0) {
                throw std::invalid_argument(
                    "Leaf value must be finite and within [-1, 1]");
            }
            policy_values.push_back(extract_policy(policy));
            scalar_values.push_back(value);
        }

        {
            py::gil_scoped_release release;
            for (std::size_t index = 0; index < ids.size(); ++index) {
                validate_commit_input(ids[index], policy_values[index], scalar_values[index]);
            }
            for (std::size_t index = 0; index < ids.size(); ++index) {
                commit_values(ids[index], policy_values[index], scalar_values[index]);
            }
        }
    }

    void solve(const int leaf_id, const double value) {
        py::gil_scoped_release release;
        solve_value(leaf_id, value);
    }

    void cancel(const py::sequence& leaf_ids) {
        const std::vector<int> ids = extract_leaf_ids(leaf_ids);
        py::gil_scoped_release release;
        cancel_values(ids);
    }

    py::list export_nodes() const {
        std::vector<NativeNode> nodes;
        {
            // Copy the C++ tree while the GIL is released.  Python object
            // creation below is intentionally kept in a separate scope.
            py::gil_scoped_release release;
            nodes = nodes_;
        }
        py::list result(nodes.size());
        for (std::size_t index = 0; index < nodes.size(); ++index) {
            const NativeNode& node = nodes[index];
            py::dict item;
            item["to_play"] = node.to_play;
            item["zobrist_hash"] = node.zobrist_hash;
            item["terminal"] = node.terminal;
            item["visit_count"] = node.visit_count;
            item["value_sum"] = node.value_sum;
            item["expanded"] = node.expanded;
            item["solved_value"] = node.solved_value;
            py::dict children;
            for (const auto& [action, edge] : node.children) {
                children[py::int_(action)] = py::make_tuple(
                    edge.prior, edge.child, edge.visit_count, edge.value_sum,
                    edge.virtual_visit_count, edge.virtual_value_sum, edge.solved_value);
            }
            item["children"] = std::move(children);
            result[index] = std::move(item);
        }
        return result;
    }

    py::dict root_statistics() const {
        NativeNode root;
        {
            py::gil_scoped_release release;
            root = nodes_.front();
        }
        py::dict result;
        result["to_play"] = root.to_play;
        result["zobrist_hash"] = root.zobrist_hash;
        result["terminal"] = root.terminal;
        result["visit_count"] = root.visit_count;
        result["value_sum"] = root.value_sum;
        result["expanded"] = root.expanded;
        result["solved_value"] = root.solved_value;
        py::dict children;
        for (const auto& [action, edge] : root.children) {
            children[py::int_(action)] = py::make_tuple(
                edge.prior, edge.visit_count, edge.value_sum,
                edge.virtual_visit_count, edge.virtual_value_sum, edge.solved_value);
        }
        result["children"] = std::move(children);
        return result;
    }

private:
    struct LeafSnapshot {
        std::vector<std::uint8_t> cells;
        std::vector<bool> legal_mask;
        std::vector<int> history;
        int to_play{};
        std::uint64_t zobrist_hash{};
        bool terminal{};
    };

    std::vector<LeafSnapshot> snapshot_leaves(
        const std::vector<int>& leaf_ids) const {
        std::vector<LeafSnapshot> snapshots;
        snapshots.reserve(leaf_ids.size());
        {
            // Every operation in this scope is C++-only.  In particular,
            // get_active_leaf() only inspects native vectors and does not
            // touch Python objects, so workers can run concurrently here.
            py::gil_scoped_release release;
            for (const int leaf_id : leaf_ids) {
                const PendingLeaf& leaf = get_active_leaf(leaf_id);
                LeafSnapshot snapshot;
                snapshot.cells = leaf.board.cells();
                snapshot.legal_mask = leaf.board.legal_mask();
                snapshot.history = leaf.board.history();
                snapshot.to_play = leaf.board.to_play();
                snapshot.zobrist_hash = leaf.board.zobrist_hash();
                snapshot.terminal = leaf.board.is_terminal();
                snapshots.push_back(std::move(snapshot));
            }
        }
        return snapshots;
    }

    static py::dict materialize_leaf_state(const LeafSnapshot& snapshot) {
        py::dict state;
        state["cells"] = snapshot.cells;
        state["legal_mask"] = snapshot.legal_mask;
        state["history"] = snapshot.history;
        state["to_play"] = snapshot.to_play;
        state["zobrist_hash"] = snapshot.zobrist_hash;
        state["terminal"] = snapshot.terminal;
        return state;
    }

    // The following helpers are deliberately Python-free.  Public wrappers
    // perform Python sequence conversion first, then call them with the GIL
    // released so tree traversal/backup can overlap other Python workers.
    static std::vector<double> extract_policy(const py::sequence& policy) {
        const py::ssize_t length = py::len(policy);
        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(length));
        for (py::ssize_t index = 0; index < length; ++index) {
            values.push_back(policy[static_cast<std::size_t>(index)].cast<double>());
        }
        return values;
    }

    static std::vector<int> extract_leaf_ids(const py::sequence& leaf_ids) {
        const py::ssize_t length = py::len(leaf_ids);
        std::vector<int> ids;
        ids.reserve(static_cast<std::size_t>(length));
        for (py::ssize_t index = 0; index < length; ++index) {
            ids.push_back(leaf_ids[static_cast<std::size_t>(index)].cast<int>());
        }
        return ids;
    }

    void commit_values(const int leaf_id, const std::vector<double>& policy,
                       const double value) {
        PendingLeaf& leaf = get_active_leaf_mutable(leaf_id);
        NativeNode& node = nodes_[static_cast<std::size_t>(leaf.node)];
        if (!node.evaluation_in_flight || node.expanded) {
            throw std::runtime_error("Native MCTS leaf is not awaiting evaluation");
        }
        expand_node(leaf.node, leaf.board, policy);
        nodes_[static_cast<std::size_t>(leaf.node)].evaluation_in_flight = false;
        backup(leaf.nodes, leaf.edges, value);
        leaf.active = false;
    }

    void validate_commit_input(const int leaf_id,
                               const std::vector<double>& policy,
                               const double value) const {
        if (!std::isfinite(value) || value < -1.0 || value > 1.0) {
            throw std::invalid_argument("Leaf value must be finite and within [-1, 1]");
        }
        const PendingLeaf& leaf = get_active_leaf(leaf_id);
        const NativeNode& node = nodes_[static_cast<std::size_t>(leaf.node)];
        if (!node.evaluation_in_flight || node.expanded) {
            throw std::runtime_error("Native MCTS leaf is not awaiting evaluation");
        }
        if (static_cast<int>(policy.size()) != leaf.board.action_size()) {
            throw std::invalid_argument("Evaluation policy length must equal the action size");
        }
        const std::vector<bool> legal_mask = leaf.board.legal_mask();
        double probability_sum = 0.0;
        for (int action = 0; action < leaf.board.action_size(); ++action) {
            const double probability = policy[static_cast<std::size_t>(action)];
            if (!std::isfinite(probability) || probability < 0.0) {
                throw std::invalid_argument(
                    "Evaluation policy must be finite and nonnegative");
            }
            if (!legal_mask[static_cast<std::size_t>(action)] && probability != 0.0) {
                throw std::invalid_argument(
                    "Evaluation policy must assign zero mass to illegal actions");
            }
            if (legal_mask[static_cast<std::size_t>(action)]) {
                probability_sum += probability;
            }
        }
        if (std::abs(probability_sum - 1.0) > 1.0e-6) {
            throw std::invalid_argument(
                "Evaluation policy must sum to one over legal actions");
        }
    }

    void solve_value(const int leaf_id, const double value) {
        if (!std::isfinite(value) || value < -1.0 || value > 1.0) {
            throw std::invalid_argument("Solved leaf value must be finite and within [-1, 1]");
        }
        PendingLeaf& leaf = get_active_leaf_mutable(leaf_id);
        NativeNode& node = nodes_[static_cast<std::size_t>(leaf.node)];
        if (!node.evaluation_in_flight || node.expanded || node.terminal) {
            throw std::runtime_error("Native MCTS leaf cannot be marked solved");
        }
        node.evaluation_in_flight = false;
        node.solved_value = value;
        backup(leaf.nodes, leaf.edges, value);
        propagate_solved(leaf.nodes, leaf.edges);
        leaf.active = false;
    }

    void cancel_values(const std::vector<int>& leaf_ids) {
        for (const int leaf_id : leaf_ids) {
            PendingLeaf& leaf = get_active_leaf_mutable(leaf_id);
            nodes_[static_cast<std::size_t>(leaf.node)].evaluation_in_flight = false;
            cancel_path(leaf.edges);
            leaf.active = false;
        }
    }
    enum class SelectKind { NoProgress, Terminal, Leaf };
    struct SelectResult {
        SelectKind kind{};
        int leaf_id{-1};
    };

    BoardCore root_board_;
    double c_puct_;
    double fpu_reduction_;
    int virtual_loss_;
    std::vector<NativeNode> nodes_;
    std::vector<std::optional<PendingLeaf>> leaves_;

    SelectResult select_one() {
        BoardCore board = root_board_;
        int node_id = 0;
        std::vector<int> nodes{0};
        std::vector<std::pair<int, int>> edges;
        while (true) {
            const std::optional<int> action = select_edge(node_id);
            if (!action.has_value()) {
                cancel_path(edges);
                return {SelectKind::NoProgress, -1};
            }
            NativeEdge& edge = nodes_[static_cast<std::size_t>(node_id)].children[*action];
            reserve(edge);
            edges.emplace_back(node_id, *action);
            board.apply(*action);
            int child_id = edge.child;
            if (child_id == -1) {
                child_id = static_cast<int>(nodes_.size());
                nodes_.push_back(NativeNode{board.to_play(), board.zobrist_hash(), board.is_terminal()});
                nodes_[static_cast<std::size_t>(node_id)].children[*action].child = child_id;
            }
            node_id = child_id;
            nodes.push_back(node_id);
            NativeNode& node = nodes_[static_cast<std::size_t>(node_id)];
            if (node.terminal) {
                if (std::isnan(node.terminal_value)) {
                    const NativeScore score = board.score();
                    if (score.black_score == score.white_score) {
                        node.terminal_value = 0.0;
                    } else {
                        const int winner = score.black_score > score.white_score ? kBlack : kWhite;
                        node.terminal_value = winner == board.to_play() ? 1.0 : -1.0;
                    }
                }
                node.solved_value = node.terminal_value;
                backup(nodes, edges, node.terminal_value);
                propagate_solved(nodes, edges);
                return {SelectKind::Terminal, -1};
            }
            if (!std::isnan(node.solved_value)) {
                backup(nodes, edges, node.solved_value);
                propagate_solved(nodes, edges);
                return {SelectKind::Terminal, -1};
            }
            if (!node.expanded) {
                if (node.evaluation_in_flight) {
                    cancel_path(edges);
                    return {SelectKind::NoProgress, -1};
                }
                node.evaluation_in_flight = true;
                const int leaf_id = static_cast<int>(leaves_.size());
                leaves_.emplace_back(PendingLeaf{node_id, std::move(nodes), std::move(edges),
                                                 std::move(board), true});
                return {SelectKind::Leaf, leaf_id};
            }
        }
    }

    std::optional<int> select_edge(const int node_id) const {
        const NativeNode& node = nodes_[static_cast<std::size_t>(node_id)];
        if (!node.expanded || node.children.empty()) {
            throw std::runtime_error("Only expanded native nodes can select edges");
        }
        int virtual_visits = 0;
        for (const auto& [_, edge] : node.children) {
            virtual_visits += edge.virtual_visit_count;
        }
        const int parent_visits = std::max(1, node.visit_count + virtual_visits);
        const double scale = c_puct_ * std::sqrt(static_cast<double>(parent_visits));
        double selected_score = -std::numeric_limits<double>::infinity();
        std::optional<int> selected;
        for (const auto& [action, edge] : node.children) {
            if (!std::isnan(node.solved_value) && edge.solved_value != node.solved_value) {
                continue;
            }
            if (edge.child != -1) {
                const NativeNode& child = nodes_[static_cast<std::size_t>(edge.child)];
                if (child.evaluation_in_flight && !child.expanded) {
                    continue;
                }
            }
            const int effective_visits = edge.visit_count + edge.virtual_visit_count;
            const double effective_value = !std::isnan(edge.solved_value) &&
                                                   edge.virtual_visit_count == 0
                ? edge.solved_value
                : (effective_visits == 0
                       ? node_mean_value(node) - fpu_reduction_
                       : (edge.value_sum - edge.virtual_value_sum) /
                             static_cast<double>(effective_visits));
            const double score = effective_value +
                scale * edge.prior / static_cast<double>(1 + effective_visits);
            if (score > selected_score) {
                selected = action;
                selected_score = score;
            }
        }
        return selected;
    }

    void expand_node(const int node_id, const BoardCore& board,
                     const std::vector<double>& policy) {
        if (static_cast<int>(policy.size()) != board.action_size()) {
            throw std::invalid_argument("Evaluation policy length must equal the action size");
        }
        NativeNode& node = nodes_[static_cast<std::size_t>(node_id)];
        if (node.terminal || node.expanded) {
            throw std::runtime_error("Native MCTS node cannot be expanded");
        }
        const std::vector<bool> legal_mask = board.legal_mask();
        double probability_sum = 0.0;
        for (int action = 0; action < board.action_size(); ++action) {
            const double probability = policy[static_cast<std::size_t>(action)];
            if (!std::isfinite(probability) || probability < 0.0) {
                throw std::invalid_argument("Evaluation policy must be finite and nonnegative");
            }
            const bool legal = legal_mask[static_cast<std::size_t>(action)];
            if (!legal && probability != 0.0) {
                throw std::invalid_argument("Evaluation policy must assign zero mass to illegal actions");
            }
            if (legal) {
                node.children.emplace(action, NativeEdge{probability});
                probability_sum += probability;
            }
        }
        if (node.children.empty() || std::abs(probability_sum - 1.0) > 1.0e-6) {
            throw std::invalid_argument("Evaluation policy must sum to one over legal actions");
        }
        node.expanded = true;
    }

    void reserve(NativeEdge& edge) const {
        ++edge.virtual_visit_count;
        edge.virtual_value_sum += static_cast<double>(virtual_loss_);
    }

    void release(NativeEdge& edge) const {
        if (edge.virtual_visit_count <= 0) {
            throw std::runtime_error("Native MCTS virtual-loss accounting is inconsistent");
        }
        --edge.virtual_visit_count;
        edge.virtual_value_sum -= static_cast<double>(virtual_loss_);
        if (edge.virtual_visit_count == 0) {
            edge.virtual_value_sum = 0.0;
        }
    }

    void backup(const std::vector<int>& nodes, const std::vector<std::pair<int, int>>& edges,
                double value) {
        if (nodes.size() != edges.size() + 1) {
            throw std::runtime_error("Native MCTS backup path is inconsistent");
        }
        NativeNode& leaf = nodes_[static_cast<std::size_t>(nodes.back())];
        ++leaf.visit_count;
        leaf.value_sum += value;
        int current_player = leaf.to_play;
        for (int index = static_cast<int>(edges.size()) - 1; index >= 0; --index) {
            const auto [parent_id, action] = edges[static_cast<std::size_t>(index)];
            NativeNode& parent = nodes_[static_cast<std::size_t>(parent_id)];
            NativeEdge& edge = parent.children[action];
            release(edge);
            if (parent.to_play != current_player) {
                value = -value;
            }
            ++edge.visit_count;
            edge.value_sum += value;
            ++parent.visit_count;
            parent.value_sum += value;
            current_player = parent.to_play;
        }
    }

    void propagate_solved(const std::vector<int>& nodes,
                          const std::vector<std::pair<int, int>>& edges) {
        for (int index = static_cast<int>(edges.size()) - 1; index >= 0; --index) {
            NativeNode& child = nodes_[static_cast<std::size_t>(nodes[index + 1])];
            if (std::isnan(child.solved_value)) {
                break;
            }
            const auto [parent_id, action] = edges[static_cast<std::size_t>(index)];
            NativeNode& parent = nodes_[static_cast<std::size_t>(parent_id)];
            parent.children[action].solved_value = -child.solved_value;
            bool all_solved = true;
            double best = -1.0;
            for (const auto& [_, edge] : parent.children) {
                if (std::isnan(edge.solved_value)) {
                    all_solved = false;
                } else {
                    best = std::max(best, edge.solved_value);
                }
            }
            if (best == 1.0 || all_solved) {
                parent.solved_value = best;
            } else {
                break;
            }
        }
    }

    void cancel_path(const std::vector<std::pair<int, int>>& edges) {
        for (const auto [node_id, action] : edges) {
            release(nodes_[static_cast<std::size_t>(node_id)].children[action]);
        }
    }

    static double node_mean_value(const NativeNode& node) {
        if (!std::isnan(node.solved_value)) {
            return node.solved_value;
        }
        return node.visit_count == 0 ? 0.0 : node.value_sum / node.visit_count;
    }

    const PendingLeaf& get_active_leaf(const int leaf_id) const {
        if (leaf_id < 0 || leaf_id >= static_cast<int>(leaves_.size()) ||
            !leaves_[static_cast<std::size_t>(leaf_id)].has_value() ||
            !leaves_[static_cast<std::size_t>(leaf_id)]->active) {
            throw std::out_of_range("Native MCTS leaf is unknown or inactive");
        }
        return *leaves_[static_cast<std::size_t>(leaf_id)];
    }

    PendingLeaf& get_active_leaf_mutable(const int leaf_id) {
        return const_cast<PendingLeaf&>(std::as_const(*this).get_active_leaf(leaf_id));
    }
};

NativeScore score_cells(const py::sequence& values, const int board_size) {
    return BoardCore::score_cells(extract_cells(values, board_size * board_size), board_size);
}

// ---------------------------------------------------------------------------
// Handcrafted feature extraction for the mixed-input policy/value network.
// This path deliberately receives only compact cell arrays.  It computes the
// deterministic D/G/S/B values and maps in C++, then returns contiguous
// float32 NumPy arrays.  The Python caller only performs the one host-to-device
// transfer and subtracts no per-cell Python loops.
// ---------------------------------------------------------------------------

constexpr int kHybridScalarCount = 19;  // D, G(3x4), S(3), B(3); I is terminal-only.

struct HybridFeatureValues {
    std::vector<float> scalars;
    std::vector<float> divergence_density;
    std::vector<float> divergence_raw;
    std::vector<float> divergence;
    std::vector<float> sparse;
    std::vector<float> dense;
};

struct HybridFeatureBatch {
    std::vector<float> scalars;
    std::vector<float> scalars_delta;
    std::vector<float> divergence;
    std::vector<float> divergence_delta;
    std::vector<float> sparse;
    std::vector<float> sparse_delta;
    std::vector<float> dense;
    std::vector<float> dense_delta;
};

std::vector<std::uint8_t> extract_cell_vector(const py::handle& value,
                                              const int action_size) {
    return extract_cells(value.cast<py::sequence>(), action_size);
}

std::vector<std::vector<std::uint8_t>> extract_cell_batch(
    const py::sequence& values, const int action_size) {
    std::vector<std::vector<std::uint8_t>> result;
    result.reserve(static_cast<std::size_t>(py::len(values)));
    for (const py::handle value : values) {
        result.push_back(extract_cell_vector(value, action_size));
    }
    if (result.empty()) {
        throw std::invalid_argument("Hybrid feature batch must not be empty");
    }
    return result;
}

float clamp_unit(const float value) {
    return std::max(0.0F, std::min(1.0F, value));
}

float gaussian_weight(const int row_delta, const int column_delta,
                      const int radius, const float sigma) {
    const int squared = row_delta * row_delta + column_delta * column_delta;
    if (squared > radius * radius) return 0.0F;
    return std::exp(-static_cast<float>(squared) / (2.0F * sigma * sigma));
}

std::vector<float> hybrid_divergence(
    const std::vector<std::uint8_t>& cells, const int board_size,
    std::vector<float>* filtered_density_output = nullptr) {
    const int area = board_size * board_size;
    constexpr int radius = 3;
    constexpr float sigma = 2.0F;
    constexpr float numerator_scale = 2.0F;
    constexpr float density_epsilon = 0.1F;
    std::vector<float> raw(static_cast<std::size_t>(area), 0.0F);
    std::vector<float> available(static_cast<std::size_t>(area), 0.0F);
    float kernel_mass = 0.0F;
    for (int dr = -radius; dr <= radius; ++dr)
        for (int dc = -radius; dc <= radius; ++dc)
            kernel_mass += gaussian_weight(dr, dc, radius, sigma);

    // Truncated Gaussian density and available-mass correction.
    for (int row = 0; row < board_size; ++row) {
        for (int column = 0; column < board_size; ++column) {
            float density = 0.0F;
            float mass = 0.0F;
            for (int dr = -radius; dr <= radius; ++dr) {
                const int source_row = row + dr;
                if (source_row < 0 || source_row >= board_size) continue;
                for (int dc = -radius; dc <= radius; ++dc) {
                    const int source_column = column + dc;
                    if (source_column < 0 || source_column >= board_size) continue;
                    const float weight = gaussian_weight(dr, dc, radius, sigma);
                    density += weight * (cells[static_cast<std::size_t>(source_row * board_size + source_column)] != 0);
                    mass += weight;
                }
            }
            raw[static_cast<std::size_t>(row * board_size + column)] = density;
            available[static_cast<std::size_t>(row * board_size + column)] = mass;
        }
    }

    // Apply the same separable period-2/3 suppression filter as the settled
    // Python analyser: [1,3,4,3,1] outer product divided by 144.
    const std::array<float, 5> period = {1.0F, 3.0F, 4.0F, 3.0F, 1.0F};
    std::vector<float> density_filtered(static_cast<std::size_t>(area), 0.0F);
    std::vector<float> available_filtered(static_cast<std::size_t>(area), 0.0F);
    for (int row = 0; row < board_size; ++row) {
        for (int column = 0; column < board_size; ++column) {
            float density = 0.0F;
            float mass = 0.0F;
            for (int dr = -2; dr <= 2; ++dr) {
                const int source_row = row + dr;
                if (source_row < 0 || source_row >= board_size) continue;
                for (int dc = -2; dc <= 2; ++dc) {
                    const int source_column = column + dc;
                    if (source_column < 0 || source_column >= board_size) continue;
                    const float weight = period[static_cast<std::size_t>(dr + 2)] *
                        period[static_cast<std::size_t>(dc + 2)] / 144.0F;
                    const std::size_t index = static_cast<std::size_t>(source_row * board_size + source_column);
                    density += raw[index] * weight;
                    mass += available[index] * weight;
                }
            }
            const std::size_t index = static_cast<std::size_t>(row * board_size + column);
            density_filtered[index] = density;
            available_filtered[index] = mass;
        }
    }

    std::vector<float> result(static_cast<std::size_t>(area), 0.0F);
    for (int index = 0; index < area; ++index) {
        const float mass = available_filtered[static_cast<std::size_t>(index)];
        const float density = density_filtered[static_cast<std::size_t>(index)];
        const float contribution = density <= density_epsilon
            ? numerator_scale * mass / kernel_mass
            : numerator_scale * (mass / kernel_mass) / density;
        result[static_cast<std::size_t>(index)] = contribution;
    }
    if (filtered_density_output != nullptr) {
        *filtered_density_output = std::move(density_filtered);
    }
    return result;
}

float hybrid_density_impact(const int board_size, const int action_row,
                            const int action_column, const int row,
                            const int column) {
    constexpr int radius = 3;
    constexpr float sigma = 2.0F;
    const std::array<float, 5> period = {1.0F, 3.0F, 4.0F, 3.0F, 1.0F};
    float impact = 0.0F;
    for (int filter_row = -2; filter_row <= 2; ++filter_row) {
        const int raw_row = row + filter_row;
        if (raw_row < 0 || raw_row >= board_size) continue;
        for (int filter_column = -2; filter_column <= 2; ++filter_column) {
            const int raw_column = column + filter_column;
            if (raw_column < 0 || raw_column >= board_size) continue;
            const float gaussian = gaussian_weight(
                action_row - raw_row, action_column - raw_column, radius, sigma);
            if (gaussian == 0.0F) continue;
            impact += gaussian *
                period[static_cast<std::size_t>(filter_row + 2)] *
                period[static_cast<std::size_t>(filter_column + 2)] / 144.0F;
        }
    }
    return impact;
}

std::pair<float, float> hybrid_filtered_density_at(
    const std::vector<std::uint8_t>& cells, const int board_size,
    const int row, const int column) {
    constexpr int radius = 3;
    constexpr float sigma = 2.0F;
    const std::array<float, 5> period = {1.0F, 3.0F, 4.0F, 3.0F, 1.0F};
    float density_filtered = 0.0F;
    float available_filtered = 0.0F;
    for (int filter_row = -2; filter_row <= 2; ++filter_row) {
        const int raw_row = row + filter_row;
        if (raw_row < 0 || raw_row >= board_size) continue;
        for (int filter_column = -2; filter_column <= 2; ++filter_column) {
            const int raw_column = column + filter_column;
            if (raw_column < 0 || raw_column >= board_size) continue;
            float raw_density = 0.0F;
            float raw_available = 0.0F;
            for (int dr = -radius; dr <= radius; ++dr) {
                const int source_row = raw_row + dr;
                if (source_row < 0 || source_row >= board_size) continue;
                for (int dc = -radius; dc <= radius; ++dc) {
                    const int source_column = raw_column + dc;
                    if (source_column < 0 || source_column >= board_size) continue;
                    const float gaussian = gaussian_weight(dr, dc, radius, sigma);
                    raw_density += gaussian *
                        (cells[static_cast<std::size_t>(source_row * board_size + source_column)] != 0);
                    raw_available += gaussian;
                }
            }
            const float filter_weight =
                period[static_cast<std::size_t>(filter_row + 2)] *
                period[static_cast<std::size_t>(filter_column + 2)] / 144.0F;
            density_filtered += raw_density * filter_weight;
            available_filtered += raw_available * filter_weight;
        }
    }
    return {density_filtered, available_filtered};
}

float hybrid_divergence_raw_at(const std::vector<std::uint8_t>& cells,
                               const int board_size, const int row,
                               const int column) {
    constexpr int radius = 3;
    constexpr float sigma = 2.0F;
    constexpr float numerator_scale = 2.0F;
    constexpr float density_epsilon = 0.1F;
    float kernel_mass = 0.0F;
    for (int dr = -radius; dr <= radius; ++dr)
        for (int dc = -radius; dc <= radius; ++dc)
            kernel_mass += gaussian_weight(dr, dc, radius, sigma);
    const auto [density, available] = hybrid_filtered_density_at(
        cells, board_size, row, column);
    return density <= density_epsilon
        ? numerator_scale * available / kernel_mass
        : numerator_scale * (available / kernel_mass) / density;
}

float occupancy_at(const std::vector<std::uint8_t>& cells, const int index,
                   const int player) {
    const auto cell = cells[static_cast<std::size_t>(index)];
    return (player == 0 ? cell != 0 : cell == player) ? 1.0F : 0.0F;
}

std::vector<float> hybrid_grid_local(const std::vector<std::uint8_t>& cells,
                                     const int board_size, const int player,
                                     const int window, const int period) {
    const int side = board_size - window + 1;
    std::vector<float> result(static_cast<std::size_t>(side * side), 0.0F);
    const int phase_count = period * period;
    for (int row = 0; row < side; ++row) {
        for (int column = 0; column < side; ++column) {
            int count = 0;
            for (int dr = 0; dr < window; ++dr)
                for (int dc = 0; dc < window; ++dc)
                    count += occupancy_at(cells, (row + dr) * board_size + column + dc, player) > 0.0F;
            const float denominator = static_cast<float>(std::max(1, count));
            float value = 0.0F;
            for (int origin_row = 0; origin_row < period; ++origin_row) {
                for (int origin_column = 0; origin_column < period; ++origin_column) {
                    int matched = 0;
                    int lattice_count = 0;
                    for (int dr = 0; dr < window; ++dr) {
                        for (int dc = 0; dc < window; ++dc) {
                            const bool active = ((dr - origin_row) % period + period) % period == 0 &&
                                ((dc - origin_column) % period + period) % period == 0;
                            if (!active) continue;
                            ++lattice_count;
                            matched += occupancy_at(cells, (row + dr) * board_size + column + dc, player) > 0.0F;
                        }
                    }
                    value += static_cast<float>(matched * matched) /
                        static_cast<float>(std::max(1, lattice_count)) / denominator;
                }
            }
            result[static_cast<std::size_t>(row * side + column)] = value;
        }
    }
    static_cast<void>(phase_count);
    return result;
}

float hybrid_grid_local_at(const std::vector<std::uint8_t>& cells,
                           const int board_size, const int player,
                           const int window, const int period,
                           const int row, const int column) {
    int count = 0;
    for (int dr = 0; dr < window; ++dr)
        for (int dc = 0; dc < window; ++dc)
            count += occupancy_at(
                cells, (row + dr) * board_size + column + dc, player) > 0.0F;
    const float denominator = static_cast<float>(std::max(1, count));
    float value = 0.0F;
    for (int origin_row = 0; origin_row < period; ++origin_row) {
        for (int origin_column = 0; origin_column < period; ++origin_column) {
            int matched = 0;
            int lattice_count = 0;
            for (int dr = 0; dr < window; ++dr) {
                for (int dc = 0; dc < window; ++dc) {
                    const bool active =
                        ((dr - origin_row) % period + period) % period == 0 &&
                        ((dc - origin_column) % period + period) % period == 0;
                    if (!active) continue;
                    ++lattice_count;
                    matched += occupancy_at(
                        cells, (row + dr) * board_size + column + dc, player) > 0.0F;
                }
            }
            value += static_cast<float>(matched * matched) /
                static_cast<float>(std::max(1, lattice_count)) / denominator;
        }
    }
    return clamp_unit(value);
}

float hybrid_grid_global(const std::vector<std::uint8_t>& cells,
                         const int board_size, const int player,
                         const int period) {
    const int area = board_size * board_size;
    int count = 0;
    for (const auto cell : cells) count += player == 0 ? cell != 0 : cell == player;
    if (count == 0) return 0.0F;
    float value = 0.0F;
    for (int origin_row = 0; origin_row < period; ++origin_row) {
        for (int origin_column = 0; origin_column < period; ++origin_column) {
            int matched = 0;
            int lattice_count = 0;
            for (int row = 0; row < board_size; ++row) {
                for (int column = 0; column < board_size; ++column) {
                    const bool active = ((row - origin_row) % period + period) % period == 0 &&
                        ((column - origin_column) % period + period) % period == 0;
                    if (!active) continue;
                    ++lattice_count;
                    matched += occupancy_at(cells, row * board_size + column, player) > 0.0F;
                }
            }
            value += static_cast<float>(matched * matched) /
                static_cast<float>(std::max(1, lattice_count)) / static_cast<float>(count);
        }
    }
    static_cast<void>(area);
    return value;
}

float hybrid_symmetry(const std::vector<std::uint8_t>& cells,
                      const int board_size, const int player) {
    int count = 0;
    for (const auto cell : cells) count += player == 0 ? cell != 0 : cell == player;
    if (count == 0) return 0.0F;
    long long overlap = 0;
    for (int row = 0; row < board_size; ++row) {
        for (int column = 0; column < board_size; ++column) {
            if (occupancy_at(cells, row * board_size + column, player) == 0.0F) continue;
            const std::array<std::pair<int, int>, 5> mapped = {{
                {board_size - 1 - row, board_size - 1 - column},
                {row, board_size - 1 - column},
                {board_size - 1 - row, column},
                {column, row},
                {board_size - 1 - column, board_size - 1 - row},
            }};
            for (const auto [mapped_row, mapped_column] : mapped)
                overlap += occupancy_at(cells, mapped_row * board_size + mapped_column, player) > 0.0F;
        }
    }
    return static_cast<float>(overlap * overlap) /
        (25.0F * static_cast<float>(count) * static_cast<float>(count));
}

float hybrid_edge(const std::vector<std::uint8_t>& cells,
                  const int board_size, const int player) {
    int count = 0;
    for (int row = 0; row < board_size; ++row) {
        for (int column = 0; column < board_size; ++column) {
            if ((row == 0 || row == board_size - 1 || column == 0 || column == board_size - 1) &&
                occupancy_at(cells, row * board_size + column, player) > 0.0F) ++count;
        }
    }
    if (count == 0) return 0.0F;
    return clamp_unit(1.0F - std::pow(1.2F, -static_cast<float>(count)));
}

void refresh_hybrid_scalars(HybridFeatureValues& result,
                            const std::vector<std::uint8_t>& cells,
                            const int board_size) {
    const int area = board_size * board_size;
    result.scalars.assign(kHybridScalarCount, 0.0F);
    const std::array<int, 3> players = {0, kBlack, kWhite};
    int occupied = 0;
    for (const auto cell : cells) occupied += cell != 0;
    const float occupied_denominator = static_cast<float>(std::max(1, occupied));
    float divergence_scalar = 0.0F;
    for (int index = 0; index < area; ++index)
        if (cells[static_cast<std::size_t>(index)] != 0)
            divergence_scalar += result.divergence_raw[static_cast<std::size_t>(index)];
    result.scalars[0] = clamp_unit((divergence_scalar / occupied_denominator) /
                                   (1.0F + divergence_scalar / occupied_denominator));

    for (int group = 0; group < 3; ++group) {
        const int player = players[static_cast<std::size_t>(group)];
        const int sparse_side = board_size - 5;
        const int dense_side = board_size - 3;
        const auto sparse_begin = result.sparse.begin() +
            static_cast<std::ptrdiff_t>(group * sparse_side * sparse_side);
        const auto dense_begin = result.dense.begin() +
            static_cast<std::ptrdiff_t>(group * dense_side * dense_side);
        const float sparse_mean = std::accumulate(
            sparse_begin, sparse_begin + sparse_side * sparse_side, 0.0F) /
            static_cast<float>(sparse_side * sparse_side);
        const float dense_mean = std::accumulate(
            dense_begin, dense_begin + dense_side * dense_side, 0.0F) /
            static_cast<float>(dense_side * dense_side);
        const int offset = 1 + group * 4;
        result.scalars[static_cast<std::size_t>(offset)] = clamp_unit(hybrid_grid_global(cells, board_size, player, 3));
        result.scalars[static_cast<std::size_t>(offset + 1)] = clamp_unit(sparse_mean);
        result.scalars[static_cast<std::size_t>(offset + 2)] = clamp_unit(hybrid_grid_global(cells, board_size, player, 2));
        result.scalars[static_cast<std::size_t>(offset + 3)] = clamp_unit(dense_mean);
        result.scalars[static_cast<std::size_t>(13 + group)] = clamp_unit(hybrid_symmetry(cells, board_size, player));
        result.scalars[static_cast<std::size_t>(16 + group)] = hybrid_edge(cells, board_size, player);
    }
}

HybridFeatureValues compute_hybrid_values(const std::vector<std::uint8_t>& cells,
                                           const int board_size) {
    HybridFeatureValues result;
    result.divergence_raw = hybrid_divergence(
        cells, board_size, &result.divergence_density);
    result.divergence = result.divergence_raw;
    result.sparse.assign(static_cast<std::size_t>(3 * (board_size - 5) * (board_size - 5)), 0.0F);
    result.dense.assign(static_cast<std::size_t>(3 * (board_size - 3) * (board_size - 3)), 0.0F);
    const std::array<int, 3> players = {0, kBlack, kWhite};
    for (int group = 0; group < 3; ++group) {
        const int player = players[static_cast<std::size_t>(group)];
        const int sparse_side = board_size - 5;
        const int dense_side = board_size - 3;
        const auto sparse = hybrid_grid_local(cells, board_size, player, 6, 3);
        const auto dense = hybrid_grid_local(cells, board_size, player, 4, 2);
        std::copy(sparse.begin(), sparse.end(), result.sparse.begin() +
            static_cast<std::ptrdiff_t>(group * sparse_side * sparse_side));
        std::copy(dense.begin(), dense.end(), result.dense.begin() +
            static_cast<std::ptrdiff_t>(group * dense_side * dense_side));
    }
    for (float& value : result.divergence)
        value = clamp_unit(value / (1.0F + value));
    refresh_hybrid_scalars(result, cells, board_size);
    return result;
}

HybridFeatureValues compute_hybrid_transition(
    const HybridFeatureValues& before,
    const std::vector<std::uint8_t>& previous,
    const std::vector<std::uint8_t>& current,
    const int board_size) {
    int changed_action = -1;
    for (int action = 0; action < board_size * board_size; ++action) {
        if (previous[static_cast<std::size_t>(action)] ==
            current[static_cast<std::size_t>(action)]) continue;
        if (changed_action >= 0 || previous[static_cast<std::size_t>(action)] != 0 ||
            current[static_cast<std::size_t>(action)] == 0) {
            return compute_hybrid_values(current, board_size);
        }
        changed_action = action;
    }
    if (changed_action < 0) return before;

    HybridFeatureValues result = before;
    const int action_row = changed_action / board_size;
    const int action_column = changed_action % board_size;
    constexpr int divergence_radius = 3;
    constexpr float divergence_sigma = 2.0F;
    constexpr float numerator_scale = 2.0F;
    constexpr float density_epsilon = 0.1F;
    float kernel_mass = 0.0F;
    for (int dr = -divergence_radius; dr <= divergence_radius; ++dr)
        for (int dc = -divergence_radius; dc <= divergence_radius; ++dc)
            kernel_mass += gaussian_weight(
                dr, dc, divergence_radius, divergence_sigma);
    // Gaussian radius 3 followed by the radius-2 suppression filter means a
    // single stone can only alter divergence outputs within radius 5.
    for (int row = std::max(0, action_row - 5);
         row <= std::min(board_size - 1, action_row + 5); ++row) {
        for (int column = std::max(0, action_column - 5);
             column <= std::min(board_size - 1, action_column + 5); ++column) {
            const std::size_t index = static_cast<std::size_t>(row * board_size + column);
            const float density_before = before.divergence_density[index];
            const float raw_before = before.divergence_raw[index];
            const float available = density_before <= density_epsilon
                ? raw_before * kernel_mass / numerator_scale
                : raw_before * kernel_mass * density_before / numerator_scale;
            const float density = density_before + hybrid_density_impact(
                board_size, action_row, action_column, row, column);
            const float raw = density <= density_epsilon
                ? numerator_scale * available / kernel_mass
                : numerator_scale * (available / kernel_mass) / density;
            result.divergence_density[index] = density;
            result.divergence_raw[index] = raw;
            result.divergence[index] = clamp_unit(raw / (1.0F + raw));
        }
    }

    const int placed_player = current[static_cast<std::size_t>(changed_action)];
    const std::array<int, 2> groups = {0, placed_player};
    for (const int group : groups) {
        const int player = group == 0 ? 0 : placed_player;
        for (const auto [window, period] :
             std::array<std::pair<int, int>, 2>{{{6, 3}, {4, 2}}}) {
            const int side = board_size - window + 1;
            auto& target = window == 6 ? result.sparse : result.dense;
            const int row_min = std::max(0, action_row - window + 1);
            const int row_max = std::min(action_row, side - 1);
            const int column_min = std::max(0, action_column - window + 1);
            const int column_max = std::min(action_column, side - 1);
            for (int row = row_min; row <= row_max; ++row) {
                for (int column = column_min; column <= column_max; ++column) {
                    const std::size_t index = static_cast<std::size_t>(
                        group * side * side + row * side + column);
                    target[index] = hybrid_grid_local_at(
                        current, board_size, player, window, period, row, column);
                }
            }
        }
    }
    refresh_hybrid_scalars(result, current, board_size);
    return result;
}

struct HybridFeatureCacheEntry {
    std::vector<std::uint8_t> cells;
    HybridFeatureValues values;
};

struct HybridFeatureCache {
    std::unordered_map<std::uint64_t, HybridFeatureCacheEntry> entries;
    std::deque<std::uint64_t> insertion_order;
};

HybridFeatureCache& hybrid_feature_cache() {
    thread_local HybridFeatureCache cache;
    return cache;
}

std::uint64_t hybrid_cells_hash(const std::vector<std::uint8_t>& cells,
                                const int board_size) {
    std::uint64_t hash = 1469598103934665603ULL ^
        static_cast<std::uint64_t>(board_size);
    for (const auto cell : cells) {
        hash ^= static_cast<std::uint64_t>(cell + 1U);
        hash *= 1099511628211ULL;
    }
    return hash;
}

HybridFeatureValues cached_hybrid_values(
    const std::vector<std::uint8_t>& cells, const int board_size) {
    constexpr std::size_t cache_limit = 1024;
    HybridFeatureCache& cache = hybrid_feature_cache();
    const std::uint64_t key = hybrid_cells_hash(cells, board_size);
    const auto found = cache.entries.find(key);
    if (found != cache.entries.end() && found->second.cells == cells) {
        return found->second.values;
    }
    HybridFeatureValues values = compute_hybrid_values(cells, board_size);
    if (cache.entries.size() >= cache_limit && !cache.insertion_order.empty()) {
        cache.entries.erase(cache.insertion_order.front());
        cache.insertion_order.pop_front();
    }
    cache.entries[key] = HybridFeatureCacheEntry{cells, values};
    cache.insertion_order.push_back(key);
    return values;
}

void cache_hybrid_values(const std::vector<std::uint8_t>& cells,
                         const int board_size,
                         const HybridFeatureValues& values) {
    constexpr std::size_t cache_limit = 1024;
    HybridFeatureCache& cache = hybrid_feature_cache();
    const std::uint64_t key = hybrid_cells_hash(cells, board_size);
    const auto found = cache.entries.find(key);
    if (found != cache.entries.end() && found->second.cells == cells) {
        found->second.values = values;
        return;
    }
    if (cache.entries.size() >= cache_limit && !cache.insertion_order.empty()) {
        cache.entries.erase(cache.insertion_order.front());
        cache.insertion_order.pop_front();
    }
    cache.entries[key] = HybridFeatureCacheEntry{cells, values};
    cache.insertion_order.push_back(key);
}

py::array_t<float> hybrid_array(const std::vector<float>& values,
                                const std::vector<py::ssize_t>& shape) {
    py::array_t<float> output(shape);
    std::copy(values.begin(), values.end(), output.mutable_data());
    return output;
}

py::dict compute_handcrafted_features(const py::sequence& current_values,
                                      const py::sequence& previous_values,
                                      const int board_size) {
    if (board_size <= 0) throw std::invalid_argument("Board size must be positive");
    const int action_size = board_size * board_size;
    const auto current = extract_cell_batch(current_values, action_size);
    const auto previous = extract_cell_batch(previous_values, action_size);
    if (current.size() != previous.size())
        throw std::invalid_argument("Current and previous feature batches must have equal size");
    HybridFeatureBatch batch;
    const std::size_t count = current.size();
    const std::size_t sparse_area = static_cast<std::size_t>(3 * (board_size - 5) * (board_size - 5));
    const std::size_t dense_area = static_cast<std::size_t>(3 * (board_size - 3) * (board_size - 3));
    batch.scalars.reserve(count * kHybridScalarCount);
    batch.scalars_delta.reserve(count * kHybridScalarCount);
    batch.divergence.reserve(count * static_cast<std::size_t>(action_size));
    batch.divergence_delta.reserve(count * static_cast<std::size_t>(action_size));
    batch.sparse.reserve(count * sparse_area);
    batch.sparse_delta.reserve(count * sparse_area);
    batch.dense.reserve(count * dense_area);
    batch.dense_delta.reserve(count * dense_area);
    // All Python objects have been copied above. Release the GIL while the
    // deterministic convolution/grid kernels run so inference batching and
    // worker message handling are not blocked by feature preparation.
    {
        py::gil_scoped_release release;
        for (std::size_t index = 0; index < count; ++index) {
            const auto before = cached_hybrid_values(previous[index], board_size);
            const auto now = compute_hybrid_transition(
                before, previous[index], current[index], board_size);
            cache_hybrid_values(current[index], board_size, now);
            for (int feature = 0; feature < kHybridScalarCount; ++feature) {
                batch.scalars.push_back(now.scalars[static_cast<std::size_t>(feature)]);
                batch.scalars_delta.push_back(now.scalars[static_cast<std::size_t>(feature)] - before.scalars[static_cast<std::size_t>(feature)]);
            }
            for (int cell = 0; cell < action_size; ++cell) {
                batch.divergence.push_back(now.divergence[static_cast<std::size_t>(cell)]);
                batch.divergence_delta.push_back(now.divergence[static_cast<std::size_t>(cell)] - before.divergence[static_cast<std::size_t>(cell)]);
            }
            for (std::size_t cell = 0; cell < sparse_area; ++cell) {
                batch.sparse.push_back(now.sparse[cell]);
                batch.sparse_delta.push_back(now.sparse[cell] - before.sparse[cell]);
            }
            for (std::size_t cell = 0; cell < dense_area; ++cell) {
                batch.dense.push_back(now.dense[cell]);
                batch.dense_delta.push_back(now.dense[cell] - before.dense[cell]);
            }
        }
    }
    py::dict output;
    output["scalars"] = hybrid_array(batch.scalars, {static_cast<py::ssize_t>(count), kHybridScalarCount});
    output["scalars_delta"] = hybrid_array(batch.scalars_delta, {static_cast<py::ssize_t>(count), kHybridScalarCount});
    output["divergence"] = hybrid_array(batch.divergence, {static_cast<py::ssize_t>(count), 1, board_size, board_size});
    output["divergence_delta"] = hybrid_array(batch.divergence_delta, {static_cast<py::ssize_t>(count), 1, board_size, board_size});
    output["sparse"] = hybrid_array(batch.sparse, {static_cast<py::ssize_t>(count), 3, board_size - 5, board_size - 5});
    output["sparse_delta"] = hybrid_array(batch.sparse_delta, {static_cast<py::ssize_t>(count), 3, board_size - 5, board_size - 5});
    output["dense"] = hybrid_array(batch.dense, {static_cast<py::ssize_t>(count), 3, board_size - 3, board_size - 3});
    output["dense_delta"] = hybrid_array(batch.dense_delta, {static_cast<py::ssize_t>(count), 3, board_size - 3, board_size - 3});
    return output;
}

py::dict old_champion_choose(
    const py::sequence& values, const py::sequence& legal_values, const int board_size,
    const int color, const double interior_diagonal, const double edge_diagonal,
    const double corner_diagonal, const double opponent_reply_coefficient,
    const int reply_top_k, const double reply_max_share) {
    if (board_size <= 0 || (color != kBlack && color != kWhite) ||
        reply_top_k <= 0 || !std::isfinite(interior_diagonal) ||
        !std::isfinite(edge_diagonal) || !std::isfinite(corner_diagonal) ||
        !std::isfinite(opponent_reply_coefficient) || !std::isfinite(reply_max_share)) {
        throw std::invalid_argument("Invalid Old Champion parameters");
    }
    const int action_size = board_size * board_size;
    const auto cells = extract_cells(values, action_size);
    std::vector<int> legal;
    legal.reserve(static_cast<std::size_t>(py::len(legal_values)));
    for (const py::handle item : legal_values) {
        const int action = item.cast<int>();
        if (action < 0 || action >= action_size || cells[static_cast<std::size_t>(action)] != 0) {
            throw std::invalid_argument("Old Champion legal actions are invalid");
        }
        legal.push_back(action);
    }
    if (legal.empty()) throw std::invalid_argument("Old Champion requires legal actions");
    const auto diagonal_weight = [&](const int action) {
        const int row = action / board_size, column = action % board_size;
        if ((row == 0 || row == board_size - 1) &&
            (column == 0 || column == board_size - 1)) return corner_diagonal;
        if (row == 0 || row == board_size - 1 || column == 0 || column == board_size - 1) return edge_diagonal;
        return interior_diagonal;
    };
    const auto for_window = [&](const int action, const int radius, auto&& fn) {
        const int row = action / board_size, column = action % board_size;
        for (int r = std::max(0, row - radius); r < std::min(board_size, row + radius + 1); ++r)
            for (int c = std::max(0, column - radius); c < std::min(board_size, column + radius + 1); ++c)
                fn(r * board_size + c);
    };
    const auto value_of_cell = [&](const std::vector<std::uint8_t>& board_cells, const int target,
                                   const int perspective, const int added_move, const int added_color) {
        const int occupant = added_move == target ? added_color : board_cells[static_cast<std::size_t>(target)];
        if (occupant != 0) return occupant == perspective ? 1.0 : -1.0;
        const int row = target / board_size, column = target % board_size;
        double value = 0.0;
        for_window(target, 1, [&](const int source) {
            const int piece = added_move == source ? added_color : board_cells[static_cast<std::size_t>(source)];
            if (piece == 0) return;
            const int sr = source / board_size, sc = source % board_size;
            const double weight = std::abs(sr - row) == 1 && std::abs(sc - column) == 1
                ? diagonal_weight(target) : 1.0;
            value += (piece == perspective ? 1.0 : -1.0) * weight;
        });
        return std::max(-1.0, std::min(1.0, value));
    };
    const auto move_gain = [&](const std::vector<std::uint8_t>& board_cells, const int move, const int player) {
        double result = 0.0;
        for_window(move, 1, [&](const int target) {
            result += value_of_cell(board_cells, target, player, move, player) -
                      value_of_cell(board_cells, target, player, -1, 0);
        });
        return result;
    };
    const auto contains_legal = [&](const int action) {
        return std::find(legal.begin(), legal.end(), action) != legal.end();
    };
    const auto is_legal_after = [&](const std::vector<std::uint8_t>& board_cells, const int action) {
        if (board_cells[static_cast<std::size_t>(action)] != 0) return false;
        bool blocked = false;
        for_window(action, 1, [&](const int neighbor) {
            if (board_cells[static_cast<std::size_t>(neighbor)] != 0) blocked = true;
        });
        return !blocked;
    };
    std::vector<double> immediate(static_cast<std::size_t>(action_size), 0.0);
    for (const int move : legal) immediate[static_cast<std::size_t>(move)] = move_gain(cells, move, color);
    const int other = color == kBlack ? kWhite : kBlack;
    std::vector<double> baseline(static_cast<std::size_t>(action_size), 0.0);
    for (const int move : legal) baseline[static_cast<std::size_t>(move)] = move_gain(cells, move, other);
    std::vector<int> sorted_replies = legal;
    std::sort(sorted_replies.begin(), sorted_replies.end(), [&](const int a, const int b) {
        return baseline[static_cast<std::size_t>(a)] > baseline[static_cast<std::size_t>(b)];
    });
    const double coefficient = std::max(0.0, opponent_reply_coefficient);
    double best_score = -std::numeric_limits<double>::infinity();
    std::vector<int> best_moves;
    for (const int move : legal) {
        std::vector<std::uint8_t> after = cells;
        after[static_cast<std::size_t>(move)] = static_cast<std::uint8_t>(color);
        std::vector<double> gains;
        for (const int reply : sorted_replies) {
            bool nearby = false;
            for_window(move, 2, [&](const int candidate) { if (candidate == reply) nearby = true; });
            if (!nearby) {
                gains.push_back(baseline[static_cast<std::size_t>(reply)]);
                if (static_cast<int>(gains.size()) >= reply_top_k) break;
            }
        }
        for_window(move, 2, [&](const int reply) {
            if (contains_legal(reply) && is_legal_after(after, reply))
                gains.push_back(move_gain(after, reply, other));
        });
        std::sort(gains.begin(), gains.end(), std::greater<double>());
        if (static_cast<int>(gains.size()) > reply_top_k) gains.resize(static_cast<std::size_t>(reply_top_k));
        double threat = 0.0;
        if (!gains.empty()) {
            const double mean = std::accumulate(gains.begin(), gains.end(), 0.0) /
                                static_cast<double>(gains.size());
            threat = reply_max_share * gains.front() + (1.0 - reply_max_share) * mean;
        }
        const double score = immediate[static_cast<std::size_t>(move)] - coefficient * threat;
        if (score > best_score + 1e-12) {
            best_score = score;
            best_moves.clear();
            best_moves.push_back(move);
        } else if (std::abs(score - best_score) <= 1e-12) {
            best_moves.push_back(move);
        }
    }
    py::dict result;
    result["best_moves"] = best_moves;
    result["score"] = best_score;
    result["immediate_gain"] = best_moves.empty() ? 0.0 : immediate[static_cast<std::size_t>(best_moves.front())];
    return result;
}

struct OldChampionCoreResult {
    std::vector<int> best_moves;
    double score{};
};

OldChampionCoreResult old_champion_choose_core(
    const std::vector<std::uint8_t>& cells, const std::vector<int>& legal,
    const int board_size, const int color, const double interior_diagonal,
    const double edge_diagonal, const double corner_diagonal,
    const double opponent_reply_coefficient, const int reply_top_k,
    const double reply_max_share, const std::vector<std::vector<int>>& windows1,
    const std::vector<std::vector<int>>& windows2,
    const std::vector<double>& diagonal_weights) {
    const int action_size = board_size * board_size;
    const int other = color == kBlack ? kWhite : kBlack;
    const auto value_of_cell = [&](const std::vector<std::uint8_t>& board_cells,
                                   const int target, const int perspective,
                                   const int added_move) {
        const int occupant = added_move == target
            ? perspective : board_cells[static_cast<std::size_t>(target)];
        if (occupant != 0) return occupant == perspective ? 1.0 : -1.0;
        const int row = target / board_size, column = target % board_size;
        double value = 0.0;
        for (const int source : windows1[static_cast<std::size_t>(target)]) {
            const int piece = added_move == source
                ? perspective : board_cells[static_cast<std::size_t>(source)];
            if (piece == 0) continue;
            const int sr = source / board_size, sc = source % board_size;
            const double weight = std::abs(sr - row) == 1 && std::abs(sc - column) == 1
                ? diagonal_weights[static_cast<std::size_t>(target)] : 1.0;
            value += (piece == perspective ? 1.0 : -1.0) * weight;
        }
        return std::max(-1.0, std::min(1.0, value));
    };
    const auto move_gain = [&](const std::vector<std::uint8_t>& board_cells,
                               const int move, const int player) {
        double result = 0.0;
        for (const int target : windows1[static_cast<std::size_t>(move)]) {
            result += value_of_cell(board_cells, target, player, move) -
                      value_of_cell(board_cells, target, player, -1);
        }
        return result;
    };
    const auto legal_after = [&](const std::vector<std::uint8_t>& board_cells,
                                 const int action) {
        if (board_cells[static_cast<std::size_t>(action)] != 0) return false;
        for (const int neighbor : windows1[static_cast<std::size_t>(action)])
            if (board_cells[static_cast<std::size_t>(neighbor)] != 0) return false;
        return true;
    };
    std::vector<unsigned char> legal_mask(static_cast<std::size_t>(action_size), 0);
    for (const int action : legal) legal_mask[static_cast<std::size_t>(action)] = 1;
    std::vector<double> immediate(static_cast<std::size_t>(action_size), 0.0);
    std::vector<double> baseline(static_cast<std::size_t>(action_size), 0.0);
    for (const int move : legal) {
        immediate[static_cast<std::size_t>(move)] = move_gain(cells, move, color);
        baseline[static_cast<std::size_t>(move)] = move_gain(cells, move, other);
    }
    std::vector<int> sorted_replies = legal;
    std::sort(sorted_replies.begin(), sorted_replies.end(), [&](int a, int b) {
        return baseline[static_cast<std::size_t>(a)] > baseline[static_cast<std::size_t>(b)];
    });
    double best_score = -std::numeric_limits<double>::infinity();
    std::vector<int> best_moves;
    std::vector<unsigned char> nearby(static_cast<std::size_t>(action_size), 0);
    for (const int move : legal) {
        std::fill(nearby.begin(), nearby.end(), 0);
        for (const int action : windows2[static_cast<std::size_t>(move)])
            nearby[static_cast<std::size_t>(action)] = 1;
        std::vector<std::uint8_t> after = cells;
        after[static_cast<std::size_t>(move)] = static_cast<std::uint8_t>(color);
        std::vector<double> gains;
        for (const int reply : sorted_replies) {
            if (!nearby[static_cast<std::size_t>(reply)]) {
                gains.push_back(baseline[static_cast<std::size_t>(reply)]);
                if (static_cast<int>(gains.size()) >= reply_top_k) break;
            }
        }
        for (const int reply : windows2[static_cast<std::size_t>(move)]) {
            if (legal_mask[static_cast<std::size_t>(reply)] &&
                legal_after(after, reply)) {
                gains.push_back(move_gain(after, reply, other));
            }
        }
        std::sort(gains.begin(), gains.end(), std::greater<double>());
        if (static_cast<int>(gains.size()) > reply_top_k)
            gains.resize(static_cast<std::size_t>(reply_top_k));
        double threat = 0.0;
        if (!gains.empty()) {
            const double mean = std::accumulate(gains.begin(), gains.end(), 0.0) /
                                static_cast<double>(gains.size());
            threat = reply_max_share * gains.front() +
                     (1.0 - reply_max_share) * mean;
        }
        const double score = immediate[static_cast<std::size_t>(move)] -
                             std::max(0.0, opponent_reply_coefficient) * threat;
        if (score > best_score + 1e-12) {
            best_score = score;
            best_moves.clear();
            best_moves.push_back(move);
        } else if (std::abs(score - best_score) <= 1e-12) {
            best_moves.push_back(move);
        }
    }
    return {std::move(best_moves), best_score};
}

py::list old_champion_generate_games(
    const int games, const int board_size, const int neighborhood_radius,
    const std::uint64_t zobrist_seed, const std::uint64_t random_seed,
    const double interior_diagonal, const double edge_diagonal,
    const double corner_diagonal, const double opponent_reply_coefficient,
    const int reply_top_k, const double reply_max_share) {
    if (games <= 0 || board_size <= 0 || neighborhood_radius != 1)
        throw std::invalid_argument("Invalid Old Champion generation dimensions");
    const int action_size = board_size * board_size;
    std::vector<std::vector<int>> windows1(static_cast<std::size_t>(action_size));
    std::vector<std::vector<int>> windows2(static_cast<std::size_t>(action_size));
    std::vector<double> diagonal_weights(static_cast<std::size_t>(action_size));
    for (int action = 0; action < action_size; ++action) {
        const int row = action / board_size, column = action % board_size;
        for (int radius : {1, 2}) {
            auto& window = radius == 1 ? windows1[static_cast<std::size_t>(action)]
                                       : windows2[static_cast<std::size_t>(action)];
            for (int r = std::max(0, row - radius); r < std::min(board_size, row + radius + 1); ++r)
                for (int c = std::max(0, column - radius); c < std::min(board_size, column + radius + 1); ++c)
                    window.push_back(r * board_size + c);
        }
        diagonal_weights[static_cast<std::size_t>(action)] =
            (row == 0 || row == board_size - 1) &&
            (column == 0 || column == board_size - 1) ? corner_diagonal :
            (row == 0 || row == board_size - 1 || column == 0 || column == board_size - 1)
                ? edge_diagonal : interior_diagonal;
    }
    std::mt19937_64 rng(random_seed);
    py::list output;
    for (int game_index = 0; game_index < games; ++game_index) {
        BoardCore board(board_size, neighborhood_radius, zobrist_seed);
        std::vector<std::vector<int>> policy_moves;
        while (!board.is_terminal()) {
            const std::vector<int> legal = board.legal_actions();
            const auto choice = old_champion_choose_core(
                board.cells(), legal, board_size, board.to_play(),
                interior_diagonal, edge_diagonal, corner_diagonal,
                opponent_reply_coefficient, reply_top_k, reply_max_share,
                windows1, windows2, diagonal_weights);
            if (choice.best_moves.empty()) break;
            std::uniform_int_distribution<std::size_t> pick(0, choice.best_moves.size() - 1);
            const int action = choice.best_moves[pick(rng)];
            policy_moves.push_back(choice.best_moves);
            board.apply(action);
        }
        const NativeScore score = board.score();
        py::dict record;
        record["moves"] = board.history();
        record["best_moves"] = policy_moves;
        std::vector<int> matrix;
        matrix.reserve(static_cast<std::size_t>(action_size));
        for (int action = 0; action < action_size; ++action) {
            const int cell = board.cells()[static_cast<std::size_t>(action)];
            const int owner = score.owners[static_cast<std::size_t>(action)];
            matrix.push_back(cell == kBlack ? 1 : cell == kWhite ? -1 :
                             owner == kBlack ? 2 : owner == kWhite ? -2 : 0);
        }
        record["final_matrix"] = matrix;
        record["black_score"] = score.black_score;
        record["white_score"] = score.white_score;
        output.append(std::move(record));
    }
    return output;
}

}  // namespace

PYBIND11_MODULE(_hexadeca_native, module) {
    module.doc() = "C++20 board, scoring, feature, and MCTS kernels for Hexadeca.";

    py::class_<NativeScore>(module, "NativeScore")
        .def_readonly("black_score", &NativeScore::black_score)
        .def_readonly("white_score", &NativeScore::white_score)
        .def_readonly("owners", &NativeScore::owners)
        .def_readonly("distances", &NativeScore::distances);

    py::class_<BoardCore>(module, "NativeBoard")
        .def(py::init<int, int, std::uint64_t>())
        .def_property_readonly("size", &BoardCore::size)
        .def_property_readonly("action_size", &BoardCore::action_size)
        .def_property_readonly("ply", &BoardCore::ply)
        .def_property_readonly("to_play", &BoardCore::to_play)
        .def_property_readonly("legal_count", &BoardCore::legal_count)
        .def_property_readonly("is_terminal", &BoardCore::is_terminal)
        .def_property_readonly("zobrist_hash", &BoardCore::zobrist_hash)
        .def_property_readonly("cells", &BoardCore::cells)
        .def_property_readonly("history", &BoardCore::history)
        .def("is_legal", &BoardCore::is_legal)
        .def("piece_at", &BoardCore::piece_at)
        .def("legal_actions", &BoardCore::legal_actions)
        .def("legal_mask", &BoardCore::legal_mask)
        .def("apply", &BoardCore::apply)
        .def("undo", &BoardCore::undo)
        .def("score", &BoardCore::score);

    py::class_<NativeSearchTree>(module, "NativeSearchTree")
        .def(py::init<int, int, std::uint64_t, const std::vector<int>&, double, double, int>())
        .def("initialize_root", &NativeSearchTree::initialize_root)
        .def("select_batch", &NativeSearchTree::select_batch)
        .def("leaf_state", &NativeSearchTree::leaf_state)
        .def("leaf_states", &NativeSearchTree::leaf_states)
        .def("leaf_packed_states", &NativeSearchTree::leaf_packed_states)
        .def("commit", &NativeSearchTree::commit)
        .def("commit_batch", &NativeSearchTree::commit_batch)
        .def("solve", &NativeSearchTree::solve)
        .def("cancel", &NativeSearchTree::cancel)
        .def("root_statistics", &NativeSearchTree::root_statistics)
        .def("export_nodes", &NativeSearchTree::export_nodes);

    module.def("score_cells", &score_cells);
    module.def("compute_handcrafted_features", &compute_handcrafted_features,
               py::arg("current_cells"), py::arg("previous_cells"),
               py::arg("board_size"));
    module.def("old_champion_choose", &old_champion_choose,
               py::arg("cells"), py::arg("legal_actions"), py::arg("board_size"),
               py::arg("color"), py::arg("interior_diagonal"),
               py::arg("edge_diagonal"), py::arg("corner_diagonal"),
               py::arg("opponent_reply_coefficient"), py::arg("reply_top_k"),
               py::arg("reply_max_share"));
    module.def("old_champion_generate_games", &old_champion_generate_games,
               py::arg("games"), py::arg("board_size"), py::arg("neighborhood_radius"),
               py::arg("zobrist_seed"), py::arg("random_seed"),
               py::arg("interior_diagonal"), py::arg("edge_diagonal"),
               py::arg("corner_diagonal"), py::arg("opponent_reply_coefficient"),
               py::arg("reply_top_k"), py::arg("reply_max_share"));
    module.def("encode_features", &encode_features,
               py::arg("states"), py::arg("board_size"), py::arg("input_planes"),
               py::arg("ruleset_id"));
    module.def("encode_packed_features", &encode_packed_features,
               py::arg("packed_states"), py::arg("board_size"),
               py::arg("input_planes"));
    module.def("decode_packed_hybrid_inputs", &decode_packed_hybrid_inputs,
               py::arg("packed_states"), py::arg("board_size"));
    module.def("pack_compact_states", &pack_compact_states,
               py::arg("states"), py::arg("board_size"), py::arg("ruleset_id"));
    module.def("pack_compact_evaluations", &pack_compact_evaluations,
               py::arg("evaluations"), py::arg("action_size"));
    module.def("unpack_compact_evaluations", &unpack_compact_evaluations,
               py::arg("packed_evaluations"), py::arg("action_size"));
}
