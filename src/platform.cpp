#include "sched/platform.hpp"

void to_json(nlohmann::json& j, const Stream &s) {
    j = nlohmann::json{{"id", s.id_}};
}

void from_json(const nlohmann::json& j, Stream &s) {
    j.at("id").get_to(s.id_);
}

void to_json(nlohmann::json& j, const Event &s) {
    j = nlohmann::json{{"id", s.id_}};
}

void from_json(const nlohmann::json& j, Event &s) {
    j.at("id").get_to(s.id_);
}