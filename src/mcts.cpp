#include "sched/mcts.hpp"

#include "sched/operation_serdes.hpp"
#include "sched/schedule.hpp"



namespace mcts {

/* broadcast `order` from rank 0 to the other ranks
*/
std::vector<std::shared_ptr<BoundOp>> mpi_bcast(
    const std::vector<std::shared_ptr<BoundOp>> &order,
    const Graph<OpBase> &g,
    MPI_Comm comm
) {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string jsonStr;

    // serialize the sequence to json
    if (0 == rank) {
        nlohmann::json json;
        to_json(json, order, g);
        jsonStr = json.dump();
        STDERR(jsonStr);
    }

    // broadcast the JSON length and resize the receiving string
    {
        size_t jsonStrSz = jsonStr.size();
        MPI_Bcast(&jsonStrSz, sizeof(jsonStrSz), MPI_BYTE, 0, comm);
        jsonStr.resize(jsonStrSz);
    }

    // broadcast the JSON
    MPI_Bcast(&jsonStr[0], jsonStr.size(), MPI_CHAR, 0, comm);

    if (0 != rank) {
        // turn json string into json
        nlohmann::json des = nlohmann::json::parse(jsonStr);

        // deserialize the string into a sequence
        std::vector<std::shared_ptr<BoundOp>> seq;
        from_json(des, g, seq);
        return seq;
    } else {
        return order;
    }

}

void Result::dump_csv() const {

    const std::string delim("|");

    for (size_t i = 0; i < simResults.size(); ++i) {
        const auto &simres = simResults[i];
        std::cout << i;
        std::cout << delim << simres.benchResult.pct01;
        std::cout << delim << simres.benchResult.pct10;
        std::cout << delim << simres.benchResult.pct50;
        std::cout << delim << simres.benchResult.pct90;
        std::cout << delim << simres.benchResult.pct99;
        std::cout << delim << simres.benchResult.stddev;
        for (const auto &op : simres.path) {
            std::cout << "|" << op->json();
        }
        std::cout << "\n"; 
    }
}


} // namespace mcts