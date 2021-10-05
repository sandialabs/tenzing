#include "sched/mcts.hpp"


#include "sched/schedule.hpp"



namespace mcts {

void mpi_bcast(std::vector<std::shared_ptr<CpuNode>> &order, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // bcast number of operations in order
    int orderSize = order.size();
    MPI_Bcast(&orderSize, 1, MPI_INT, 0, comm);

    // bcast length of name of each operation
    std::vector<int> nameLengths;
    if (0 == rank) {
        for (const auto &op : order) {
            nameLengths.push_back(op->name().size());
        }
    } else {
        nameLengths.resize(orderSize);
    }
    MPI_Bcast(nameLengths.data(), nameLengths.size(), MPI_INT, 0, comm);

    // bcast names
    size_t totalLength = 0;
    for (auto l : nameLengths) {
        totalLength += l;
    }
    std::vector<char> allNames;
    if (0 == rank) {
        for (const auto &op : order) {
            for (auto c : op->name()) {
                allNames.push_back(c);
            }
        }
    } else {
        allNames.resize(totalLength);
    }
    MPI_Bcast(allNames.data(), allNames.size(), MPI_CHAR, 0, comm);

    // break into strings
    std::vector<std::string> names;
    size_t off = 0;
    for (auto l : nameLengths) {
        names.push_back(std::string(&allNames[off], l));
        off += l;
    }

    // find corresponding op in order if recieving
    if (0 != rank) {
        std::vector<size_t> pos;
        for (const std::string &name : names) {

            bool found = false;
            for (size_t oi = 0; oi < order.size(); ++oi) {
                if (order[oi]->name() == name) {
                    pos.push_back(oi);
                    found = true;
                    break;
                }
            }
            if (!found) {
                THROW_RUNTIME("couldn't find op for name " << name);
            }
        }

        // reorder operations
        std::vector<std::shared_ptr<CpuNode>> perm;
        for (size_t oi : pos) {
            perm.push_back(order[oi]);
        }
        order = perm;
    }
}


} // namespace mcts