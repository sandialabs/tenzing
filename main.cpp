#include <vector>
#include <set>
#include <iostream>
#include <string>

class Operation {
public:
    virtual void run() {}

    std::set<Operation *> succs;
    std::set<Operation *> preds;


    // do op after this
    Operation *then(Operation *op) {
        succs.insert(op);
        op->preds.insert(this);
        return op;
    }

    virtual ~Operation() {};
    virtual std::string name() {return "<anon>";}
};

class Start : public Operation {
    public:
    std::string name() override {return "Start"; }
};

class End : public Operation {
    public:
    std::string name() override {return "End"; }
};

class SpMV : public Operation {
public:
std::string name_;
SpMV(const std::string name) : name_(name) {}
std::string name() override {return name_; }
};

class VectorAdd : public Operation {
public:
std::string name_;
VectorAdd(const std::string name) : name_(name) {}
std::string name() override {return name_; }
};

class Scatter : public Operation {
public:
std::string name() override {return "scatter"; }
};

class PostRecv : public Operation {
public:
std::string name() override {return "PostRecv"; }
};

class WaitRecv : public Operation {
public:
std::string name() override {return "WaitRecv"; }
};

class PostSend : public Operation {
public:
std::string name() override {return "PostSend"; }
};

class WaitSend : public Operation {
public:
std::string name() override {return "WaitSend"; }
};



class Schedule {
public:
std::set<Operation *> remaining; // possible next operations
std::vector<Operation *> order;

void run() {
    for (Operation *op : order) {
        op->run();
    }
}
};



std::vector<Schedule> make_schedules(Operation *start) {
    std::vector<Schedule> currs; // input generation
    std::vector<Schedule> ret;
    
    {
        Schedule s;
        s.remaining.insert(start);
        currs.push_back(s);
    }


    while(!currs.empty()) {
        // all possible schedules derived from current schedules
        // followed by legal next operations
        std::vector<Schedule> nexts;
        
        for (Schedule &curr : currs) {
            // if no more allowed operations, the schedule
            // must be complete
            if (curr.remaining.empty()) {
                ret.push_back(curr);
            } else {
                // create schedules which are `curr` followed
                // by each legal next operation
                for (Operation *nextOp : curr.remaining) {

                    // check if nextOp's preds are all done
                    bool allDone = true;
                    for (Operation *check : nextOp->preds) {
                        if (curr.order.end() == std::find(curr.order.begin(), curr.order.end(), check)) {
                            allDone = false;
                            break;
                        }
                    }

                    // if all preds are done, nextOp can be
                    // next
                    if (allDone) {
                        Schedule next = curr;
                        next.remaining.erase(nextOp);
                        next.order.push_back(nextOp);
                        for (Operation *succ : nextOp->succs) {
                            next.remaining.insert(succ);
                        }
                        nexts.push_back(next);
                    }



                }
            }
        }
        currs = std::move(nexts);
    }

    return ret;
};



int main(void) {

    Start *start = new Start();
    Scatter *scatter = new Scatter();
    SpMV *yl = new SpMV("yl");
    SpMV *yr = new SpMV("yr");
    PostSend *postSend = new PostSend();
    WaitSend *waitSend = new WaitSend();
    PostRecv *postRecv = new PostRecv();
    WaitRecv *waitRecv = new WaitRecv();
    VectorAdd *y = new VectorAdd("y");
    End *end = new End();
    

    // immidiately recv, local spmv, or scatter
    start->then(yl);
    start->then(postRecv);
    start->then(scatter)->then(postSend);
    
    // remote matrix after recv
    waitRecv->then(yr)->then(end);

    // add after local and remote done, then end
    yl->then(y);
    yr->then(y);

    // end once add and send is done
    y->then(end);
    waitSend->then(end);

    // initiate sends and recvs before waiting for either
    postSend->then(waitSend);
    postSend->then(waitRecv);
    postRecv->then(waitSend);
    postRecv->then(waitRecv);

    std::vector<Schedule> schedules = make_schedules(start);

    std::cerr << schedules.size() << " schedules:\n";
    for (Schedule &s : schedules) {
        for (Operation *op : s.order) {
            std::cerr << op->name() << ", ";
        }
        std::cerr << "\n";
    }

    for (Schedule &sched : schedules) {
        sched.run();
    }

}