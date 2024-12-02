#include "perceptron.h"

#include <vector>
#include <cmath>

extern "C" {
#include "bp/bp.param.h"
#include "core.param.h"
#include "globals/assert.h"
#include "statistics.h"
}

#define WEIGHT_INIT_VALUE 0
#define THETA floor(1.93 * HIST_LENGTH + 14)
#define DEBUG(proc_id, args...) _DEBUG(proc_id, DEBUG_BP_DIR, ##args)
#define SAT_INC_MIN_MAX(val, inc, min, max) \
    ((val) + (inc) > (max) ? (max) : \
     (val) + (inc) < (min) ? (min) : \
     (val) + (inc))

namespace {

struct Perceptron_Entry {
  int8_t bias;
  std::vector<int8_t> weights;
};

struct Perceptron_State {
  std::vector<Perceptron_Entry> table;
};

std::vector<Perceptron_State> perceptron_state_all_cores;

uns32 get_table_index(const Addr addr) {
  const uns32 cooked_addr = (addr >> 2) & N_BIT_MASK(HIST_LENGTH);
  return cooked_addr;
}

int compute_output(const Perceptron_Entry& entry, const uns32 hist) {
  int sum = entry.bias;
  const uns32 masked_hist = hist >> (32 - HIST_LENGTH);
  
  for(size_t i = 0; i < HIST_LENGTH; i++) {
    const int history_bit = (masked_hist >> i) & 0x1;
    const int x = 2 * history_bit - 1; 
    sum += x * entry.weights[i];
  }
  return sum;
}

void train_perceptron(Perceptron_Entry& entry, const uns32 hist, bool outcome) {
  const int y = outcome ? 1 : -1;
  const uns32 masked_hist = hist >> (32 - HIST_LENGTH);
  
  // Update bias
  entry.bias = SAT_INC_MIN_MAX(entry.bias, y, -128, 127);
  
  // Update weights
  for(size_t i = 0; i < HIST_LENGTH; i++) {
    const int history_bit = (masked_hist >> i) & 0x1;
    const int x = 2 * history_bit - 1;
    entry.weights[i] = SAT_INC_MIN_MAX(entry.weights[i], x * y, -128, 127);
  }
}

}  // namespace

void bp_perceptron_init() {
  perceptron_state_all_cores.resize(NUM_CORES);
  for(auto& perceptron_state : perceptron_state_all_cores) {
    perceptron_state.table.resize(1 << HIST_LENGTH);
    for(auto& entry : perceptron_state.table) {
      entry.weights.resize(HIST_LENGTH, WEIGHT_INIT_VALUE);
      entry.bias = WEIGHT_INIT_VALUE;
    }
  }
}

uns8 bp_perceptron_pred(Op* op) {
  const uns   proc_id = op->proc_id;
  const auto& perceptron_state = perceptron_state_all_cores.at(proc_id);
  
  const Addr  addr = op->oracle_info.pred_addr;
  const uns32 hist = op->oracle_info.pred_global_hist;
  const uns32 table_index = get_table_index(addr);
  const int   output = compute_output(perceptron_state.table[table_index], hist);
  const uns8  pred = output >= 0 ? 1 : 0;
  
  DEBUG(proc_id, "Predicting with perceptron for op_num:%s index:%d\n",
        unsstr64(op->op_num), table_index);
  DEBUG(proc_id, "Predicting addr:%s output:%d pred:%d dir:%d\n",
        hexstr64s(addr), output, pred, op->oracle_info.dir);
  
  return pred;
}

void bp_perceptron_update(Op* op) {
  if(op->table_info->cf_type != CF_CBR) {
    return;
  }
  
  const uns proc_id = op->proc_id;
  auto& perceptron_state = perceptron_state_all_cores.at(proc_id);
  
  const Addr  addr = op->oracle_info.pred_addr;
  const uns32 hist = op->oracle_info.pred_global_hist;
  const uns32 table_index = get_table_index(addr);
  const bool  outcome = op->oracle_info.dir;

  const int output = compute_output(perceptron_state.table[table_index], hist);
  const uns8 pred = output >= 0 ? 1 : 0;  
  
  /*Update if prediction is wrong*/ 
  if (pred != outcome || abs(output) < THETA){
    DEBUG(proc_id, "Training perceptron for op_num:%s index:%d dir:%d\n",
          unsstr64(op->op_num), table_index, outcome);
  
    train_perceptron(perceptron_state.table[table_index], hist, outcome);
  
    DEBUG(proc_id, "Updated addr:%s table:%u outcome:%d\n",
          hexstr64s(addr), table_index, outcome);
  }
}

// The only speculative state of perceptron is the global history which is managed
// by bp.c. Thus, no internal timestamping and recovery mechanism is needed.
void bp_perceptron_timestamp(Op* op) {}
void bp_perceptron_recover(Recovery_Info* info) {}
void bp_perceptron_spec_update(Op* op) {}
void bp_perceptron_retire(Op* op) {}
uns8 bp_perceptron_full(uns proc_id) { return 0; }