import numpy as np
import os
import math

PARETO = 1.16
# PARETO = 0.9

class ClientEvent():
    def __init__(self, type, id, dur=0):
        self.type = type
        self.id = id
        self.dur = dur

    def read(self):
        return f"{self.type} {self.id} {self.dur}"

class ClientProcess():
    def __init__(self, n, arr, dep, straggler=0.1, straggler_dep=100):
        self.n = n
        self.state = [0] * n # 0 means not part, otherwise
        self.arr_rate = arr
        self.dep_rate = dep
        self.num_part = 0
        self.idle_set = set()
        self.all_set = set(range(self.n))
        self.straggler = straggler
        self.straggler_set = set()
        self.straggler_dep = straggler_dep

    def init(self):
        # Designate set of clients as perpetual stragglers?
        strag_count = math.floor(self.n * self.straggler)
        sample = np.random.choice(self.n, size=strag_count, replace=False)
        self.straggler_set = set(sample) 

        #client_set = np.arange(self.n)
        #sample = np.random.choice(client_set, self.num_part, replace=False)
        self.state = [0] * self.n
        self.idle_set = set()
        for id in range(self.n):
            self.idle_set.add(id)
        #durations = np.random.geometric(1.0/self.dep_rate, self.num_part)
        #for ind, dur in zip(sample, durations):
        #    self.part_set.add(ind)
        #    self.state[ind] = dur

    def step(self, pop=False):
        join = set()
        depart = set()
        for ind in range(self.n):
            if ind not in self.idle_set:
                self.state[ind] = max(0, self.state[ind] - 1)
                if self.state[ind] == 0:
                    # Remove recently completed 
                    self.idle_set.add(ind)
                    self.num_part -= 1
                    depart.add(ind)

        if not pop:
            avail_list = list(self.idle_set)
            for ind in avail_list:
                # Sample if participating
                arr_sample = np.random.binomial(size=1, n=1, p=1/self.arr_rate)
                if arr_sample[0]==1:
                    join.add(ind)
                    # In case where client leaves then joins again we operate normally
                    # Since departure events are prioritized, no race conditions occur 
                    #if ind in depart:
                        # TODO Handling this this way appears to cause problems, need to acknowledge client finishes
                        # Handle case where client leaves then joins again
                    #    depart.remove(ind)
                    dur_sample = math.ceil(np.random.pareto(PARETO) * self.dep_rate) # 1.16 is 80-20 rule for pareto
                    #if ind in self.straggler_set:
                    #    #dur_rate = self.straggler_dep
                    #    dur_sample = self.straggler_dep # Just fix the straggler behavior
                    #else:
                    #    dur_rate = self.dep_rate
                    #    dur_sample = np.random.geometric(1.0/dur_rate)
                        #dur_sample = self.dep_rate
                    
                    self.idle_set.remove(ind)
                    self.state[ind] = dur_sample
                    self.num_part += 1

        return join, depart

    def get_uids(self):
        return list(self.all_set - self.idle_set)

    def generate_schedule(self, s=20000):
        self.init()

        # s number of steps
        event_list = []
        validator = []

        for i in range(s):
            j, d = self.step()
            for id in d:
                event = ClientEvent("d", id)
                event_list.append(event)
            for id in j: 
                event = ClientEvent("e", id, self.state[id])
                event_list.append(event)

            cur = self.get_uids()
            validator.append(cur)

        # Disable end of program checkin
#        while len(self.idle_set) < self.n:
#            _, d = self.step(pop=True) # There should be no join events
#            for id in d:
#                event = ClientEvent("d", id)
#                event_list.append(event)
#
#            cur = self.get_uids()
#            validator.append(cur) 
#
        return event_list, validator

def load_plan(fname):
    plan = []
    with open(fname) as pfile:
        for l in pfile.readlines():
            info = l.split()
            new_event = [info[0], int(info[1]), int(info[2])]
            plan.append(new_event)
    return plan

POP=10
timeline=150000
if __name__ == "__main__":
    params = {"pop": POP, "arr": POP, "dur": 25, "straggler": 0.2, "straggler_dep": 200}
    p = ClientProcess(params["pop"], params["arr"], params["dur"], straggler=params["straggler"], straggler_dep=params["straggler_dep"])
    p.init()
    #print(list(p.part_set))
    #for _ in range(20):
    #    p.step()
    #    print(list(p.part_set))
    el, valid = p.generate_schedule(timeline)
    ofile = f"async{PARETO}_plan_{params['pop']}_{params['arr']}_{params['dur']}.txt" # _{params['straggler']}_{params['straggler_dep']}.txt"
    #print(valid)

    output_dir = "/ws/fs_mount/lora_lib_plans"
    ofile = os.path.join(output_dir, ofile)

    with open(ofile, "w") as ofstream:
        for idx, e in enumerate(el):
            s = e.read()
            ofstream.write(s + "\n")
