import numpy as np

class ClientEvent():
    def __init__(self, type, id, dur=0):
        self.type = type
        self.id = id
        self.dur = dur

    def read(self):
        return f"{self.type} {self.id} {self.dur}"

class SyncClientProcess():
    def __init__(self, n, part, lr):
        self.n = n
        self.state = [0] * n # 0 means not part, otherwise
        self.part = part
        self.lr = lr
        self.num_part = 0
        self.idle_set = set()
        self.all_set = set(range(self.n))

    def init(self):
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
                if arr_sample:
                    join.add(ind)
                    # In case where client leaves then joins again we operate normally
                    # Since departure events are prioritized, no race conditions occur
                    #if ind in depart:
                    #    # Handle case where client leaves then joins again
                    #    depart.remove(ind)
                    dur_sample = np.random.geometric(1.0/self.dep_rate)
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

        for i in range(s):
            all_clients = np.array([i for i in range(self.n)])
            part_clients = np.random.choice(all_clients, self.part, replace=False)

            for id in part_clients:
                event = ClientEvent("e", id, self.lr)
                event_list.append(event)
            for id in part_clients:
                event = ClientEvent("d", id)
                event_list.append(event)

        return event_list

def load_plan(fname):
    plan = []
    with open(fname) as pfile:
        for l in pfile.readlines():
            info = l.split()
            new_event = [info[0], int(info[1]), int(info[2])]
            plan.append(new_event)
    return plan

pop = 50
if __name__ == "__main__":
    params = {"pop": pop, "part": pop, "lr": 25}
    p = SyncClientProcess(params["pop"], params["part"], params["lr"])
    p.init()
    #print(list(p.part_set))
    #for _ in range(20):
    #    p.step()
    #    print(list(p.part_set))
    el = p.generate_schedule(400)
    ofile = f"sync_plan_{params['pop']}_{params['part']}_{params['lr']}.txt"
    with open(ofile, "w") as ofstream:
        for idx, e in enumerate(el):
            s = e.read()
            ofstream.write(s + "\n")
