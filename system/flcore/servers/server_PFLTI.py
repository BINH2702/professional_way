import copy
import torch
from flcore.clients.client_PFLTI import clientPFLTI
from flcore.servers.serverbase import Server

class PFLTI(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        #select clients
        self.set_slow_clients()
        self.set_clients(args, clientPFLTI)
        # self.mom_model = copy.deepcopy(args.model)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()
            # send all param to client
            self.send_models()
            # send mom_modl tp client
            self.send_mom_model()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model with one step update")
                self.evaluate_one_step()

            # choose several clients to send back upated model to server
            for client in self.selected_clients:
                client.train()
                # client.train()

            self.receive_models()
            self.aggregate_parameters()
            self.update_mom_model()

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))

        self.save_results()

    # def send_mom_model(self, client_mom_model):


    def evaluate_one_step(self):
        models_temp = []
        for c in self.clients:
            models_temp.append(copy.deepcopy(c.model))
            c.train_one_step()

        stats = self.test_metrics()

        # set local model back to client for training process
        for i, c in enumerate(self.clients):
            c.clone_model(models_temp[i], c.model)

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])

        self.rs_test_acc.append(test_acc)
        print("Average Test Accurancy: {:.4f}".format(test_acc))