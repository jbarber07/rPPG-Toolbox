"""PhysNet Trainer."""
import os
from collections import OrderedDict
import subprocess

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class PhysnetTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.loss_model = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.TRAIN.LR)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    batch[0].to(torch.float32).to(self.device))
                BVP_label = batch[1].to(
                    torch.float32).to(self.device)
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
                loss = self.loss_model(rPPG, BVP_label)
                loss.backward()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                tbar.set_postfix(loss=loss.item())

            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                BVP_label = valid_batch[1].to(
                    torch.float32).to(self.device)
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    valid_batch[0].to(torch.float32).to(self.device))
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
                loss_ecg = self.loss_model(rPPG, BVP_label)
                valid_loss.append(loss_ecg.item())
                valid_step += 1
                vbar.set_postfix(loss=loss_ecg.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def get_gpu_utilization(self):
        """Get the current GPU utilization."""
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                encoding='utf-8')
            return float(result.strip())
        except Exception as e:
            print(f"Failed to get GPU utilization: {e}")
            return -1

    def write_profiling_info(self,file_path, info):
        """Write profiling information to a file, ensuring no overwrite."""
        if not os.path.exists(file_path):
            with open(file_path, 'w') as file:
                for key, value in info.items():
                    file.write(f"{key}: {value}\n")
        else:
            with open(file_path, 'a') as file:
                for key, value in info.items():
                    file.write(f"{key}: {value}\n")

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print("\n===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")

        # TensorBoard setup
        if not os.path.exists('./logs/inference_metrics/PHYSNET'):
            os.makedirs('./logs/inference_metrics/PHYSNET')

        writer = SummaryWriter(log_dir='./logs/inference_metrics/PHYSNET', filename_suffix='PHYSNET')

        # Profiling information dictionary
        profiling_info = {}

        # Measure model size
        model_size = os.path.getsize(self.config.INFERENCE.MODEL_PATH) / (1024 * 1024)  # Size in Megabytes
        print(f"Model size: {model_size:.2f} MB")
        writer.add_scalar('Model Size (MB)', model_size)
        profiling_info['Model Size (MB)'] = model_size

        # Number of parameters
        num_parameters = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {num_parameters}")
        writer.add_scalar('Number of Parameters', num_parameters)
        profiling_info['Number of Parameters'] = num_parameters

        # Memory usage of the model weights
        model_memory = sum(p.element_size() * p.nelement() for p in self.model.parameters()) / (1024 ** 2)  # in MB
        print(f"Model memory usage: {model_memory:.2f} MB")
        writer.add_scalar('Model Memory Usage (MB)', model_memory)
        profiling_info['Model Memory Usage (MB)'] = model_memory

        with torch.no_grad():
            for batch_index, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(self.config.DEVICE), test_batch[1].to(self.config.DEVICE)

                # Log input and output shapes
                writer.add_text('Input Shape', str(data.shape), batch_index)
                writer.add_text('Output Shape', str(label.shape), batch_index)
                if batch_index == 0:  # Log only once
                    profiling_info['Input Shape'] = str(data.shape)
                    profiling_info['Output Shape'] = str(label.shape)

                # Start profiling
                torch.cuda.synchronize()
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()

                pred_ppg_test, _, _, _ = self.model(data)

                end_time.record()
                torch.cuda.synchronize()  # Wait for the events to be recorded!
                inference_time = start_time.elapsed_time(end_time)  # Time in milliseconds

                # Peak memory usage
                peak_memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB

                # GPU utilization
                gpu_utilization = self.get_gpu_utilization()

                # Logging to TensorBoard
                writer.add_scalar('Inference Time (ms)', inference_time, batch_index)
                writer.add_scalar('Peak Memory Usage (MB)', peak_memory_usage, batch_index)
                writer.add_scalar('GPU Utilization (%)', gpu_utilization, batch_index)

                if batch_index == 0:  # Log only once
                    profiling_info['Inference Time (ms)'] = inference_time
                    profiling_info['Peak Memory Usage (MB)'] = peak_memory_usage
                    profiling_info['GPU Utilization (%)'] = gpu_utilization

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    label = label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]
        
        writer.close()
        print('')
        
        # Save profiling information to a file
        profiling_file_path = f"./logs/inference_metrics/profiling_results_{self.model_file_name}.txt"
        self.write_profiling_info(profiling_file_path, profiling_info)
        
        gt_hr_fft_all, predict_hr_fft_all = calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR:  # saving test outputs 
            self.save_test_outputs(predictions, labels, self.config)
            for subj_index, sorted_preds in predictions.items():
                sorted_keys = sorted(sorted_preds.keys())
                with open(f"{self.config.TEST.OUTPUT_SAVE_DIR}/subj_{subj_index}_ppg.txt", 'w') as file:
                    for key in sorted_keys:
                        np.savetxt(file, sorted_preds[key], delimiter=',')
        return gt_hr_fft_all, predict_hr_fft_all

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
