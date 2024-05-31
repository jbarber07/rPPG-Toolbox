"""Trainer for DeepPhys."""

import logging
import os
from collections import OrderedDict
import time

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
import psutil
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
import subprocess


class DeepPhysTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        
        if config.TOOLBOX_MODE == "train_and_test":
            self.model = DeepPhys(img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            self.num_train_batches = len(data_loader["train"])
            self.criterion = torch.nn.MSELoss()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.model = DeepPhys(img_size=config.TEST.DATA.PREPROCESS.RESIZE.H).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("DeepPhys trainer initialized in incorrect toolbox mode!")

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
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(
                    self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix({"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]})

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
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(
                    self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)
                pred_ppg_valid = self.model(data_valid)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
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
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        config = self.config
        print("\n===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
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
        if not os.path.exists('./logs/inference_metrics/DEEPPHYS'):
            os.makedirs('./logs/inference_metrics/DEEPPHYS')
        
        writer = SummaryWriter(log_dir='./logs/inference_metrics/DEEPPHYS', filename_suffix='DEEPPHYS')
        
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
                data_test, labels_test = test_batch[0].to(self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)

                # Log input and output shapes
                writer.add_text('Input Shape', str(data_test.shape), batch_index)
                writer.add_text('Output Shape', str(labels_test.shape), batch_index)
                if batch_index == 0:  # Log only once
                    profiling_info['Input Shape'] = str(data_test.shape)
                    profiling_info['Output Shape'] = str(labels_test.shape)

                # Start profiling
                torch.cuda.synchronize()
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                
                pred_ppg_test = self.model(data_test)
                
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
                    labels_test = labels_test.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()
                    
                # Save each prediction to a file
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        writer.close()
        print('')
        
        # Save profiling information to a file
        profiling_file_path = f"./logs/inference_metrics/profiling_results_DeepPhys.txt"
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
        """Inits parameters from args and the writer for TensorboardX."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
 