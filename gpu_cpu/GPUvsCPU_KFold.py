import uproot
import awkward as ak
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import pyplot
import sklearn.metrics
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import timeit
import time
import os
import seaborn as sns
from tqdm.notebook import tqdm
import torch.nn.functional as F
import numpy
from sklearn.model_selection import RepeatedKFold 


# Function to check device availability
def device_available(force_cpu=False): 
    if force_cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Using CPU")
    else:
        device = torch.device('cuda')
        print("Using GPU")
    return device

# Function to move data to the selected device
def feed_device(data, device): 
    if isinstance(data, (list, tuple)):
        return [feed_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# Class to wrap a dataloader to move data to a device
class DeviceDataLoader(): 
    def __init__(self, dl, device):
        self.dl = dl   
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield feed_device(b, self.device)
    
    def __len__(self):
        return len(self.dl)  

# Neural Network Model
'''
class ANN_Model(nn.Module): 
    def __init__(
        self,
        input_features=12,
        hidden1=512, hidden2=512,
        hidden3=64, hidden4=28,
        out_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features, hidden1)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.f_connected3 = nn.Linear(hidden2, hidden3)
        self.out = nn.Linear(hidden3, out_features)
    
    def forward(self, x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = F.relu(self.f_connected3(x))
        x = torch.sigmoid(self.out(x))
        return x
'''

class ANN_Model(nn.Module):
    def __init__(self, 
                 input_features = 12, 
                 out_features=2,
                 hdn_dim=30):
       
        super().__init__()
       
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_features, hdn_dim),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(hdn_dim, hdn_dim),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(hdn_dim, hdn_dim),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(hdn_dim, hdn_dim),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(hdn_dim, out_features),
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out

#Train and evaluate each fold. 
def train_evaluate(model, train_loader, test_loader, device, loss_function, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        train_losses = [] #Store the losses of the model
        for inpt, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"): 
            inpt, target = inpt.to(device), target.to(device) #Move data to device
            optimizer.zero_grad() #clears gradients
            out=model(inpt) #FORWARD PASS: predictions
            loss = loss_function(out, target) #loss function: compares the model predictions to the real data.
            loss.backward()  
            optimizer.step()  
            train_losses.append(loss.item())
        avg_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    #VALIDATION
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad(): #to not calculate and update gradients since its a validation
        for inpt, target in test_loader:
            inpt, target = inpt.to(device), target.to(device)
            out = model(inpt)
            _, preds = torch.max(out, dim=1) 
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
    
    #METRICS
    accuracy = accuracy_score(test_targets, test_preds)
    precision_pos = precision_score(test_targets, test_preds, pos_label=1)  
    precision_neg = precision_score(test_targets, test_preds, pos_label=0)
    return accuracy, precision_pos, precision_neg


# Main function to run the entire process
def main():
    file_train = uproot.open('/home/emma-gutierrez/Servicio_Becario/Tutorials/Lab_challenge/data/train_D02kpipi0vxVc-cont0p5.root')
    tree_train = file_train['d0tree']
    df_train = tree_train.arrays(library="pd")
    shuffled_df = df_train.sample(frac=1, random_state=4)

    df_sig = shuffled_df.loc[shuffled_df['isSignal'] == 1].sample(n=20000, random_state=42)
    df_bkg = shuffled_df.loc[shuffled_df['isSignal'] == 0].sample(n=20000, random_state=42)
    df_comb = pd.concat([df_sig, df_bkg])
    df_comb = df_comb.drop(['vM', 'vpCMS', '__index__'], axis=1)
    df_comb = df_comb.sample(frac=1, random_state=1)

    # Convert data to numpy arrays 
    X = df_comb.drop(['isSignal'], axis=1)
    Y = df_comb['isSignal']
    X = torch.tensor(X.values.astype(np.float32))
    Y = torch.tensor(Y.values.astype(np.int64))

    # DEFINE NUMBER OF K-FOLDS
    n_splits = 5 #number of folds
    n_repeats = 1 #number of repetitions
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
    # List to store results 
    results_gpu_df = pd.DataFrame(columns=['Batch_Size', 'Time', 'Accuracy', 'Precision'])
    results_cpu_df = pd.DataFrame(columns=['Batch_Size', 'Time', 'Accuracy', 'Precision'])

    torch.manual_seed(5) #ensures reproducibility of the data in pytorch. (used for randperm)
    batch_size = 4000 #amount of data that the NN is analyzing. 
    data_diff_sizes = []
    for i in range(10):
        train_tensor = torch.utils.data.TensorDataset(X, Y)
        indices = torch.randperm(len(train_tensor))[:batch_size]
        selected_samples = [train_tensor[k] for k in indices]
        data_diff_sizes.append(selected_samples)
        batch_size += 4000

    batch_size = 4000
    for selected_samples in data_diff_sizes:     
        print(f"Training with batch size: {batch_size}")
        for fold, (train_index, test_index) in enumerate(rkf.split(X)):
            print(f"\nFold {fold + 1} of {n_splits * n_repeats}")
        
            # Split data into training and validation
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
        
            train_dataset = TensorDataset(X_train, Y_train)
            test_dataset = TensorDataset(X_test, Y_test)
            train_loader = DataLoader(train_dataset, 
                                    batch_size=64,
                                    shuffle=True,
                                    num_workers=4,
                                    pin_memory=True)
            test_loader = DataLoader(test_dataset, 
                                    batch_size=64,
                                    shuffle=False,
                                    num_workers=4,
                                    pin_memory=True) 
        
            # CODE FOR GPU 
            device = device_available(force_cpu=False)
            train_loader_gpu = DeviceDataLoader(train_loader, device)
            test_loader_gpu = DeviceDataLoader(test_loader, device)

            model = ANN_Model()
            model = feed_device(model, device)
            loss_function = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            start_time = time.time()
            accuracy, precision_pos, precision_neg = train_evaluate(model, train_loader_gpu, test_loader_gpu, device, loss_function, optimizer)
            training_time = time.time() - start_time
        
            new_row = pd.DataFrame({'Batch_Size': [batch_size], 'Time': [training_time], 'Accuracy': [accuracy], "Precision Pos": [precision_pos], "Precision Neg": [precision_neg]})
            results_gpu_df = pd.concat([results_gpu_df, new_row], ignore_index=True)

            print(f"GPU - Fold {fold + 1} - Accuracy: {accuracy:.4f}, Precision Pos: {precision_pos:.4f},  Precision Neg: {precision_neg:.4f}, Training Time: {training_time:.2f}s")

            # CODE FOR CPU
            device = device_available(force_cpu=True)
            train_loader_cpu = DeviceDataLoader(train_loader, device)
            test_loader_cpu = DeviceDataLoader(test_loader, device)

            model = ANN_Model()
            model = feed_device(model, device)
            loss_function = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            start_time = time.time()
            accuracy, precision_pos, precision_neg = train_evaluate(model, train_loader_cpu, test_loader_cpu, device, loss_function, optimizer)
            training_time = time.time() - start_time

            new_row = pd.DataFrame({'Batch_Size': [batch_size], 'Time': [training_time], 'Accuracy': [accuracy], "Precision Pos": [precision_pos], "Precision Neg": [precision_neg]})
            results_cpu_df = pd.concat([results_cpu_df, new_row], ignore_index=True)

            print(f"CPU - Fold {fold + 1} - Accuracy: {accuracy:.4f}, Precision Pos: {precision_pos:.4f}, Precision Neg: {precision_neg:.4f}, Training Time: {training_time:.2f}s")
            
        batch_size += 4000
    
    results_gpu_df.to_csv("results_gpu_nn.csv", index=False)
    results_cpu_df.to_csv("results_cpu_nn.csv", index=False)

    df_gpu = pd.read_csv('results_gpu_nn.csv')
    df_cpu = pd.read_csv('results_cpu_nn.csv')

    # Calculate the average metrics for each batch size for GPU
    avg_gpu = df_gpu.groupby('Batch_Size').mean().reset_index()
    print("Average Metrics for GPU:")
    print(avg_gpu)

    # Calculate the average metrics for each batch size for CPU
    avg_cpu = df_cpu.groupby('Batch_Size').mean().reset_index()
    print("\nAverage Metrics for CPU:")
    print(avg_cpu)

    # Plotting the average metrics
    plt.figure(figsize=(10, 5))
    plt.plot(avg_gpu['Batch_Size'], avg_gpu['Time'], marker='o', label='GPU')
    plt.plot(avg_cpu['Batch_Size'], avg_cpu['Time'], marker='o', label='CPU')
    plt.xlabel('Batch Size')
    plt.ylabel('Average Time (s)')
    plt.title('Average Time Comparison: GPU vs CPU Processing')
    plt.legend()
    plt.grid(True)
    plt.savefig("avg_time.pdf")

    plt.figure(figsize=(10, 5))
    plt.plot(avg_gpu['Batch_Size'], avg_gpu['Accuracy'], marker='o', label='GPU')
    plt.plot(avg_cpu['Batch_Size'], avg_cpu['Accuracy'], marker='o', label='CPU')
    plt.xlabel('Batch Size')
    plt.ylabel('Average Accuracy')
    plt.title('Average Accuracy Comparison: GPU vs CPU')
    plt.legend()
    plt.grid(True)
    plt.savefig("avg_acc.pdf")

    plt.figure(figsize=(10, 5))
    plt.plot(avg_gpu['Batch_Size'], avg_gpu['Precision Pos'], marker='o', label='GPU')
    plt.plot(avg_cpu['Batch_Size'], avg_cpu['Precision Pos'], marker='o', label='CPU')
    plt.xlabel('Batch Size')
    plt.ylabel('Average Precision Pos')
    plt.title('Average Precision Pos Comparison: GPU vs CPU')
    plt.legend()
    plt.grid(True)
    plt.savefig("avg_prec_pos.pdf")

    plt.figure(figsize=(10, 5))
    plt.plot(avg_gpu['Batch_Size'], avg_gpu['Precision Neg'], marker='o', label='GPU')
    plt.plot(avg_cpu['Batch_Size'], avg_cpu['Precision Neg'], marker='o', label='CPU')
    plt.xlabel('Batch Size')
    plt.ylabel('Average Precision Neg')
    plt.title('Average Precision Neg Comparison: GPU vs CPU')
    plt.legend()
    plt.grid(True)
    plt.savefig("avg_prec_neg.pdf")

if __name__ == "__main__":
    main()