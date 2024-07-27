# proxmox_lxc_docker_cuda
notes of install and run cuda on docker within lxc and proxmox.

Tested with:
- Proxmox version: 8.2.4
- CPU: Intel 13700KF
- 2x 3090 24GB
- 96GB DDR5

use root for follwing:

**Create a Blacklist Configuration File:**
   - Open or create the blacklist configuration file using a text editor, for example, `vim`:
     ```bash
     vim /etc/modprobe.d/blacklist-nouveau.conf
     ```
   - Add the following lines to the file:
     ```plaintext
     blacklist nouveau
     options nouveau modeset=0
     ```
     
**PVE Kernel Headers**
  - Install kernel header for all version of pve
   ```
   apt install pve-headers
   ```

**Download and install cuda 12.5 from official source:**
```
wget https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda_12.5.1_555.42.06_linux.run
sh cuda_12.5.1_555.42.06_linux.run
```

**Create udev rules:**
    - Open or create the blacklist configuration file using a text editor, for example, `vim`:
      ```bash
      vim /etc/udev/rules.d/70-nvidia.rules
      ```
    - Add the following lines to the file:
     ```plaintext
     KERNEL=="nvidia", RUN+="/bin/bash -c '/usr/bin/nvidia-smi -L && /bin/chmod 666 /dev/nvidia*'"
     KERNEL=="nvidia_uvm", RUN+="/bin/bash -c '/usr/bin/nvidia-modprobe -c0 -u && /bin/chmod 0666 /dev/nvidia-uvm*'"
     ```
     
**Enable Kernel Modules:**
  - Open: 
  ```bash
       vim /etc/modules-load.d/modules.conf
  ```
  - Add the following lines to the file:
  
  ```plaintext
  # Nvidia modules
  nvidia
  nvidia_uvm
  ```

**Enable LXC conf:**
- Open: 
  ```bash
  vim /etc/pve/lxc/9090.conf
  ```
  - Add the following lines to the file:

  ```
  lxc.cgroup2.devices.allow: c 195:* rwm
  lxc.cgroup2.devices.allow: c 507:* rwm
  lxc.cgroup2.devices.allow: c 243:* rwm
  lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file
  lxc.mount.entry: /dev/nvidia1 dev/nvidia1 none bind,optional,create=file
  lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file
  lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file
  lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file
  ```

**Install NV driver and cuda toolkits inside LXC (Ubuntu 22.04):**
Push the cuda file into the container:
```
# enter push cmd
```

Install Driver without kernel header


Install cuda


**Install nvidia container toolkits inside LXC (Ubuntu 22.04):**
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```


**Test inside LXC:**
```
# nvidia-smi

+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.42.06              Driver Version: 555.42.06      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        On  |   00000000:01:00.0 Off |                  N/A |
|  0%   38C    P8             31W /  370W |       4MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 3090        On  |   00000000:03:00.0 Off |                  N/A |
|  0%   44C    P8             17W /  350W |       4MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

Test Run with gpu hello.cu
```
#include <iostream>
#include <cuda_runtime.h>

__global__ void calculate(int *results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 20) {
        results[idx] = idx * idx + idx;
    }
}

int main() {
    const int arraySize = 20;
    int results[arraySize];

    // Allocate memory on the GPU
    int *d_results;
    cudaMalloc(&d_results, arraySize * sizeof(int));

    // Define the number of threads and blocks
    int blockSize = 20;
    int numBlocks = (arraySize + blockSize - 1) / blockSize;

    // Launch the kernel
    calculate<<<numBlocks, blockSize>>>(d_results);

    // Copy the results back to the CPU
    cudaMemcpy(results, d_results, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < arraySize; ++i) {
        std::cout << (i + 1) << "*" << (i + 1) << "+" << (i + 1) << " = " << results[i] << std::endl;
    }

    // Free GPU memory
    cudaFree(d_results);

    return 0;
}
```

Verify with docker gpu hello world
```
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```
